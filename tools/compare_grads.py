"""Compare world-model Jacobians with simulator Jacobians around a foot touchdown.

The simulator is CPU MuJoCo (float64) configured like the LAFAN1 training run of the checkpoint
(old contact model: every robot geom collides with the floor, condim 3, no self-collisions; friction
1.0, no domain randomization). MuJoCo has no analytic step gradients here, so its Jacobians are central
finite differences at several step sizes; the world model's are autograd (fp32).

Procedure: run the trained policy in the simulator on a walking clip, pick a step where one foot is in
the air and touches down within the window, and from that exact state roll both systems forward on the
same recorded actions. Report d s_k / d a_0 and d s_k / d s_0 for k = 1..N per state block.

Usage: python tools/compare_grads.py --ckpt logs/lafan1_main/ckpt_016000.pt --clip walk1_subject1
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import mujoco
import numpy as np
import torch
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_ARTICULATION, KNEES_BENT_KEYFRAME, get_spec
from mjlab.entity import EntityCfg
from mjlab.scene import Scene, SceneCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.spec_config import CollisionCfg

import supermimic.networks as networks
from supermimic.env import DECIMATION, make_sim_cfg
from supermimic.kinematics import G1Kinematics
from supermimic.mdp import MDP, OBS_DIM, normalize_xi
from supermimic.motion import MotionLib
from supermimic.networks import Policy, WorldModel
from supermimic.robot import G1_JOINT_NAMES, action_scale, default_joint_pos_list
from supermimic.rotations import quat_to_mat, so3_log

BLOCKS = {"root pos": slice(0, 3), "root rot": slice(3, 6), "root lin vel": slice(6, 9), "root ang vel": slice(9, 12),
          "joint pos": slice(12, 41), "joint vel": slice(41, 70)}  # fmt: skip


# ------------------------------------------------------------------------------------------------ simulator
class CpuSim:
    def __init__(self):
        old = CollisionCfg(geom_names_expr=(".*_collision",), contype=0, conaffinity=1, condim=3, priority=1, friction=(1.0,))
        cfg = EntityCfg(init_state=KNEES_BENT_KEYFRAME, collisions=(old,), spec_fn=get_spec, articulation=G1_ARTICULATION)
        scene = Scene(SceneCfg(num_envs=1, terrain=TerrainEntityCfg(terrain_type="plane"), entities={"robot": cfg}), device="cpu")
        self.m = scene.spec.compile()
        make_sim_cfg().mujoco.apply(self.m)
        self.d = mujoco.MjData(self.m)
        self.qadr = np.array([self.m.joint("robot/" + n).qposadr[0] for n in G1_JOINT_NAMES])
        self.vadr = np.array([self.m.joint("robot/" + n).dofadr[0] for n in G1_JOINT_NAMES])
        jid = {self.m.joint("robot/" + n).id: i for i, n in enumerate(G1_JOINT_NAMES)}
        self.act_joint = np.array([jid[self.m.actuator_trnid[a, 0]] for a in range(self.m.nu)])  # actuator -> G1 joint index
        self.default_q = np.array(default_joint_pos_list())
        self.alpha = np.array([action_scale(n) for n in G1_JOINT_NAMES])
        floor = self.m.geom("terrain").id
        self.floor = floor
        self.foot_body = {s: self.m.body(f"robot/{s}_ankle_roll_link").id for s in ("left", "right")}

    def set_state(self, x: np.ndarray):
        """x = (qpos, qvel) of the whole model."""
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[:], self.d.qvel[:] = x[: self.m.nq], x[self.m.nq :]
        mujoco.mj_forward(self.m, self.d)

    def get_state(self) -> np.ndarray:
        return np.concatenate((self.d.qpos, self.d.qvel)).copy()

    def step(self, a: np.ndarray):
        """One 50 Hz control step with action a (G1 joint order)."""
        target = self.default_q + self.alpha * a
        self.d.ctrl[:] = target[self.act_joint]
        for _ in range(DECIMATION):
            mujoco.mj_step(self.m, self.d)

    def foot_contacts(self) -> dict:
        out = {"left": 0, "right": 0}
        for c in self.d.contact[: self.d.ncon]:
            for g in (c.geom1, c.geom2):
                b = self.m.geom_bodyid[g]
                for s, fb in self.foot_body.items():
                    if b == fb and self.floor in (c.geom1, c.geom2):
                        out[s] += 1
        return out

    def S(self):
        """State tuple (p, R, v, w, q, qd) in the world model's convention (torch, float64)."""
        qp, qv = self.d.qpos, self.d.qvel
        R = quat_to_mat(torch.tensor(qp[3:7]))
        w = R @ torch.tensor(qv[3:6])
        return (torch.tensor(qp[0:3]), R, torch.tensor(qv[0:3]), w, torch.tensor(qp[self.qadr]), torch.tensor(qv[self.vadr]))


def feat(S, R_nom) -> torch.Tensor:
    """70-d state vector: root pos, root rotation as a rotation vector relative to R_nom, root lin/ang vel
    (world), joint pos, joint vel. Differences of this vector are what the Jacobians map to."""
    p, R, v, w, q, qd = S
    return torch.cat((p, so3_log(R_nom.transpose(-1, -2) @ R), v, w, q, qd), -1)


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="logs/lafan1_main/ckpt_016000.pt")
    ap.add_argument("--motion_dir", default="data/lafan1_g1")
    ap.add_argument("--clip", default="walk1_subject1")
    ap.add_argument("--start_s", type=float, default=4.0, help="seconds of policy rollout before searching for a touchdown")
    ap.add_argument("--N", type=int, default=6, help="control steps in the comparison window")
    ap.add_argument("--td", type=int, default=2, help="touchdown happens at this step of the window")
    ap.add_argument("--eps", type=float, nargs="+", default=[1e-2, 1e-3, 1e-4, 1e-5])
    ap.add_argument("--out", default="results/grad_compare.json")
    a = ap.parse_args()
    torch.set_default_dtype(torch.float32)
    dev = "cpu"

    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    networks.AMP_DTYPE = None  # fp32 Jacobians
    wm = WorldModel()
    wm.load_state_dict(ck["wm"])
    wm.eval()
    K, Hh = ck["args"]["goal_frames"], ck["args"]["hist"]
    pi = Policy(OBS_DIM, 73 * K, hist=Hh)
    pi.load_state_dict(ck["pi"])
    pi.eval()
    kin = G1Kinematics(dev)
    motion = MotionLib.load(a.motion_dir, kin, dev)
    mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list()))
    xi = torch.zeros(1, 33)
    xi[:, 0] = 1.0
    xi_n = normalize_xi(xi)
    sim = CpuSim()

    # 1) policy rollout from the clip start; record states, actions, foot contacts
    c = motion.names.index(a.clip)
    f = int(motion.clip_start[c])
    p, R, v, w, q, qd = (x.double() for x in motion.ref_state(torch.tensor([f])))
    from supermimic.rotations import mat_to_quat

    x0 = np.zeros(sim.m.nq + sim.m.nv)
    x0[0:3], x0[3:7] = p[0].numpy(), mat_to_quat(R)[0].numpy()
    x0[7:][: 0] = 0
    qpos = x0[: sim.m.nq]
    qpos[sim.qadr] = q[0].numpy()
    qvel = x0[sim.m.nq :]
    qvel[0:3] = v[0].numpy()
    qvel[3:6] = (R[0].T @ w[0]).numpy()
    qvel[sim.vadr] = qd[0].numpy()
    sim.set_state(x0)
    hist, a_prev = None, torch.zeros(1, 29)
    states, actions, contacts = [], [], []
    T = int(a.start_s / 0.02) + 60
    for t in range(T):
        S = tuple(x.float().unsqueeze(0) for x in sim.S())
        o = mdp.obs(S, a_prev, xi, torch.tensor([f + t]))
        hist = o.unsqueeze(1).repeat(1, Hh, 1) if hist is None else torch.cat((hist[:, 1:], o.unsqueeze(1)), 1)
        with torch.no_grad():
            act = pi(hist, motion.goal(torch.tensor([f + t]), K))
        states.append(sim.get_state()), contacts.append(sim.foot_contacts()), actions.append(act[0].double().numpy())
        sim.step(act[0].double().numpy())
        a_prev = act
    # 2) window start: one foot airborne at t0 and touching down within the next N steps
    t0 = None
    for t in range(int(a.start_s / 0.02), T - a.N):
        for s in ("left", "right"):
            if all(contacts[t + k][s] == 0 for k in range(a.td)) and contacts[t + a.td][s] > 0:
                t0, side = t, s
                break
        if t0 is not None:
            break
    assert t0 is not None, "no touchdown found"
    td = next(k for k in range(1, a.N) if contacts[t0 + k][side] > 0)
    print(f"window: clip {a.clip}, t0 = {t0} ({t0 * 0.02:.2f} s), {side} foot airborne, touchdown at step +{td}, N = {a.N}")
    print("foot contacts per step:", [(contacts[t0 + k]["left"], contacts[t0 + k]["right"]) for k in range(a.N + 1)])
    acts = np.stack(actions[t0 : t0 + a.N])
    x_start = states[t0]

    # 3) simulator rollout and finite-difference Jacobians (float64)
    def sim_roll(x, A):
        sim.set_state(x)
        out = []
        for k in range(a.N):
            sim.step(A[k])
            out.append(sim.S())
        return out

    nom = sim_roll(x_start, acts)
    R_nom = [S[1] for S in nom]
    F_nom = [feat(S, Rn) for S, Rn in zip(nom, R_nom)]

    def sim_jac_action(eps):
        J = np.zeros((a.N, 70, 29))
        for j in range(29):
            Ap, Am = acts.copy(), acts.copy()
            Ap[0, j] += eps
            Am[0, j] -= eps
            sp, sm = sim_roll(x_start, Ap), sim_roll(x_start, Am)
            for k in range(a.N):
                J[k, :, j] = ((feat(sp[k], R_nom[k]) - feat(sm[k], R_nom[k])) / (2 * eps)).numpy()
        return J

    # initial-state Jacobian w.r.t. the 70-d feature of s_0 (perturb root pos / vel, joint pos / vel; rotation
    # perturbed through the free-joint quaternion)
    sim.set_state(x_start)
    S0 = sim.S()
    R0 = S0[1]

    def x_from_feat_delta(dx):
        x = x_start.copy()
        x[0:3] += dx[0:3]
        Rn = R0 @ _exp(torch.tensor(dx[3:6]))
        x[3:7] = mat_to_quat(Rn).numpy()
        v_w = S0[2].numpy() + dx[6:9]
        w_w = S0[3].numpy() + dx[9:12]
        x[sim.m.nq : sim.m.nq + 3] = v_w
        x[sim.m.nq + 3 : sim.m.nq + 6] = (Rn.T @ torch.tensor(w_w)).numpy()
        x[sim.qadr] += dx[12:41]
        x[sim.m.nq + sim.vadr] += dx[41:70]
        return x

    def sim_jac_state(eps):
        J = np.zeros((a.N, 70, 70))
        for j in range(70):
            d = np.zeros(70)
            d[j] = eps
            sp, sm = sim_roll(x_from_feat_delta(d), acts), sim_roll(x_from_feat_delta(-d), acts)
            for k in range(a.N):
                J[k, :, j] = ((feat(sp[k], R_nom[k]) - feat(sm[k], R_nom[k])) / (2 * eps)).numpy()
        return J

    J_sim_a = {eps: sim_jac_action(eps) for eps in a.eps}
    J_sim_s = {eps: sim_jac_state(eps) for eps in a.eps}

    # 4) world model rollout and autograd Jacobians (fp32)
    S0f = tuple(x.float().unsqueeze(0) for x in S0)
    A_t = torch.tensor(acts, dtype=torch.float32)

    def wm_roll(S, A):
        out = []
        for k in range(a.N):
            S = wm.step(S, A[k : k + 1], xi_n)
            out.append(S)
        return out

    with torch.no_grad():
        wm_nom = wm_roll(S0f, A_t)
    R_wm = [S[1] for S in wm_nom]

    def wm_feats_from_a0(a0):
        A = torch.cat((a0.unsqueeze(0), A_t[1:]), 0)
        return torch.stack([feat(S, Rn)[0] for S, Rn in zip(wm_roll(S0f, A), R_wm)])

    def wm_feats_from_dx(dx):
        p, R, v, w, q, qd = S0f
        S = (p + dx[0:3], R @ _exp(dx[3:6]), v + dx[6:9], w + dx[9:12], q + dx[12:41], qd + dx[41:70])
        return torch.stack([feat(Sk, Rn)[0] for Sk, Rn in zip(wm_roll(S, A_t), R_wm)])

    J_wm_a = torch.autograd.functional.jacobian(wm_feats_from_a0, A_t[0]).numpy()  # (N, 70, 29)
    J_wm_s = torch.autograd.functional.jacobian(wm_feats_from_dx, torch.zeros(70)).numpy()  # (N, 70, 70)

    # 5) report
    pred_err = [float((feat(wm_nom[k], R_nom[k].float())[0] - F_nom[k].float()).abs().max()) for k in range(a.N)]
    res = {"clip": a.clip, "t0": t0, "side": side, "touchdown_step": td, "N": a.N, "eps": a.eps,
           "contacts": [(contacts[t0 + k]["left"], contacts[t0 + k]["right"]) for k in range(a.N + 1)],
           "wm_open_loop_max_abs_state_err": pred_err, "d_s/d_a0": {}, "d_s/d_s0": {}}  # fmt: skip
    for name, Jw, Js in (("d_s/d_a0", J_wm_a, J_sim_a), ("d_s/d_s0", J_wm_s, J_sim_s)):
        print(f"\n=== {name}   (cos: cosine similarity WM vs sim; rel: ||J_wm - J_sim|| / ||J_sim||; sim = FD at eps {a.eps[1]:g})")
        print("FD consistency across eps (||J_eps - J_1e-3|| / ||J_1e-3||, all blocks), per step:")
        for eps in a.eps:
            r = [np.linalg.norm(Js[eps][k] - Js[a.eps[1]][k]) / (np.linalg.norm(Js[a.eps[1]][k]) + 1e-12) for k in range(a.N)]
            print(f"   eps {eps:g}: " + " ".join(f"{x:6.3f}" for x in r))
        J_ref = Js[a.eps[1]]
        hdr = f"{'block':14s}" + "".join(f"  k={k + 1}:cos/rel/|wm|/|sim|" for k in range(a.N))
        print(hdr)
        res[name] = {}
        for b, sl in BLOCKS.items():
            row, rec = f"{b:14s}", []
            for k in range(a.N):
                jw, js = Jw[k][sl].ravel(), J_ref[k][sl].ravel()
                cos = float(jw @ js / (np.linalg.norm(jw) * np.linalg.norm(js) + 1e-12))
                rel = float(np.linalg.norm(jw - js) / (np.linalg.norm(js) + 1e-12))
                rec.append({"cos": cos, "rel": rel, "norm_wm": float(np.linalg.norm(jw)), "norm_sim": float(np.linalg.norm(js))})
                row += f"  {cos:5.2f}/{rel:5.2f}/{np.linalg.norm(jw):6.2f}/{np.linalg.norm(js):6.2f}"
            print(row)
            res[name][b] = rec
    print("\nWM open-loop max |state error| per step:", " ".join(f"{x:.4f}" for x in pred_err))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("wrote", a.out)


def _exp(v):
    from supermimic.rotations import so3_exp

    return so3_exp(v.float() if v.dtype != torch.float64 else v).to(v.dtype)


if __name__ == "__main__":
    main()
