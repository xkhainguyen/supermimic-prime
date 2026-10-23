"""Does the policy gradient BPTT computes through the world model match the simulator's?

Objective, identical for both systems (only the dynamics differ):
  J = sum_{t<H} gamma^t c_t r_{t+1} + gamma^H c_H V(s_H)
closed loop with the trained policy (no observation or action noise), analytic reward and soft
termination applied to the system's own states, the run's critic for the bootstrap.

World model: dJ/dtheta by autograd (BPTT), and dJ/da_t for an additive perturbation of the action at
step t (the policy keeps acting afterwards, so this is the closed-loop adjoint).
Simulator (mjlab, MuJoCo-Warp): dJ/da_t by central finite differences, one world per perturbed
rollout; then dJ/dtheta = sum_t (dJ/da_t)^T da_t/dtheta along the simulator's nominal trajectory.

Usage: python tools/apg_compare.py --ckpt logs/lafan1_main/last.pt --contact_model legacy
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import torch

import supermimic.networks as networks
from supermimic.env import TrackingSim
from supermimic.kinematics import G1Kinematics
from supermimic.mdp import CRITIC_EXTRA, INV_DIM, MDP, OBS_DIM, normalize_xi
from supermimic.motion import MotionLib
from supermimic.networks import Critic, Policy, WorldModel
from supermimic.robot import default_joint_pos_list
from supermimic.rotations import quat_to_mat

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="logs/lafan1_main/last.pt")
ap.add_argument("--motion_dir", default="data/lafan1_g1")
ap.add_argument("--contact_model", default="legacy", choices=["paper", "legacy"])
ap.add_argument("--clips", nargs="+", default=["walk1_subject1", "run1_subject2", "dance2_subject1", "fallAndGetUp1_subject1"])
ap.add_argument("--start_s", type=float, default=3.0, help="seconds of closed-loop policy rollout before the start state")
ap.add_argument("--H", type=int, default=32)
ap.add_argument("--eps", type=float, nargs="+", default=[1e-2, 1e-3])
ap.add_argument("--out", default="results/apg_compare.json")
a = ap.parse_args()

dev = "cuda:0"
ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
cargs = ck["args"]
networks.AMP_DTYPE = None  # fp32 for both systems
torch.backends.cuda.matmul.allow_tf32 = False
K, Hh, H, gamma = cargs["goal_frames"], cargs["hist"], a.H, cargs["gamma"]
kin = G1Kinematics(dev)
motion = MotionLib.load(a.motion_dir, kin, dev)
mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=dev))
wm = WorldModel().to(dev)
wm.load_state_dict(ck["wm"])
pi = Policy(OBS_DIM, 73 * K, hist=Hh).to(dev)
pi.load_state_dict(ck["pi"])
critic = Critic(INV_DIM + CRITIC_EXTRA + 73 * K).to(dev)
critic.load_state_dict(ck["critic"])
for m in (wm, pi, critic):
    m.eval()
for p in list(wm.parameters()) + list(critic.parameters()):
    p.requires_grad_(False)
M = len(a.clips)
clip_ids = [motion.names.index(c) for c in a.clips]


def objective_terms(S_list, f_list, A_list, Ap_list, xi_n, S_H, f_H):
    """J from a state/action trajectory (lists over t of batched tensors)."""
    B = S_list[0][0].shape[0]
    flat = lambda xs: torch.cat(xs, 0)  # noqa: E731
    S_all = tuple(flat([St[i] for St in S_list]) for i in range(6))
    r = mdp.reward(S_all, flat(f_list), flat(A_list), flat(Ap_list)).view(H, B)
    d = mdp.soft_termination(S_all, flat(f_list)).view(H, B)
    alive = torch.cumprod(1 - d, 0)
    c_t = torch.cat((torch.ones(1, B, device=dev), alive[:-1]), 0)
    disc = gamma ** torch.arange(H, device=dev, dtype=torch.float32).unsqueeze(-1)
    V = critic(torch.cat((mdp.critic_features(S_H, f_H, xi_n), motion.goal(f_H, K)), -1))
    return (disc * c_t * r).sum(0) + gamma**H * alive[-1] * V


# ------------------------------------------------------------------------------------------------ start states
base = TrackingSim(M, motion, mdp, eval_mode=True, device=dev, contact_model=a.contact_model)
base.reset(torch.arange(M, device=dev), motion.clip_start[torch.tensor(clip_ids, device=dev)])
hist = None
with torch.no_grad():
    for _ in range(int(a.start_s / 0.02)):
        o = mdp.obs(base.state(), base.a_prev, base.xi, base.f)
        hist = o.unsqueeze(1).repeat(1, Hh, 1) if hist is None else torch.cat((hist[:, 1:], o.unsqueeze(1)), 1)
        base.step(pi(hist, motion.goal(base.f, K)))
    o = mdp.obs(base.state(), base.a_prev, base.xi, base.f)
    hist = torch.cat((hist[:, 1:], o.unsqueeze(1)), 1)
qpos0, qvel0 = base.sim.data.qpos.clone(), base.sim.data.qvel.clone()
f0, a_prev0, xi = base.f.clone(), base.a_prev.clone(), base.xi.clone()
xi_n = normalize_xi(xi)
print("start states: terminated during warm-up", base.failed_eval.tolist(), "frames:", (f0 - motion.clip_start[torch.tensor(clip_ids, device=dev)]).tolist())
del base

# ------------------------------------------------------------------------------------------------ simulator, batched
# world layout per start m: [nominal] + for eps, t, j, sign
pert = [(e, t, j, s) for e in a.eps for t in range(H) for j in range(29) for s in (1, -1)]
P = 1 + len(pert)
W = M * P
sim = TrackingSim(W, motion, mdp, eval_mode=True, device=dev, contact_model=a.contact_model)
rep = torch.arange(M, device=dev).repeat_interleave(P)
sim.sim.data.qpos[:] = qpos0[rep]
sim.sim.data.qvel[:] = qvel0[rep]
sim.f, sim.a_prev = f0[rep].clone(), a_prev0[rep].clone()
sim.done_eval[:] = False
sim.failed_eval[:] = False
delta = torch.zeros(W, H, 29, device=dev)
for m in range(M):
    for k, (e, t, j, s) in enumerate(pert):
        delta[m * P + 1 + k, t, j] = s * e
h = hist[rep].clone()
S_list, f_list, A_list, Ap_list, H_nom = [], [], [], [], []
with torch.no_grad():
    for t in range(H):
        f_t = sim.f.clone()
        act = pi(h, motion.goal(f_t, K)) + delta[:, t]
        H_nom.append(h[torch.arange(M, device=dev) * P].clone())
        Ap_list.append(sim.a_prev.clone()), A_list.append(act)
        sim.step(act)
        S = sim.state()
        S_list.append(S), f_list.append(sim.f.clone())
        o = mdp.obs(S, act, sim.xi, sim.f)
        h = torch.cat((h[:, 1:], o.unsqueeze(1)), 1)
    J_sim = objective_terms(S_list, f_list, A_list, Ap_list, xi_n[rep], S_list[-1], f_list[-1])
J_sim = J_sim.view(M, P)
G_sim = {e: torch.zeros(M, H, 29, device=dev) for e in a.eps}
for k, (e, t, j, s) in enumerate(pert):
    G_sim[e][:, t, j] += s * J_sim[:, 1 + k] / (2 * e)
nominal_idx = torch.arange(M, device=dev) * P
print(f"simulator: {W} worlds; nominal J per start: {[round(x, 4) for x in J_sim[:, 0].tolist()]}; "
      f"nominal terminated: {sim.failed_eval[nominal_idx].tolist()}")  # fmt: skip

# simulator policy gradient: sum_t G_t^T da_t/dtheta along the nominal trajectory
params = [p for p in pi.parameters()]


def flat_grad(gs):
    return torch.cat([g.reshape(-1) if g is not None else torch.zeros(p.numel(), device=dev) for g, p in zip(gs, params)])


ref_eps = a.eps[-1]
g_sim = []
for m in range(M):
    acts = torch.stack([pi(H_nom[t][m : m + 1], motion.goal(f_list[t - 1][nominal_idx[m] : nominal_idx[m] + 1] if t else f0[m : m + 1], K))[0] for t in range(H)])
    g_sim.append(flat_grad(torch.autograd.grad(acts, params, G_sim[ref_eps][m])))

# ------------------------------------------------------------------------------------------------ world model, BPTT
# start state of the WM rollout = simulator state before step 0

R0 = quat_to_mat(qpos0[:, 3:7])
w0 = (R0 @ qvel0[:, 3:6].unsqueeze(-1)).squeeze(-1)
qadr, vadr = sim.q_adr, sim.v_adr
S0 = (qpos0[:, 0:3], R0, qvel0[:, 0:3], w0, qpos0[:, qadr], qvel0[:, vadr])
g_wm, G_wm = [], []
for m in range(M):
    dl = torch.zeros(H, 1, 29, device=dev, requires_grad=True)
    S = tuple(x[m : m + 1] for x in S0)
    hm, ap, f = hist[m : m + 1], a_prev0[m : m + 1], f0[m : m + 1]
    Sl, fl, Al, Apl = [], [], [], []
    for t in range(H):
        act = pi(hm, motion.goal(f, K)) + dl[t]
        Apl.append(ap), Al.append(act)
        S = wm.step(S, act, xi_n[m : m + 1])
        f = f + 1
        Sl.append(S), fl.append(f)
        o = mdp.obs(S, act, xi[m : m + 1], f)
        hm = torch.cat((hm[:, 1:], o.unsqueeze(1)), 1)
        ap = act
    J = objective_terms(Sl, fl, Al, Apl, xi_n[m : m + 1], Sl[-1], fl[-1])[0]
    gp = torch.autograd.grad(J, params + [dl])
    g_wm.append(flat_grad(gp[:-1]))
    G_wm.append(gp[-1][:, 0])
    print(f"start {m} ({a.clips[m]}): J_wm {float(J):.4f}  J_sim {float(J_sim[m, 0]):.4f}")


# ------------------------------------------------------------------------------------------------ report
def cos(x, y):
    return float((x * y).sum() / (x.norm() * y.norm() + 1e-20))


res = {"ckpt": a.ckpt, "clips": a.clips, "H": H, "eps": a.eps, "starts": []}
print(f"\nFD consistency: rel. diff of dJ/da between eps {a.eps[0]:g} and {ref_eps:g}, per start: "
      + " ".join(f"{float((G_sim[a.eps[0]][m] - G_sim[ref_eps][m]).norm() / G_sim[ref_eps][m].norm()):.3f}" for m in range(M)))  # fmt: skip
for m in range(M):
    per_t = [cos(G_wm[m][t], G_sim[ref_eps][m][t]) for t in range(H)]
    r = {"clip": a.clips[m], "cos_param_grad": cos(g_wm[m], g_sim[m]),
         "norm_param_grad_wm": float(g_wm[m].norm()), "norm_param_grad_sim": float(g_sim[m].norm()),
         "cos_action_adjoint_all": cos(G_wm[m], G_sim[ref_eps][m]), "cos_action_adjoint_per_t": per_t,
         "norm_adjoint_per_t_wm": G_wm[m].norm(dim=-1).tolist(), "norm_adjoint_per_t_sim": G_sim[ref_eps][m].norm(dim=-1).tolist()}  # fmt: skip
    res["starts"].append(r)
    print(f"\n{a.clips[m]}: dJ/dtheta cosine WM vs sim {r['cos_param_grad']:.3f}  |wm| {r['norm_param_grad_wm']:.3g}  |sim| {r['norm_param_grad_sim']:.3g}"
          f"  | dJ/da (all t) cosine {r['cos_action_adjoint_all']:.3f}")  # fmt: skip
    print("   dJ/da_t cosine by t: " + " ".join(f"{x:.2f}" for x in per_t))
    print("   |dJ/da_t| wm/sim:    " + " ".join(f"{w / max(s, 1e-12):.2f}" for w, s in zip(r["norm_adjoint_per_t_wm"], r["norm_adjoint_per_t_sim"])))
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(res, open(a.out, "w"), indent=1)
print("wrote", a.out)
