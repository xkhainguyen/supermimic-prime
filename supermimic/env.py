"""Motion-tracking simulator on mjlab (MuJoCo-Warp, CUDA-graph stepping).

The environment is a thin batched wrapper: it applies the joint setpoint, steps MuJoCo-Warp, labels
the transition with the analytic reward/termination of :mod:`supermimic.mdp`, and resets episodes to
reference-motion states drawn by return-based bin sampling (paper Appendix D.1).

State is read from qpos/qvel directly (free-joint linear velocity is world-frame, angular velocity
is body-frame), so no extra forward() is needed after a step or a reset.
"""

from __future__ import annotations

import torch
from mjlab.managers.event_manager import RecomputeLevel
from mjlab.scene import Scene, SceneCfg
from mjlab.sim import MujocoCfg, Simulation, SimulationCfg
from mjlab.terrains import TerrainEntityCfg

from .mdp import MAX_STEP_REWARD, MDP, flatten_state
from .motion import CONTROL_DT, MotionLib
from .robot import FOOT_GEOMS, G1_JOINT_NAMES, action_scale, g1_cfg
from .rotations import mat_to_quat, quat_to_mat, so3_exp

PHYSICS_DT = 0.005
DECIMATION = round(CONTROL_DT / PHYSICS_DT)


def make_sim_cfg(nconmax: int = 64, njmax: int = 512) -> SimulationCfg:
    """mjlab's G1 tracking-task solver settings; contact/constraint capacity raised because self-collisions
    during falls and floor work exceed the task's njmax=250 (overflowing worlds silently drop constraints)."""
    return SimulationCfg(
        nconmax=nconmax,
        njmax=njmax,
        mujoco=MujocoCfg(timestep=PHYSICS_DT, integrator="implicitfast", cone="pyramidal", iterations=10, ls_iterations=20),
    )


class TrackingSim:
    """Batched G1 motion-tracking simulator."""

    def __init__(self, num_envs: int, motion: MotionLib, mdp: MDP, gamma: float = 0.99, eval_mode: bool = False,
                 max_episode_steps: int = 500, reset_noise: bool = True, device: str = "cuda:0",
                 sim_cfg: SimulationCfg | None = None, contact_model: str = "paper", auto_reset: bool = True,
                 sampler: str = "completion"):  # fmt: skip
        """contact_model "legacy" reproduces the first LAFAN1 run's contacts (see robot.LEGACY_COLLISION);
        auto_reset False leaves finished episodes running (for paired rollouts)."""
        self.device = dev = device
        self.auto_reset = auto_reset
        # "completion": a bin's score is the discounted return earned from the start frame divided by the
        # largest attainable before the clip ends; "vibed": raw return with an infinite-horizon mean-reward
        # bootstrap, which scores short clip-end episodes far below their true difficulty
        self.sampler = sampler
        scene_cfg = SceneCfg(num_envs=num_envs, terrain=TerrainEntityCfg(terrain_type="plane"), entities={"robot": g1_cfg(contact_model)})
        self.scene = Scene(scene_cfg, device=dev)
        if sim_cfg is None:
            sim_cfg = make_sim_cfg() if contact_model == "paper" else make_sim_cfg(nconmax=128, njmax=500)
        self.sim = Simulation(num_envs=num_envs, cfg=sim_cfg, spec=self.scene.spec, device=dev)
        self.scene.initialize(mj_model=self.sim.mj_model, model=self.sim.model, data=self.sim.data)
        self.robot = self.scene["robot"]
        self.jid = torch.tensor(self.robot.find_joints(G1_JOINT_NAMES, preserve_order=True)[0], device=dev)
        assert [self.robot.joint_names[i] for i in self.jid.tolist()] == G1_JOINT_NAMES
        idx = self.robot.indexing
        self.q_adr = idx.joint_q_adr[self.jid]
        self.v_adr = idx.joint_v_adr[self.jid]
        self.root_q_adr, self.root_v_adr = idx.free_joint_q_adr, idx.free_joint_v_adr
        self.N = num_envs
        self.motion, self.mdp, self.gamma = motion, mdp, gamma
        self.eval_mode = eval_mode
        self.max_episode_steps = max_episode_steps if not eval_mode else 10**9
        self.reset_noise = reset_noise and not eval_mode
        self.default_q = mdp.default_q.clone()
        self.alpha = torch.tensor([action_scale(n) for n in G1_JOINT_NAMES], device=dev)

        # ---- sim context xi = [friction, joint offsets(29), torso com offset(3)] ----
        self.sim.expand_model_fields(("geom_friction", "body_ipos"))
        local_geoms = self.robot.find_geoms(FOOT_GEOMS if contact_model == "paper" else ".*_collision")[0]
        self.geom_ids = idx.geom_ids[torch.tensor(local_geoms, device=dev)]
        self.torso_bid = int(idx.body_ids[self.robot.find_bodies("torso_link")[0][0]])
        # per-episode sim context (paper App. A, Table A.3), resampled in reset()
        self.xi = torch.zeros(self.N, 33, device=dev)
        self.xi[:, 0] = 1.0

        # ---- episode bookkeeping ----
        self.f = torch.zeros(self.N, dtype=torch.long, device=dev)
        self.ep_step = torch.zeros(self.N, dtype=torch.long, device=dev)
        self.a_prev = torch.zeros(self.N, 29, device=dev)
        self.start_bin = torch.zeros(self.N, dtype=torch.long, device=dev)
        self.ep_disc_ret = torch.zeros(self.N, device=dev)
        self.ep_ret = torch.zeros(self.N, device=dev)
        self.start_f = torch.zeros(self.N, dtype=torch.long, device=dev)
        self.G_max = MAX_STEP_REWARD / (1 - gamma)
        self.bin_G = torch.full((motion.num_bins,), 0.0, device=dev)
        self.bin_w_avg = torch.full((motion.num_bins,), 1.0 / motion.num_bins, device=dev)
        self.done_eval = torch.zeros(self.N, dtype=torch.bool, device=dev)
        self.failed_eval = torch.zeros(self.N, dtype=torch.bool, device=dev)
        self.reset(torch.arange(self.N, device=dev))

    # ------------------------------------------------------------------ sim context
    def _sample_context(self, env_ids: torch.Tensor):
        n, dev = len(env_ids), self.device
        self.xi[env_ids, 0] = torch.empty(n, device=dev).uniform_(0.3, 1.2)
        self.xi[env_ids, 1:30] = torch.empty(n, 29, device=dev).uniform_(-0.01, 0.01)
        self.xi[env_ids, 30:33] = torch.stack(
            (
                torch.empty(n, device=dev).uniform_(-0.025, 0.025),
                torch.empty(n, device=dev).uniform_(-0.05, 0.05),
                torch.empty(n, device=dev).uniform_(-0.05, 0.05),
            ),
            -1,
        )

    def _apply_context(self, env_ids: torch.Tensor):
        """Write foot friction (shared by the foot geoms, which out-prioritize the ground) and torso CoM offset."""
        model = self.sim.model
        model.geom_friction[env_ids[:, None], self.geom_ids[None, :], 0] = self.xi[env_ids, :1]
        ipos0 = self.sim.get_default_field("body_ipos")[self.torso_bid]
        model.body_ipos[env_ids, self.torso_bid] = ipos0 + self.xi[env_ids, 30:33]
        self.sim.recompute_constants(RecomputeLevel.set_const)

    # ------------------------------------------------------------------ state
    def state(self):
        qpos, qvel = self.sim.data.qpos, self.sim.data.qvel
        root = qpos[:, self.root_q_adr]
        R = quat_to_mat(root[:, 3:7])  # mujoco quats are wxyz
        rv = qvel[:, self.root_v_adr]
        w = (R @ rv[:, 3:6].unsqueeze(-1)).squeeze(-1)
        return (root[:, :3].clone(), R, rv[:, :3].clone(), w, qpos[:, self.q_adr].clone(), qvel[:, self.v_adr].clone())

    # ------------------------------------------------------------------ reset
    def sample_frames(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        m = self.motion
        soft = torch.softmax(-25.0 * (self.bin_G if self.sampler == "completion" else self.bin_G / self.G_max), 0)
        self.bin_w_avg.mul_(0.999).add_(0.001 * soft)
        probs = 0.5 / m.num_bins + 0.4 * soft + 0.1 * self.bin_w_avg / self.bin_w_avg.sum()
        bins = torch.multinomial(probs, n, replacement=True)
        start = m.bin_start[bins]
        end = torch.minimum(m.bin_end[bins] if self.sampler == "completion" else start + 50, m.clip_end[m.bin_clip[bins]])  # >= 1 step left
        f = start + (torch.rand(n, device=self.device) * (end - start).float()).long()
        return f, bins

    def reset(self, env_ids: torch.Tensor, frames: torch.Tensor | None = None):
        if len(env_ids) == 0:
            return
        n = len(env_ids)
        dev = self.device
        if frames is None:
            frames, bins = self.sample_frames(n)
        else:
            bins = self.motion.frame_bin[frames]
        m = self.motion
        p, R, v, w, q, qd = (x.clone() for x in m.ref_state(frames))
        if self.reset_noise:
            p = p + torch.stack(
                (torch.empty(n, device=dev).uniform_(-0.05, 0.05), torch.empty(n, device=dev).uniform_(-0.05, 0.05),
                 torch.empty(n, device=dev).uniform_(-0.01, 0.01)), -1)  # fmt: skip
            rv = torch.stack(
                (torch.empty(n, device=dev).uniform_(-0.1, 0.1), torch.empty(n, device=dev).uniform_(-0.1, 0.1),
                 torch.empty(n, device=dev).uniform_(-0.2, 0.2)), -1)  # fmt: skip
            R = so3_exp(rv) @ R
            v = v + torch.stack(
                (torch.empty(n, device=dev).uniform_(-0.5, 0.5), torch.empty(n, device=dev).uniform_(-0.5, 0.5),
                 torch.empty(n, device=dev).uniform_(-0.2, 0.2)), -1)  # fmt: skip
            w = w + torch.stack(
                (torch.empty(n, device=dev).uniform_(-0.52, 0.52), torch.empty(n, device=dev).uniform_(-0.52, 0.52),
                 torch.empty(n, device=dev).uniform_(-0.78, 0.78)), -1)  # fmt: skip
            q = q + torch.empty_like(q).uniform_(-0.1, 0.1)
        q = torch.clamp(q, self.mdp.kin.q_lo, self.mdp.kin.q_hi)
        if not self.eval_mode:
            self._sample_context(env_ids)
        self._apply_context(env_ids)
        self.sim.reset(env_ids)
        self.robot.reset(env_ids)
        self.robot.write_root_link_pose_to_sim(torch.cat((p, mat_to_quat(R)), -1), env_ids=env_ids)
        self.robot.write_root_link_velocity_to_sim(torch.cat((v, w), -1), env_ids=env_ids)
        self.robot.write_joint_state_to_sim(q, qd, joint_ids=self.jid, env_ids=env_ids)
        self.robot.set_joint_position_target(q, joint_ids=self.jid, env_ids=env_ids)
        self.f[env_ids] = frames
        self.ep_step[env_ids] = 0
        self.a_prev[env_ids] = 0.0
        self.start_bin[env_ids] = bins
        self.ep_disc_ret[env_ids] = 0.0
        self.ep_ret[env_ids] = 0.0
        self.start_f[env_ids] = frames

    # ------------------------------------------------------------------ step
    def step(self, a: torch.Tensor) -> dict:
        S = self.state()
        f = self.f.clone()
        xi = self.xi.clone()
        a_prev = self.a_prev.clone()
        ep_step = self.ep_step.clone()
        target = self.default_q + xi[:, 1:30] + self.alpha * a
        self.robot.set_joint_position_target(target, joint_ids=self.jid)
        for _ in range(DECIMATION):
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(PHYSICS_DT)
        S1 = self.state()
        f1 = f + 1
        r, terms = self.mdp.reward(S1, f1, a, a_prev, terms=True)
        term = self.mdp.termination(S1, f1)
        clip_end = self.motion.clip_end[self.motion.clip_id[f]]
        timeout = (f1 >= clip_end) | (ep_step + 1 >= self.max_episode_steps)
        done = term | timeout
        out = {
            "s": flatten_state(S), "a": a.clone(), "a_prev": a_prev, "f": f, "s1": flatten_state(S1), "r": r,
            "term": term, "done": done, "ep_step": ep_step, "xi": xi, "terms": terms,
        }  # fmt: skip
        # bookkeeping
        self.ep_disc_ret += (self.gamma ** ep_step.float()) * r
        self.ep_ret += r
        self.f = f1
        self.ep_step = ep_step + 1
        self.a_prev = a.clone()
        if self.eval_mode:
            self.failed_eval |= term & ~self.done_eval
            self.done_eval |= done
            return out
        if not self.auto_reset:
            return out
        ids = done.nonzero(as_tuple=False).squeeze(-1)
        if len(ids) > 0:
            # bin return estimate; truncated episodes bootstrap with their mean reward
            T = self.ep_step[ids].float()
            if self.sampler == "completion":
                f_s = self.start_f[ids]
                L = (self.motion.clip_end[self.motion.clip_id[f_s]] - f_s).float()  # steps available before the clip ends
                g = self.gamma
                rest = torch.clamp(L - T, min=0)
                boot = torch.where(term[ids], 0.0, g**T * (self.ep_ret[ids] / T) * (1 - g**rest) / (1 - g))
                G = (self.ep_disc_ret[ids] + boot) / (MAX_STEP_REWARD * (1 - g**L) / (1 - g)).clamp(min=1e-6)
            else:
                boot = torch.where(term[ids], 0.0, (self.gamma**T) * (self.ep_ret[ids] / T) / (1 - self.gamma))
                G = self.ep_disc_ret[ids] + boot
            b = self.start_bin[ids]
            self.bin_G.index_put_((b,), 0.8 * self.bin_G[b] + 0.2 * G)
            out["ep_len"] = T
            out["ep_ret_norm"] = self.ep_ret[ids] / (T * MAX_STEP_REWARD)
            self.reset(ids)
        return out
