"""Stress test: train a fresh policy and critic purely in imagination through a frozen world model.

Isolates whether the WM generalizes or only works because it is co-trained with the policy it serves.
No simulator data reaches training. Rollout starts:
  ref     reference states ref_state(f) + reset noise, sim context xi from the DR distribution
  buffer  a frozen replay buffer collected once, before training, by the checkpoint's own policy
Measured in the real simulator only (no feedback):
  paired  from identical start states and xi, H-step closed-loop rollouts of the current policy in the
          WM and in the sim: imagined vs sim return (exploitation gap), sim termination rate, and the WM's
          open-loop error when replaying the sim's actions (hallucination along the policy's trajectories)
  eval    full-clip evaluation on the whole library (normalized return, failed clips)

Usage: python tools/frozen_wm.py --wm_ckpt logs/lafan1_main/ckpt_020000.pt --log_dir logs/frozen_late_ref --starts ref
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

parser = argparse.ArgumentParser()
parser.add_argument("--wm_ckpt", required=True)
parser.add_argument("--log_dir", required=True)
parser.add_argument("--motion_dir", default="data/lafan1_g1")
parser.add_argument("--contact_model", default="legacy", choices=["paper", "legacy"], help="sim the WM was trained in")
parser.add_argument("--starts", default="ref", choices=["ref", "buffer"])
parser.add_argument("--buffer_steps", type=int, default=1000)
parser.add_argument("--buffer_envs", type=int, default=4096)
parser.add_argument("--iterations", type=int, default=15000)
parser.add_argument("--horizon", type=int, default=32)
parser.add_argument("--pi_batch", type=int, default=4096)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--lam", type=float, default=0.95)
parser.add_argument("--critic_epochs", type=int, default=5)
parser.add_argument("--critic_minibatches", type=int, default=4)
parser.add_argument("--critic_clip", action="store_true")
parser.add_argument("--pair_every", type=int, default=250)
parser.add_argument("--pair_envs", type=int, default=1024)
parser.add_argument("--eval_every", type=int, default=1000)
parser.add_argument("--log_every", type=int, default=50)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

import torch
import torch.nn as nn

import supermimic.networks as networks
from supermimic.buffer import ReplayBuffer
from supermimic.env import TrackingSim
from supermimic.kinematics import G1Kinematics
from supermimic.mdp import CRITIC_EXTRA, INV_DIM, MAX_STEP_REWARD, MDP, OBS_DIM, flatten_state, normalize_xi, unflatten_state
from supermimic.motion import MotionLib
from supermimic.networks import Critic, Policy, WorldModel
from supermimic.robot import default_joint_pos_list
from supermimic.rotations import so3_exp

torch.backends.cuda.matmul.allow_tf32 = True
torch.manual_seed(args.seed)
dev = "cuda:0"
os.makedirs(args.log_dir, exist_ok=True)
ck = torch.load(args.wm_ckpt, map_location=dev, weights_only=False)
cargs = ck["args"]
networks.AMP_DTYPE = torch.bfloat16 if cargs.get("amp", "none") == "bf16" else None
json.dump(vars(args) | {"wm_iteration": ck["iteration"], "amp": cargs.get("amp")}, open(os.path.join(args.log_dir, "args.json"), "w"), indent=2)

kin = G1Kinematics(dev)
motion = MotionLib.load(args.motion_dir, kin, dev)
mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=dev))
K, Hh, HZ = cargs["goal_frames"], cargs["hist"], args.horizon
GOAL_DIM = 73 * K
G_MAX = MAX_STEP_REWARD / (1 - args.gamma)

wm = WorldModel().to(dev)
wm.load_state_dict(ck["wm"])
wm.eval()
for p in wm.parameters():
    p.requires_grad_(False)
pi = Policy(OBS_DIM, GOAL_DIM, hist=Hh).to(dev)
critic = Critic(INV_DIM + CRITIC_EXTRA + GOAL_DIM).to(dev)
opt_pi = torch.optim.Adam(pi.parameters(), lr=args.lr, betas=(0.9, 0.95))
opt_v = torch.optim.Adam(critic.parameters(), lr=args.lr)
valid = torch.nonzero(motion.clip_id.new_ones(motion.num_frames).bool() & (torch.arange(motion.num_frames, device=dev) + HZ <= motion.clip_end[motion.clip_id])).squeeze(-1)


def sample_xi(n):
    xi = torch.zeros(n, 33, device=dev)
    xi[:, 0].uniform_(0.3, 1.2)
    xi[:, 1:30].uniform_(-0.01, 0.01)
    xi[:, 30].uniform_(-0.025, 0.025)
    xi[:, 31:33].uniform_(-0.05, 0.05)
    return xi


def reference_starts(n):
    """ref_state(f) with the simulator's reset noise; history = the first observation repeated (a fresh episode)."""
    f = valid[torch.randint(0, len(valid), (n,), device=dev)]
    p, R, v, w, q, qd = (x.clone() for x in motion.ref_state(f))
    u = lambda lo, hi: torch.empty(n, device=dev).uniform_(lo, hi)  # noqa: E731
    p = p + torch.stack((u(-0.05, 0.05), u(-0.05, 0.05), u(-0.01, 0.01)), -1)
    R = so3_exp(torch.stack((u(-0.1, 0.1), u(-0.1, 0.1), u(-0.2, 0.2)), -1)) @ R
    v = v + torch.stack((u(-0.5, 0.5), u(-0.5, 0.5), u(-0.2, 0.2)), -1)
    w = w + torch.stack((u(-0.52, 0.52), u(-0.52, 0.52), u(-0.78, 0.78)), -1)
    q = torch.clamp(q + torch.empty_like(q).uniform_(-0.1, 0.1), kin.q_lo, kin.q_hi)
    xi = sample_xi(n)
    a_prev = torch.zeros(n, 29, device=dev)
    o = mdp.obs((p, R, v, w, q, qd), a_prev, xi, f)
    o_hist = mdp.add_noise(o.unsqueeze(1).expand(-1, Hh, -1).clone())
    return (p, R, v, w, q, qd), xi, f, a_prev, o_hist


# ------------------------------------------------------------------------------------------------ frozen buffer
buffer = None
if args.starts == "buffer":
    print("collecting the frozen buffer with the checkpoint's policy", flush=True)
    pi_src = Policy(OBS_DIM, GOAL_DIM, hist=Hh).to(dev)
    pi_src.load_state_dict(ck["pi"])
    pi_src.eval()
    env_b = TrackingSim(args.buffer_envs, motion, mdp, gamma=args.gamma, device=dev, contact_model=args.contact_model)
    buffer = ReplayBuffer(args.buffer_steps, args.buffer_envs, dev)
    obs_hist = torch.zeros(args.buffer_envs, Hh, OBS_DIM, device=dev)
    with torch.no_grad():
        for _ in range(args.buffer_steps):
            o = mdp.add_noise(mdp.obs(env_b.state(), env_b.a_prev, env_b.xi, env_b.f))
            fresh = env_b.ep_step == 0
            obs_hist = torch.where(fresh[:, None, None], o.unsqueeze(1).expand_as(obs_hist), torch.cat((obs_hist[:, 1:], o.unsqueeze(1)), 1))
            a = pi_src(obs_hist, motion.goal(env_b.f, K))
            buffer.add(env_b.step(a + 0.1 * torch.randn_like(a)))
    del env_b, pi_src
    torch.cuda.empty_cache()


def buffer_starts(n):
    starts, hists, got = [], [], 0
    while got < n:
        s, h = buffer.sample_starts(2 * n, Hh)
        ok = (s["f"] + HZ <= motion.clip_end[motion.clip_id[s["f"]]]).nonzero(as_tuple=False).squeeze(-1)[: n - got]
        starts.append({k: v[ok] for k, v in s.items()}), hists.append({k: v[ok] for k, v in h.items()})
        got += len(ok)
    s = {k: torch.cat([d[k] for d in starts]) for k in starts[0]}
    h = {k: torch.cat([d[k] for d in hists]) for k in hists[0]}
    with torch.no_grad():
        o_hist = mdp.add_noise(mdp.obs(unflatten_state(h["s"]), h["a_prev"], h["xi"], h["f"]))
    return unflatten_state(s["s"]), s["xi"], s["f"], s["a_prev"], o_hist


# ------------------------------------------------------------------------------------------------ imagination
def _imag_step(S, o_hist, f, xi, xi_n):
    a = pi(o_hist, motion.goal(f, K))
    S1 = wm.step(S, a, xi_n)
    o1 = mdp.add_noise(mdp.obs(S1, a, xi, f + 1))
    return S1, a, torch.cat((o_hist[:, 1:], o1.unsqueeze(1)), 1)


imag_step = torch.compile(_imag_step)
reward_term = torch.compile(lambda S, f, a, ap: (mdp.reward(S, f, a, ap), mdp.soft_termination(S, f)))
critic_in = torch.compile(lambda S, f, xi_n, g: torch.cat((mdp.critic_features(S, f, xi_n), g), -1))


def imagine(S, xi, f, a_prev, o_hist, grad: bool):
    """H-step closed-loop rollout through the WM; returns (states, actions, previous actions, frames)."""
    xi_n = normalize_xi(xi)
    Ss, As, Aps, Fs = [], [], [], []
    with torch.set_grad_enabled(grad):
        for _ in range(HZ):
            Aps.append(a_prev)
            S, a_prev, o_hist = imag_step(S, o_hist, f, xi, xi_n)
            f = f + 1
            Ss.append(S), As.append(a_prev), Fs.append(f)
    return Ss, As, Aps, Fs


def policy_update(S0, xi, f0, a_prev, o_hist):
    B = S0[0].shape[0]
    xi_n = normalize_xi(xi)
    pi.obs_norm.update(o_hist[:, -1]), pi.goal_norm.update(motion.goal(f0, K))
    Ss, As, Aps, Fs = imagine(S0, xi, f0, a_prev, o_hist, grad=True)
    flat = lambda xs: torch.stack(xs).flatten(0, 1)  # noqa: E731
    r, d = reward_term(tuple(flat([St[i] for St in Ss]) for i in range(6)), flat(Fs), flat(As), flat(Aps))
    r, d = r.view(HZ, B), d.view(HZ, B)
    alive = torch.cumprod(1 - d, 0)
    c_t = torch.cat((torch.ones(1, B, device=dev), alive[:-1]), 0)
    disc = args.gamma ** torch.arange(HZ, device=dev, dtype=torch.float32).unsqueeze(-1)
    for p in critic.parameters():
        p.requires_grad_(False)
    V_H = critic(critic_in(Ss[-1], Fs[-1], xi_n, motion.goal(Fs[-1], K)))
    if args.critic_clip:
        V_H = V_H.clamp(-G_MAX, G_MAX)
    J = (disc * c_t * r).sum(0) + args.gamma**HZ * alive[-1] * V_H
    loss = -J.mean()
    opt_pi.zero_grad(set_to_none=True)
    loss.backward()
    gn = nn.utils.clip_grad_norm_(pi.parameters(), 1.0)
    nn.utils.clip_grad_value_(pi.parameters(), 0.1)
    if torch.isfinite(gn):
        opt_pi.step()
    for p in critic.parameters():
        p.requires_grad_(True)
    with torch.no_grad():
        S_c = tuple(torch.cat((S0[i].unsqueeze(0), torch.stack([St[i].detach() for St in Ss])), 0).flatten(0, 1) for i in range(6))
        f_c = torch.cat((f0.unsqueeze(0), torch.stack(Fs)), 0).flatten()
        g_c = motion.goal(f_c, K)
        X = critic_in(S_c, f_c, xi_n.repeat(HZ + 1, 1), g_c).view(HZ + 1, B, -1)
    critic.norm.update(X)
    with torch.no_grad():
        V = critic(X)
        R_ = V[-1]
        targets = []
        for t in range(HZ - 1, -1, -1):
            R_ = r[t].detach() + args.gamma * (1 - d[t].detach()) * ((1 - args.lam) * V[t + 1] + args.lam * R_)
            targets.append(R_)
        targets = torch.stack(targets[::-1])
        if args.critic_clip:
            targets = targets.clamp(-G_MAX, G_MAX)
    Xf, Yf = X[:-1].reshape(-1, X.shape[-1]), targets.reshape(-1)
    n_ = Xf.shape[0]
    mb = n_ // args.critic_minibatches
    for _ in range(args.critic_epochs):
        perm = torch.randperm(n_, device=dev)
        for i in range(args.critic_minibatches):
            idx = perm[i * mb : (i + 1) * mb]
            v_loss = ((critic(Xf[idx]) - Yf[idx]) ** 2).mean()
            opt_v.zero_grad(set_to_none=True)
            v_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            opt_v.step()
    imag_ret = ((c_t * r).detach().sum(0) / (HZ * MAX_STEP_REWARD)).mean()
    return float(imag_ret), float(v_loss), float(gn), float(V_H.detach().mean())


# ------------------------------------------------------------------------------------------------ measurements
pair_env = TrackingSim(args.pair_envs, motion, mdp, gamma=args.gamma, device=dev, contact_model=args.contact_model, auto_reset=False)
eval_env = TrackingSim(len(motion.names), motion, mdp, eval_mode=True, device=dev, contact_model=args.contact_model)
ERR_FLOOR = torch.cat((torch.full((3,), 3e-3), torch.full((3,), 5e-3), torch.full((3,), 2e-2), torch.full((3,), 5e-2),
                       torch.full((29,), 3e-3), torch.full((29,), 5e-2))).to(dev)  # fmt: skip


@torch.no_grad()
def paired():
    """Same start state and xi in sim and WM; H steps of the current policy in each."""
    P = args.pair_envs
    ids = torch.arange(P, device=dev)
    f0 = valid[torch.randint(0, len(valid), (P,), device=dev)]
    pair_env.reset(ids, f0)
    S0, xi = pair_env.state(), pair_env.xi.clone()
    xi_n = normalize_xi(xi)
    a_prev = torch.zeros(P, 29, device=dev)
    o_hist = mdp.add_noise(mdp.obs(S0, a_prev, xi, f0).unsqueeze(1).expand(-1, Hh, -1).clone())
    # sim, closed loop
    r_sim, alive, A_sim, S_sim = [], torch.ones(P, device=dev), [], []
    oh, ap = o_hist.clone(), a_prev
    for t in range(HZ):
        a = pi(oh, motion.goal(pair_env.f, K))
        out = pair_env.step(a)
        r_sim.append(out["r"] * alive)
        alive = alive * (~out["term"]).float()
        A_sim.append(a), S_sim.append(pair_env.state())
        o1 = mdp.add_noise(mdp.obs(pair_env.state(), a, xi, pair_env.f))
        oh = torch.cat((oh[:, 1:], o1.unsqueeze(1)), 1)
    sim_ret = (torch.stack(r_sim).sum(0) / (HZ * MAX_STEP_REWARD)).mean()
    # WM, closed loop from the same start
    Ss, As, Aps, Fs = imagine(S0, xi, f0, a_prev, o_hist, grad=False)
    r, d = reward_term(tuple(torch.stack([St[i] for St in Ss]).flatten(0, 1) for i in range(6)), torch.stack(Fs).flatten(),
                       torch.stack(As).flatten(0, 1), torch.stack(Aps).flatten(0, 1))  # fmt: skip
    r, d = r.view(HZ, P), d.view(HZ, P)
    c_t = torch.cat((torch.ones(1, P, device=dev), torch.cumprod(1 - d, 0)[:-1]), 0)
    imag_ret = ((c_t * r).sum(0) / (HZ * MAX_STEP_REWARD)).mean()
    # WM, open loop on the sim's actions: normalized state error (same metric as the WM loss), at H and mean
    S = S0
    errs = []
    std = torch.maximum(wm.out_norm.std(), ERR_FLOOR)
    for t in range(HZ):
        S = wm.step(S, A_sim[t], xi_n)
        e = ((wm.state_error(S, S_sim[t]) / std) ** 2).mean(-1)
        errs.append((e * (alive > 0).float()).sum() / (alive > 0).float().sum().clamp(min=1))
    return {"pair_imag_ret": float(imag_ret), "pair_sim_ret": float(sim_ret), "pair_gap": float(imag_ret - sim_ret),
            "pair_sim_term": float(1 - alive.mean()), "wm_open_err_8": float(errs[7]), "wm_open_err_H": float(errs[-1])}  # fmt: skip


@torch.no_grad()
def full_eval():
    C = len(motion.names)
    ids = torch.arange(C, device=dev)
    eval_env.done_eval[:] = False
    eval_env.failed_eval[:] = False
    eval_env.reset(ids, motion.clip_start.clone())
    ret, hist = torch.zeros(C, device=dev), None
    while not bool(eval_env.done_eval.all()):
        o = mdp.obs(eval_env.state(), eval_env.a_prev, eval_env.xi, eval_env.f)
        hist = o.unsqueeze(1).repeat(1, Hh, 1) if hist is None else torch.cat((hist[:, 1:], o.unsqueeze(1)), 1)
        alive = ~eval_env.done_eval.clone()
        out = eval_env.step(pi(hist, motion.goal(eval_env.f, K)))
        ret += torch.where(alive, out["r"], 0.0)
        eval_env.f = torch.minimum(eval_env.f, motion.clip_end - 1)
    norm = ret / ((motion.clip_len - 1).float() * MAX_STEP_REWARD)
    return {"eval_return": float(norm.mean()), "eval_failed": int(eval_env.failed_eval.sum())}


# ------------------------------------------------------------------------------------------------ loop
log = open(os.path.join(args.log_dir, "log.csv"), "w")
cols = ["iteration", "wall_h", "imag_ret", "v_loss", "pi_gradnorm", "V_H", "pair_imag_ret", "pair_sim_ret", "pair_gap",
        "pair_sim_term", "wm_open_err_8", "wm_open_err_H", "eval_return", "eval_failed"]  # fmt: skip
log.write(",".join(cols) + "\n")
t0, agg, last = time.time(), [], {}
for it in range(args.iterations + 1):
    if it % args.pair_every == 0:
        pi.eval()
        last.update(paired())
        if it % args.eval_every == 0:
            last.update(full_eval())
        pi.train()
    if it == args.iterations:
        break
    lr = args.lr * (0.03 + 0.97 * 0.5 * (1 + math.cos(math.pi * min(it, 250_000) / 250_000)))
    for o in (opt_pi, opt_v):
        for g in o.param_groups:
            g["lr"] = lr
    starts = reference_starts(args.pi_batch) if args.starts == "ref" else buffer_starts(args.pi_batch)
    agg.append(policy_update(*starts))
    if (it + 1) % args.log_every == 0:
        m = [sum(x[i] for x in agg) / len(agg) for i in range(4)]
        row = {"iteration": it + 1, "wall_h": (time.time() - t0) / 3600, "imag_ret": m[0], "v_loss": m[1], "pi_gradnorm": m[2], "V_H": m[3]} | last
        log.write(",".join(f"{row.get(c, float('nan')):.6g}" if c != "iteration" else str(row[c]) for c in cols) + "\n")
        log.flush()
        print(" ".join(f"{c} {row.get(c, float('nan')):.4g}" if c != "iteration" else f"it {row[c]}" for c in cols), flush=True)
        agg, last = [], {}
torch.save({"pi": pi.state_dict(), "critic": critic.state_dict(), "args": vars(args)}, os.path.join(args.log_dir, "final.pt"))
