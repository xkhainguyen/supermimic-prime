"""SuperMimic training (Algorithm A.1): collect -> fit world model -> BPTT policy + critic update."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--iterations", type=int, default=320_000)
parser.add_argument("--buffer_steps", type=int, default=1000)
parser.add_argument("--utd", type=float, default=0.4, help="updates per vectorized simulator step")
parser.add_argument("--horizon", type=int, default=32)
parser.add_argument("--goal_frames", type=int, default=10)
parser.add_argument("--hist", type=int, default=10)
parser.add_argument("--wm_window", type=int, default=8)
parser.add_argument("--wm_batches", type=int, default=10)
parser.add_argument("--wm_batch", type=int, default=2048)
parser.add_argument("--pi_batch", type=int, default=4096)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--lam", type=float, default=0.95)
parser.add_argument("--action_noise", type=float, default=0.1)
parser.add_argument("--critic_epochs", type=int, default=5)
parser.add_argument("--critic_minibatches", type=int, default=4)
parser.add_argument("--no_critic", action="store_true")
parser.add_argument("--warmup_steps", type=int, default=64, help="sim steps before any update")
parser.add_argument("--wm_only_iters", type=int, default=500, help="world-model-only iterations at the start")
parser.add_argument("--lr_decay_iters", type=int, default=250_000)
parser.add_argument("--ckpt_every", type=int, default=2000)
parser.add_argument("--log_every", type=int, default=50)
parser.add_argument("--motion_dir", type=str, default=os.path.join(REPO, "data", "lafan1_g1"))
parser.add_argument("--log_dir", type=str, default=os.path.join(REPO, "logs", "run"))
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--resume", type=str, default=None, help="checkpoint path, or 'auto' for <log_dir>/last.pt if present")
parser.add_argument("--critic_clip", action="store_true", help="clamp critic targets and the policy's bootstrap value to the attainable return range")
parser.add_argument("--critic_huber", type=float, default=0.0, help="Huber delta for the critic loss (0: squared error)")
parser.add_argument("--critic_ema", type=float, default=0.0, help="EMA rate of a target critic for the GAE targets (0: none, as in the paper)")
parser.add_argument("--critic_lr_scale", type=float, default=1.0)
parser.add_argument("--lr_world_scale", type=str, default="sqrt", choices=["sqrt", "none"], help="multi-GPU lr scaling")
parser.add_argument("--amp", type=str, default="bf16", choices=["none", "bf16"], help="autocast dtype for MLP matmuls")
args = parser.parse_args()

import torch
import torch.nn as nn

from supermimic.buffer import ReplayBuffer
from supermimic.env import TrackingSim
from supermimic.kinematics import G1Kinematics
from supermimic.mdp import CRITIC_EXTRA, INV_DIM, MAX_STEP_REWARD, MDP, OBS_DIM, invariant_features, normalize_xi, unflatten_state
from supermimic.motion import MotionLib
import supermimic.networks as networks
from supermimic.networks import Critic, Policy, WorldModel
from supermimic.robot import default_joint_pos_list

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# data parallel (torchrun): one simulator, buffer and motion library per GPU; gradients and running
# normalizer statistics are averaged across GPUs; lr scales with sqrt(#GPUs) (paper Table A.5)
WORLD = int(os.environ.get("WORLD_SIZE", "1"))
RANK = int(os.environ.get("RANK", "0"))
if WORLD > 1:
    import torch.distributed as dist

    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    args.lr *= math.sqrt(WORLD) if args.lr_world_scale == "sqrt" else 1.0
dev = f"cuda:{torch.cuda.current_device()}"
torch.manual_seed(args.seed)  # identical network initialization on every rank
networks.AMP_DTYPE = torch.bfloat16 if args.amp == "bf16" else None
if RANK == 0:
    os.makedirs(args.log_dir, exist_ok=True)
    json.dump(vars(args) | {"world_size": WORLD}, open(os.path.join(args.log_dir, "args.json"), "w"), indent=2)


def sync_grads(m: nn.Module):
    if WORLD == 1:
        return
    for p in m.parameters():
        if p.grad is not None:
            dist.all_reduce(p.grad)
            p.grad /= WORLD

kin = G1Kinematics(dev)
motion = MotionLib.load(args.motion_dir, kin, dev)
mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=dev))
K, Hh, HZ = args.goal_frames, args.hist, args.horizon
GOAL_DIM = 73 * K
CRITIC_IN = INV_DIM + CRITIC_EXTRA + GOAL_DIM

wm = WorldModel().to(dev)
pi = Policy(OBS_DIM, GOAL_DIM, hist=Hh).to(dev)
critic = Critic(CRITIC_IN).to(dev)
opt_wm = torch.optim.Adam(wm.parameters(), lr=args.lr)
opt_pi = torch.optim.Adam(pi.parameters(), lr=args.lr, betas=(0.9, 0.95))
opt_v = torch.optim.Adam(critic.parameters(), lr=args.lr * args.critic_lr_scale)
start_it, wall_offset, sim_steps = 0, 0.0, 0
if args.resume == "auto":
    last = os.path.join(args.log_dir, "last.pt")
    args.resume = last if os.path.exists(last) else None
if args.resume:
    print(f"resuming from {args.resume}", flush=True)
    ck = torch.load(args.resume, map_location=dev)
    wm.load_state_dict(ck["wm"]), pi.load_state_dict(ck["pi"]), critic.load_state_dict(ck["critic"])
    opt_wm.load_state_dict(ck["opt_wm"]), opt_pi.load_state_dict(ck["opt_pi"]), opt_v.load_state_dict(ck["opt_v"])
    start_it, wall_offset, sim_steps = ck["iteration"], ck["wall_time"], ck["sim_steps"]
torch.manual_seed(args.seed + 1000 * RANK + start_it)  # per-rank simulator, sampling and noise streams
G_MAX = MAX_STEP_REWARD / (1 - args.gamma)  # largest attainable discounted return
critic_targ = None


# per-component floors on the error normalization (pos, rot, lin vel, ang vel, q, qd increments)
ERR_FLOOR = torch.cat((torch.full((3,), 3e-3), torch.full((3,), 5e-3), torch.full((3,), 2e-2), torch.full((3,), 5e-2),
                       torch.full((29,), 3e-3), torch.full((29,), 5e-2))).to(dev)  # fmt: skip


def _wm_rollout(s, a, xi):
    S = unflatten_state(s[:, 0])
    preds = []
    for j in range(args.wm_window):
        S = wm.step(S, a[:, j], normalize_xi(xi[:, j]))
        preds.append(S)
    return tuple(torch.stack([P[i] for P in preds], 1) for i in range(6))  # (B, k, ...)


def _wm_err(S_hat, s1, mask):
    std = torch.maximum(wm.out_norm.std(), ERR_FLOOR)
    e = (wm.state_error(S_hat, unflatten_state(s1)) / std) ** 2  # (B, k, D)
    e = e.mean(-1)
    return (e * mask).sum() / s1.shape[0], e[:, -1]


def _imag_step(S, o_hist, a_prev, f, xi, xi_n, g):
    a = pi(o_hist, g)
    S1 = wm.step(S, a, xi_n)
    f1 = f + 1
    o1 = mdp.add_noise(mdp.obs(S1, a, xi, f1))
    o_hist = torch.cat((o_hist[:, 1:], o1.unsqueeze(1)), 1)
    return S1, a, o_hist, f1, motion.goal(f1, K)


def _reward_term(S, f, a, a_prev):
    return mdp.reward(S, f, a, a_prev), mdp.soft_termination(S, f)


def _critic_input(S, f, xi_n, g):
    return torch.cat((mdp.critic_features(S, f, xi_n), g), -1)


wm_rollout = torch.compile(_wm_rollout)
wm_err = torch.compile(_wm_err)
imag_step = torch.compile(_imag_step)
reward_term = torch.compile(_reward_term)
critic_input_c = torch.compile(_critic_input)


def lr_at(it: int) -> float:
    if it >= args.lr_decay_iters:
        return args.lr * 0.03
    return args.lr * (0.03 + 0.97 * 0.5 * (1 + math.cos(math.pi * it / args.lr_decay_iters)))


def set_requires_grad(m: nn.Module, flag: bool):
    for p in m.parameters():
        p.requires_grad_(flag)


if args.critic_ema > 0:
    import copy

    critic_targ = copy.deepcopy(critic)
    set_requires_grad(critic_targ, False)


# ------------------------------------------------------------------------------------------------
def wm_update():
    w = buffer.sample_windows(args.wm_batch, args.wm_window)
    S_hat = wm_rollout(w["s"], w["a"], w["xi"])
    loss, e_last = wm_err(S_hat, w["s1"], w["mask"])
    opt_wm.zero_grad(set_to_none=True)
    loss.backward()
    sync_grads(wm)
    nn.utils.clip_grad_norm_(wm.parameters(), 1.0)
    opt_wm.step()
    m = w["mask"][:, -1]
    return loss.detach() / args.wm_window, (e_last.detach() * m).sum() / m.sum().clamp(min=1)


def sample_valid_starts(B: int):
    starts, hists, n = [], [], 0
    while n < B:
        s, h = buffer.sample_starts(2 * B, Hh)
        f = s["f"]
        ok = f + HZ <= motion.clip_end[motion.clip_id[f]]
        idx = ok.nonzero(as_tuple=False).squeeze(-1)[: B - n]
        starts.append({k: v[idx] for k, v in s.items()}), hists.append({k: v[idx] for k, v in h.items()})
        n += len(idx)
    cat = lambda ds: {k: torch.cat([d[k] for d in ds]) for k in ds[0]}  # noqa: E731
    return cat(starts), cat(hists)


PROFILE = bool(int(os.environ.get("SM_PROFILE", "0")))


def policy_update():
    if PROFILE:
        torch.cuda.synchronize(); tq0 = time.time()
    s, h = sample_valid_starts(args.pi_batch)
    S = unflatten_state(s["s"])
    xi, f, a_prev = s["xi"], s["f"], s["a_prev"]
    xi_n = normalize_xi(xi)
    with torch.no_grad():
        o_hist = mdp.obs(unflatten_state(h["s"]), h["a_prev"], h["xi"], h["f"])
    o_hist = mdp.add_noise(o_hist)
    set_requires_grad(wm, False)
    set_requires_grad(critic, False)
    B = args.pi_batch
    g0 = motion.goal(f, K)
    g = g0
    S0 = S
    Ss, As, Aps, Fs, Gs = [], [], [], [], []
    for t in range(HZ):
        Aps.append(a_prev)
        S, a_prev, o_hist, f, g = imag_step(S, o_hist, a_prev, f, xi, xi_n, g)
        Ss.append(S), As.append(a_prev), Fs.append(f), Gs.append(g)
    flat = lambda xs: torch.stack(xs).flatten(0, 1)  # noqa: E731  (H*B, ...)
    S_all = tuple(flat([St[i] for St in Ss]) for i in range(6))
    f_all = flat(Fs)
    r, d = reward_term(S_all, f_all, flat(As), flat(Aps))
    r, d = r.view(HZ, B), d.view(HZ, B)
    alive = torch.cumprod(1 - d, 0)  # c_{t+1}
    c_t = torch.cat((torch.ones(1, B, device=dev), alive[:-1]), 0)
    disc = args.gamma ** torch.arange(HZ, device=dev, dtype=torch.float32).unsqueeze(-1)
    J = (disc * c_t * r).sum(0)
    c = alive[-1]
    r_sum = (c_t * r).detach().sum(0)
    rews, dones = r.detach(), d.detach()
    V_boot = torch.zeros(B, device=dev)
    if not args.no_critic:
        V_boot = critic(_critic_input(S, f, xi_n, g))
        if args.critic_clip:
            V_boot = V_boot.clamp(-G_MAX, G_MAX)
        J = J + (args.gamma**HZ) * c * V_boot
    loss = -J.mean()
    if PROFILE:
        torch.cuda.synchronize(); tq1 = time.time()
    opt_pi.zero_grad(set_to_none=True)
    loss.backward()
    if PROFILE:
        torch.cuda.synchronize(); tq2 = time.time()
    sync_grads(pi)
    gn = nn.utils.clip_grad_norm_(pi.parameters(), 1.0)
    nn.utils.clip_grad_value_(pi.parameters(), 0.1)
    if torch.isfinite(gn):
        opt_pi.step()
    set_requires_grad(wm, True)
    set_requires_grad(critic, True)
    # ---------------- critic: GAE(lambda) targets over the imagined transitions ----------------
    v_loss = torch.zeros(())
    if not args.no_critic:
        with torch.no_grad():
            S_c = tuple(torch.cat((S0[i].unsqueeze(0), torch.stack([St[i].detach() for St in Ss])), 0).flatten(0, 1) for i in range(6))
            f_c = torch.cat((s["f"].unsqueeze(0), torch.stack(Fs)), 0).flatten()
            g_c = torch.cat((g0.unsqueeze(0), torch.stack(Gs)), 0).flatten(0, 1)
            xi_c = xi_n.unsqueeze(0).expand(HZ + 1, -1, -1).flatten(0, 1)
            X = critic_input_c(S_c, f_c, xi_c, g_c).view(HZ + 1, B, -1)
        critic.norm.update(X)
        with torch.no_grad():
            if critic_targ is not None:
                critic_targ.norm.load_state_dict(critic.norm.state_dict())
            V = (critic_targ if critic_targ is not None else critic)(X)
            R = V[-1]
            targets = []
            for t in range(HZ - 1, -1, -1):
                R = rews[t] + args.gamma * (1 - dones[t]) * ((1 - args.lam) * V[t + 1] + args.lam * R)
                targets.append(R)
            targets = torch.stack(targets[::-1])  # (H, B)
            if args.critic_clip:
                targets = targets.clamp(-G_MAX, G_MAX)
        Xf, Yf = X[:-1].reshape(-1, X.shape[-1]), targets.reshape(-1)
        n = Xf.shape[0]
        mb = n // args.critic_minibatches
        for _ in range(args.critic_epochs):
            perm = torch.randperm(n, device=dev)
            for i in range(args.critic_minibatches):
                idx = perm[i * mb : (i + 1) * mb]
                pred = critic(Xf[idx])
                if args.critic_huber > 0:
                    v_loss = 2 * nn.functional.huber_loss(pred, Yf[idx], delta=args.critic_huber)  # = squared error inside delta
                else:
                    v_loss = ((pred - Yf[idx]) ** 2).mean()
                opt_v.zero_grad(set_to_none=True)
                v_loss.backward()
                sync_grads(critic)
                nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                opt_v.step()
        if critic_targ is not None:
            with torch.no_grad():
                for pt, p in zip(critic_targ.parameters(), critic.parameters()):
                    pt.lerp_(p, args.critic_ema)
    if PROFILE:
        torch.cuda.synchronize()
        print(f"[prof] imagine fwd {1e3*(tq1-tq0):.0f} ms | bwd {1e3*(tq2-tq1):.0f} ms | critic {1e3*(time.time()-tq2):.0f} ms", flush=True)
    A_all = torch.stack(As).detach()
    diag = torch.stack((A_all.abs().mean(), V_boot.detach().mean(), A_all.abs().amax()))
    return loss.detach(), v_loss.detach(), (r_sum / HZ / 0.1).mean(), gn.detach(), c.mean().detach(), diag


# ------------------------------------------------------------------------------------------------
env = TrackingSim(args.num_envs, motion, mdp, gamma=args.gamma, device=dev)
buffer = ReplayBuffer(args.buffer_steps, args.num_envs, dev)
obs_hist = torch.zeros(args.num_envs, Hh, OBS_DIM, device=dev)

@torch.no_grad()
def collect(n_steps: int, stats: dict):
    global obs_hist, sim_steps
    for _ in range(n_steps):
        S = env.state()
        o = mdp.add_noise(mdp.obs(S, env.a_prev, env.xi, env.f))
        fresh = env.ep_step == 0
        obs_hist = torch.where(fresh[:, None, None], o.unsqueeze(1).expand_as(obs_hist), torch.cat((obs_hist[:, 1:], o.unsqueeze(1)), 1))
        g = motion.goal(env.f, K)
        pi.obs_norm.update(o)
        pi.goal_norm.update(g)
        a = pi(obs_hist, g)
        a = torch.clamp(a + args.action_noise * torch.randn_like(a), -20, 20)
        out = env.step(a)
        buffer.add(out)
        S0, S1 = unflatten_state(out["s"]), unflatten_state(out["s1"])
        wm.in_norm.update(wm.inputs(S0, out["a"], normalize_xi(out["xi"])))
        wm.out_norm.update(wm.targets(S0, S1))
        sim_steps += args.num_envs * WORLD
        stats["r"].append(out["r"].mean() / 0.1)
        stats["qd_max"].append(out["s1"][:, 42:71].abs().amax())
        stats["v_max"].append(out["s1"][:, 7:13].abs().amax())
        if "ep_len" in out:
            stats["ep_len"].append(out["ep_len"].mean())
            stats["ep_ret"].append(out["ep_ret_norm"].mean())
            stats["term"].append(out["term"].float().sum())
            stats["done"].append(out["done"].float().sum())

stats = {"r": [], "ep_len": [], "ep_ret": [], "term": [], "done": [], "qd_max": [], "v_max": []}
collect(max(args.warmup_steps, args.wm_window), stats)
log_f = open(os.path.join(args.log_dir, "train_log.csv"), "a") if RANK == 0 else None
if start_it == 0 and RANK == 0:
    log_f.write("iteration,wall_h,sim_steps,wm_loss,wm_err8,pi_loss,v_loss,imag_rew,pi_gradnorm,imag_survival,sim_rew,ep_len,ep_ret_norm,term_frac,it_per_s,lr\n")
t_start = time.time() - wall_offset
t_log = time.time()
step_acc = 0.0
agg = {"wm": [], "wm8": [], "pi": [], "v": [], "ir": [], "gn": [], "surv": [], "a_abs": [], "V": [], "a_max": []}
for it in range(start_it, args.iterations):
    lr = lr_at(it)
    for o in (opt_wm, opt_pi, opt_v):
        for gparam in o.param_groups:
            gparam["lr"] = lr * (args.critic_lr_scale if o is opt_v else 1.0)
    step_acc += 1.0 / args.utd
    n = int(step_acc)
    step_acc -= n
    prof = PROFILE and it % 20 == 0 and it > args.wm_only_iters + 5
    if prof:
        torch.cuda.synchronize(); tp0 = time.time()
    collect(n, stats)
    if prof:
        torch.cuda.synchronize(); tp1 = time.time()
    for _ in range(args.wm_batches):
        l, e8 = wm_update()
    if prof:
        torch.cuda.synchronize(); tp2 = time.time()
    agg["wm"].append(l), agg["wm8"].append(e8)
    if it >= args.wm_only_iters:
        pl, vl, ir, gn, surv, diag = policy_update()
        if prof:
            torch.cuda.synchronize(); tp3 = time.time()
            print(f"[prof] collect {1e3*(tp1-tp0):.0f} ms ({n} steps) | wm {1e3*(tp2-tp1):.0f} ms | policy+critic {1e3*(tp3-tp2):.0f} ms", flush=True)
        agg["pi"].append(pl), agg["v"].append(vl), agg["ir"].append(ir), agg["gn"].append(gn), agg["surv"].append(surv)
        agg["a_abs"].append(diag[0]), agg["V"].append(diag[1]), agg["a_max"].append(diag[2])
    if (it + 1) % args.log_every == 0 and RANK == 0:
        mean = lambda x: float(sum(float(v) for v in x) / max(len(x), 1))  # noqa: E731
        ips = args.log_every / (time.time() - t_log)
        t_log = time.time()
        wall_h = (time.time() - t_start) / 3600
        term_frac = float(sum(float(v) for v in stats["term"]) / max(sum(float(v) for v in stats["done"]), 1))
        row = [it + 1, wall_h, sim_steps, mean(agg["wm"]), mean(agg["wm8"]), mean(agg["pi"]), mean(agg["v"]), mean(agg["ir"]),
               mean(agg["gn"]), mean(agg["surv"]), mean(stats["r"]), mean(stats["ep_len"]), mean(stats["ep_ret"]), term_frac, ips, lr]  # fmt: skip
        log_f.write(",".join(f"{x:.6g}" if isinstance(x, float) else str(x) for x in row) + "\n")
        log_f.flush()
        print(f"it {it + 1} | {wall_h:.3f} h | wm {row[3]:.4f} e8 {row[4]:.4f} | pi {row[5]:.3f} v {row[6]:.4f} imag_r {row[7]:.3f} surv {row[9]:.3f} "
              f"| sim_r {row[10]:.3f} ep_len {row[11]:.1f} ep_ret {row[12]:.3f} term {term_frac:.2f} | gn {row[8]:.2f} |a| {mean(agg['a_abs']):.3f} max {mean(agg['a_max']):.2f} V {mean(agg['V']):.3f} "
              f"| sim max qd {max([float(x) for x in stats['qd_max']] or [0]):.0f} vel {max([float(x) for x in stats['v_max']] or [0]):.1f} "
              f"| {ips:.2f} it/s", flush=True)  # fmt: skip
        stats = {k: [] for k in stats}
        agg = {k: [] for k in agg}
    elif (it + 1) % args.log_every == 0:
        stats = {k: [] for k in stats}
        agg = {k: [] for k in agg}
    if WORLD > 1 and (it + 1) % 1000 == 0:  # ranks must hold identical parameters
        with torch.no_grad():
            fp = torch.stack([torch.cat([p.double().sum().view(1) for p in m.parameters()]).sum() for m in (wm, pi, critic)])
            fps = [torch.zeros_like(fp) for _ in range(WORLD)]
            dist.all_gather(fps, fp)
            if RANK == 0:
                dev_ = max(float((f - fps[0]).abs().max()) for f in fps)
                print(f"[ddp] it {it + 1} max parameter-checksum difference across ranks: {dev_:.3e}", flush=True)
    if ((it + 1) % args.ckpt_every == 0 or it + 1 == args.iterations) and RANK == 0:
        ck = {"wm": wm.state_dict(), "pi": pi.state_dict(), "critic": critic.state_dict(), "opt_wm": opt_wm.state_dict(),
              "opt_pi": opt_pi.state_dict(), "opt_v": opt_v.state_dict(), "iteration": it + 1,
              "wall_time": time.time() - t_start, "sim_steps": sim_steps, "args": vars(args),
              "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}  # fmt: skip
        last = os.path.join(args.log_dir, "last.pt")
        torch.save(ck, last + ".tmp")
        os.replace(last + ".tmp", last)
        path = os.path.join(args.log_dir, f"ckpt_{it + 1:06d}.pt")
        torch.save(ck, path + ".tmp")
        os.replace(path + ".tmp", path)
