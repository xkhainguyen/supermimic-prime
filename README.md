# SuperMimic Prime (SuperMimic on mjlab)

Unofficial reproduction of *SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable
World Model* (Liao et al., CoRL 2026, arXiv 2610.28480) on mjlab / MuJoCo-Warp. No official code is
released. This is a port of [zita-ch/supermimic_vibed](https://github.com/zita-ch/supermimic_vibed)
(Isaac Lab 3 kit-less, Newton + MuJoCo-Warp; MIT, see `LICENSE`) with the Isaac Lab
simulator layer replaced by mjlab. Target: the LAFAN1 controlled comparison (40 G1 clips, 2.45 h, one
GPU), where the Isaac Lab port reports eval normalized return 0.829 with 0/40 failed clips at 1.47 h.

## What changed vs. the Isaac Lab port

| File | Status |
|---|---|
| `supermimic/{networks,buffer,mdp,motion,rotations,kinematics}.py` | unchanged (plain torch + mujoco) |
| `supermimic/robot.py` | mjlab's G1 (`g1_29dof_rev_1_0` MJCF, BeyondMimic actuators) instead of URDF -> MJCF -> USD |
| `supermimic/env.py` | mjlab `Scene` + `Simulation` (CUDA-graph stepping); state read from qpos/qvel |
| `supermimic/train.py`, `evaluate.py` | Isaac launcher removed; `--resume auto` for requeue |

Simulator settings: 5 ms physics step x 4 (50 Hz control), implicitfast, pyramidal cone, Newton
solver with mjlab's tracking-task iterations (10 / 20 line search), nconmax 128, njmax 500.

Differences from the Isaac Lab port that can change results:
* Collision geometry: mjlab's G1 capsules / foot spheres instead of the URDF convex mesh hulls.
  Self-collisions off in both. All robot geoms use condim 3 and priority over the ground, so the
  per-env friction in the sim context applies to every robot contact (Isaac: max of the pair, with
  ground mu = 0.3).
* Solver iterations follow mjlab's tracking task; the Isaac port used Newton's MJWarp defaults.

## Setup

```bash
uv sync
tools/download_data.sh           # 40 LAFAN1 G1 CSVs (HF lvhaidong/LAFAN1_Retargeting_Dataset), CC BY-NC-ND
```

BONES-SEED (the paper's full setup) is not redistributed here; get it from its owners, then:

```bash
python tools/prepare_bones.py --src /path/to/bones-seed/g1/csv --out data/bones_seed   # name + feasibility filter, 50 Hz packed library
```

## Usage

```bash
uv run python tools/test_env.py                                       # FK vs sim, state round-trip, sim throughput
uv run python supermimic/train.py --log_dir logs/lafan1_main --iterations 20000 --resume auto
uv run python supermimic/evaluate.py --run_dir logs/lafan1_main --watch   # scores each checkpoint on all clips
uv run python supermimic/evaluate.py --run_dir logs/lafan1_main --ckpt logs/lafan1_main/ckpt_016000.pt --record results/traj.npz
uv run python tools/render.py results/traj.npz results/out.mp4 --clips dance2_subject1 --seconds 12
uv run python tools/plot.py logs/lafan1_main --out results/success_curve.png
```

Checkpoints every 2000 iterations in `logs/<run>/`; the eval watcher scores each on all 40 clips
(friction 1.0, no DR or noise, deterministic policy, failure = training termination) and appends to
`logs/<run>/eval.csv`.

## Results (LAFAN1, 40 clips, one shared H200, seed 0)

| Iteration | Train time (h) | Eval return | Failed clips |
|---|---|---|---|
| 2000 | 0.20 | 0.515 | 14 |
| 4000 | 0.42 | 0.713 | 4 |
| 6000 | 0.63 | 0.788 | 1 |
| 8000 | 0.84 | 0.811 | 0 |
| 10000 | 1.06 | 0.808 | 2 |
| 12000 | 1.27 | 0.829 | 1 |
| 14000 | 1.48 | 0.820 | 1 |
| 16000 | 1.70 | 0.839 | 0 |
| 18000 | 1.91 | 0.850 | 0 |
| 20000 | 2.13 | 0.854 | 0 |

Return 0.82 at 1.27 h (paper, Fig. 4a: within 1.2 h on one H100; Isaac Lab port: 0.824 at 1.24 h);
return >= 0.82 with 0/40 failed at 1.70 h (Isaac Lab port: 1.47 h), held through 20000 iterations. Train time includes the eval
watcher sharing the GPU (about 2.9 -> 2.0 it/s while it runs); the paper does not count evaluation.
Training throughput: 2.9 it/s on H200, 2.6 it/s on L40S; simulator alone 222k control steps/s at
4096 envs on L40S.

## BONES-SEED (paper full setup, one GPU)

129,458 clips (262.9 h after the paper's name and feasibility filters; the paper reports 129,514). Paper target for
one GPU and 3 h: return above SONIC's 0.90 with 25 failed clips. Evaluation scores every clip once, failure = training
termination.

```bash
python tools/bin_clusters.py --motion_dir data/bones_seed --out data/bin_prior_bones.npz
uv run python supermimic/train.py --log_dir logs/bones --motion_dir data/bones_seed --iterations 80000 --ckpt_every 4000 \
  --resume auto --critic_clip --sampler completion --bin_prior data/bin_prior_bones.npz --bin_prior_alpha 0.3 --softmax_cap 50
```

| Config | Best checkpoint | Return | Failed clips | Notes |
|---|---|---|---|---|
| Paper settings only | it 16k, 1.8 h | 0.931 | 279 | critic spikes from ~7k, NaN by 21k |
| + `--critic_clip` | it 16k, 1.9 h | 0.930 | 273 | degrades after ~5 h |
| + completion sampler | swings between 117 and 1258 failed | | | |
| + balanced prior and softmax cap | it 52k, ~6 h | 0.944 | 4 | 4 to 15 failed for 6.1 to 7.5 h, then collapses (826 failed at 76k) |

## What differs from the paper, and why

These are the only departures from the paper's recipe. Each was added after the paper-faithful run failed in a
specific, logged way; none changes the model, the reward or the policy objective.

1. **Critic clamp** (`--critic_clip`). Critic targets and the bootstrap value are clamped to +-G_max (G_max = max step
   reward / (1 - gamma)). Without it the critic loss spikes (first at 6.7k to 7.7k iterations) and the run goes to NaN.
2. **Completion reset sampler** (`--sampler completion`, after LaoYang1994/SuperMimic-Rep). Every visited bin of a
   finished episode is scored by its discounted return-to-go divided by the best attainable until the clip end or time
   limit, clipped to [0, 1]. The start-bin-only score in the Isaac Lab port ranks short clip-end episodes as hard.
3. **Cluster-balanced reset prior** (`--bin_prior`, `--softmax_cap`, after Shimingyi/allmimic). Reset bins get a per-cluster
   weight (alpha 0.3) so large motion families do not dominate, and the softmax term is capped at 50x the uniform share.
   This is what took BONES-SEED from hundreds of failed clips to single digits.

Simulator: mjlab's G1 and contacts instead of the Isaac Lab URDF (see above). `--contact_model legacy` restores the
first LAFAN1 run's contacts.

## How these results compare with the paper

* **LAFAN1.** Equal, not better. Return 0.82 at 1.27 h (paper: within 1.2 h on an H100, evaluation not counted); our
  clock includes an evaluation watcher sharing the GPU. 0.854 and 0/40 failed at 20k iterations; the paper gives no
  failed-clip count.
* **BONES-SEED.** The best checkpoint has fewer failures than the paper's 1-GPU figure (4 vs 25) and a higher return than
  SONIC (0.944 vs 0.90), but it needs the three changes above, took ~6 h instead of 3 h, and is a single seed. With the
  paper's recipe alone we get 0.93 and 273 to 279 failed clips. The balanced run is not stable: after ~7.5 h the critic
  loss rises and failures climb to thousands. Treat the 4-failure number as the best of a trajectory, not a converged result.
* **Flips and rolls.** About 64 of 66 such clips fail under both contact models, so the contact model is not the cause.

## Credits and license

Code derived from zita-ch/supermimic_vibed (MIT, ETH Zurich, Chong Zhang); the original license is in
`LICENSE`. Changes in this repository are released under the same MIT terms. The LAFAN1 retargeted
clips are CC BY-NC-ND and are not included.
