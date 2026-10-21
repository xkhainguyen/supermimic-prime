# SuperMimic on mjlab

Unofficial reproduction of *SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable
World Model* (Liao et al., CoRL 2026, arXiv 2610.28480) on mjlab / MuJoCo-Warp. No official code is
released. This is a port of [zita-ch/supermimic_vibed](https://github.com/zita-ch/supermimic_vibed)
(Isaac Lab 3 kit-less, Newton + MuJoCo-Warp; MIT, see `LICENSE.supermimic_vibed`) with the Isaac Lab
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

On the CSAIL cluster the clips live in `~/datasets/lafan1_g1_csv` and `data/lafan1_g1` is a symlink.

## Usage

```bash
sbatch slurm/test_env.sbatch                       # FK vs sim, state round-trip, playback, sim throughput
sbatch slurm/train.sbatch lafan1_main 20000        # train + eval watcher on one GPU, requeue-safe
sbatch slurm/render.sbatch logs/lafan1_main/ckpt_016000.pt results/out.mp4 --clips dance2_subject1 --seconds 12
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
