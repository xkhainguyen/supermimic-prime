# SuperMimic Prime

Unofficial reproduction of *SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model*
(Liao et al., CoRL 2026, arXiv 2610.28480) on [mjlab](https://github.com/mujocolab/mjlab) / MuJoCo-Warp. The paper has
no official code. This is a port of [zita-ch/supermimic_vibed](https://github.com/zita-ch/supermimic_vibed) (Isaac Lab)
with the simulator layer replaced by mjlab.

## Results

| Dataset | Setup | Return | Failed clips | Time |
|---|---|---|---|---|
| LAFAN1 (40 clips) | 1 GPU, paper recipe | 0.854 | 0 / 40 | 2.1 h (0.82 at 1.3 h) |
| BONES-SEED (129k clips) | 1 GPU, paper recipe | 0.931 | 279 | 1.8 h |
| BONES-SEED | + 3 changes below | 0.944 | 4 | ~6 h |

- **LAFAN1:** matches the paper (0.82 within 1.2 h on an H100). The paper gives no failed-clip count.
- **BONES-SEED:** the paper reports return above 0.90 with 25 failed clips after 3 h. The 4-failure checkpoint
  needs the three changes below, takes about twice as long, and is one seed. It is also not stable: after ~7.5 h the
  critic loss rises and failures climb to the thousands. Read 4 as the best point of a run, not a converged result.
- Flips and rolls fail (about 64 of 66 clips) under every contact model we tried.

## Changes from the paper

Each was added after the paper recipe failed in a logged way. None touches the model, reward or policy objective.

1. `--critic_clip`: clamp critic targets and the bootstrap to +-G_max. Without it the critic spikes around 7k
   iterations and training goes to NaN.
2. `--sampler completion` (after LaoYang1994/SuperMimic-Rep): score every visited reset bin by return-to-go over
   the best attainable until clip end, clipped to [0, 1].
3. `--bin_prior`, `--softmax_cap` (after Shimingyi/allmimic): per-cluster reset weights (alpha 0.3) and a softmax cap of 50x
   uniform. This took BONES-SEED from hundreds of failures to single digits.

Versus the Isaac Lab port: mjlab's G1 MJCF and contacts (capsules and foot spheres, no URDF hulls) and mjlab's
tracking-task solver settings (5 ms step x 4, 10 / 20 iterations). `--contact_model legacy` restores the first
LAFAN1 run's contacts.

## Setup

```bash
uv sync
tools/download_data.sh    # 40 LAFAN1 G1 clips (CC BY-NC-ND), not bundled
python tools/prepare_bones.py --src /path/to/bones-seed/g1/csv --out data/bones_seed   # BONES-SEED, optional
```

## Run

```bash
# LAFAN1
uv run python supermimic/train.py --log_dir logs/lafan1 --iterations 20000 --resume auto
uv run python supermimic/evaluate.py --run_dir logs/lafan1 --watch      # scores each checkpoint

# BONES-SEED
python tools/bin_clusters.py --motion_dir data/bones_seed --out data/bin_prior_bones.npz
uv run python supermimic/train.py --log_dir logs/bones --motion_dir data/bones_seed --iterations 80000 \
  --ckpt_every 4000 --resume auto --critic_clip --sampler completion \
  --bin_prior data/bin_prior_bones.npz --bin_prior_alpha 0.3 --softmax_cap 50

# Video
uv run python supermimic/evaluate.py --run_dir logs/lafan1 --ckpt logs/lafan1/ckpt_016000.pt --record results/traj.npz
uv run python tools/render.py results/traj.npz results/out.mp4 --clips dance2_subject1 --seconds 12
```

Evaluation: all clips, friction 1.0, no randomization or noise, deterministic policy; failure is the training
termination.

## License

MIT. Derived from zita-ch/supermimic_vibed (ETH Zurich, Chong Zhang); the original notice is in `LICENSE`.
