"""Cluster-balanced reset-bin prior (off-paper; recipe from Shimingyi/allmimic tools/bin_clusters.py, CHANGES #28).

Rare motions are a sliver of BONES-SEED (flips ~0.04% of bins), so the return-based sampler almost never starts
there. Each 1 s bin (MotionLib bins, leftovers merged into the last one) gets 12 kinematic features from the
reference: pelvis height (mean, min), uprightness R22 (mean, min), root angular speed (max), vertical root velocity
(max, min), joint speed RMS (mean), and shares of frames that are airborne / hand / knee / pelvis-or-torso near the
floor. allmimic uses collision-geometry lowest points; here body origins with per-group height thresholds stand in.
Features standardized by median / IQR, k-means (k-means++ on a subsample, Lloyd on all bins). Bin weight in cluster
c: share_c / n_c with share_c ~ n_c^alpha, mean 1 over bins.

  python tools/bin_clusters.py --motion_dir data/bones_seed --out data/bin_prior_bones.npz
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from supermimic.kinematics import G1Kinematics  # noqa: E402
from supermimic.motion import MotionLib  # noqa: E402

FEATS = ("z_mean", "z_min", "up_mean", "up_min", "w_max", "vz_max", "vz_min", "qd_rms", "air", "hand", "knee", "body")
# body-origin heights (m) that count as touching the floor
NEAR = {"foot": (("left_ankle_roll_link", "right_ankle_roll_link"), 0.08),
        "hand": (("left_wrist_yaw_link", "right_wrist_yaw_link"), 0.08),
        "knee": (("left_knee_link", "right_knee_link"), 0.08),
        "body": (("pelvis", "torso_link"), 0.15)}  # fmt: skip


def bin_features(lib: MotionLib, chunk: int = 1 << 18) -> torch.Tensor:
    kin, dev, nb = lib.kin, lib.device, lib.num_bins
    gidx = {g: [kin.body_names.index(b) for b in bs] for g, (bs, _) in NEAR.items()}
    s = torch.zeros(nb, 7, device=dev, dtype=torch.float64)  # sums: z, up, qd_rms, air, hand, knee, body
    mx = torch.full((nb, 2), -1e9, device=dev)  # w_max, vz_max
    mn = torch.full((nb, 3), 1e9, device=dev)  # z_min, up_min, vz_min
    for a in range(0, lib.num_frames, chunk):
        f = torch.arange(a, min(a + chunk, lib.num_frames), device=dev)
        P, _, _, _ = kin.forward(lib.root_p[f], lib.root_R[f], lib.root_v[f], lib.root_w[f], lib.q[f], lib.qd[f], all_bodies=True)
        near = {g: (P[:, gidx[g], 2] < NEAR[g][1]).any(1) for g in NEAR}
        air = ~(near["foot"] | near["hand"] | near["knee"] | near["body"])
        z, up = lib.root_p[f, 2], lib.root_gz[f]
        qd = lib.qd[f].square().mean(1).sqrt()
        b = lib.frame_bin[f]
        s.index_add_(0, b, torch.stack([z, up, qd, air.float(), near["hand"].float(), near["knee"].float(), near["body"].float()], 1).double())
        mx.scatter_reduce_(0, b[:, None].expand(-1, 2), torch.stack([lib.root_w[f].norm(dim=1), lib.root_v[f, 2]], 1), "amax")
        mn.scatter_reduce_(0, b[:, None].expand(-1, 3), torch.stack([z, up, lib.root_v[f, 2]], 1), "amin")
    n = (lib.bin_end - lib.bin_start).double()[:, None]
    m = (s / n).float()
    return torch.stack([m[:, 0], mn[:, 0], m[:, 1], mn[:, 1], mx[:, 0], mx[:, 1], mn[:, 2], m[:, 2], m[:, 3], m[:, 4], m[:, 5], m[:, 6]], 1)


def kmeans(x: torch.Tensor, k: int, iters: int = 30, sample: int = 200_000, seed: int = 0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    sub = x[torch.randperm(len(x), generator=g)[:sample].to(x.device)]
    c = [sub[torch.randint(len(sub), (1,), generator=g).item()]]
    d2 = (sub - c[0]).square().sum(1)
    for _ in range(k - 1):
        c.append(sub[torch.multinomial((d2 / d2.sum()).cpu(), 1, generator=g).item()])
        d2 = torch.minimum(d2, (sub - c[-1]).square().sum(1))
    c = torch.stack(c)
    for _ in range(iters):
        lab = torch.cat([torch.cdist(x[i : i + 65536], c).argmin(1) for i in range(0, len(x), 65536)])
        cnt = torch.bincount(lab, minlength=k)
        acc = torch.zeros_like(c).index_add_(0, lab, x)
        c = torch.where(cnt[:, None] > 0, acc / cnt.clamp(min=1)[:, None], c)
    return lab, c


def weights(lab: np.ndarray, k: int, alpha: float) -> np.ndarray:
    n = np.bincount(lab, minlength=k).astype(np.float64)
    share = np.where(n > 0, n**alpha, 0)
    share /= share.sum()
    w = (share / np.maximum(n, 1))[lab]
    return (w / w.mean()).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion_dir", default="data/bones_seed")
    ap.add_argument("--out", default="data/bin_prior_bones.npz")
    ap.add_argument("--k", type=int, default=256)
    ap.add_argument("--alphas", type=float, nargs="+", default=[1.0, 0.5, 0.3, 0.0])
    ap.add_argument("--groups", default="flip|side_roll|cartwheel|lying|crawl", help="|-separated name substrings to report")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    lib = MotionLib.load(args.motion_dir, G1Kinematics(dev), dev, precompute_bodies=False)
    with torch.no_grad():
        feat = bin_features(lib)
        med = feat.median(0).values
        iqr = feat.quantile(0.75, 0) - feat.quantile(0.25, 0)
        lab, cen = kmeans((feat - med) / torch.where(iqr > 1e-6, iqr, torch.ones_like(iqr)), args.k)
    lab = lab.cpu().numpy()
    w = {a: weights(lab, args.k, a) for a in args.alphas}
    clip_of_bin = lib.bin_clip.cpu().numpy()
    names = np.array(lib.names)
    rep = {"bins": int(len(lab)), "k": args.k, "cluster_sizes_sorted": np.sort(np.bincount(lab, minlength=args.k)).tolist()}
    rep["expected_share_pct"] = {}
    for gname in args.groups.split("|"):
        hit = np.char.find(np.char.lower(names[clip_of_bin]), gname) >= 0
        rep["expected_share_pct"][gname] = {str(a): round(float(w[a][hit].sum() / len(lab) * 100), 4) for a in args.alphas}
        rep["expected_share_pct"][gname]["bins"] = int(hit.sum())
    np.savez(args.out, num_bins=len(lab), cluster=lab.astype(np.int16), features=feat.cpu().numpy(),
             **{f"w_alpha_{a:g}": w[a] for a in args.alphas})  # fmt: skip
    json.dump(rep, open(args.out.replace(".npz", "_report.json"), "w"), indent=1)
    print(json.dumps(rep["expected_share_pct"], indent=1))


if __name__ == "__main__":
    main()
