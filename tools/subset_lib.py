"""Write a packed motion library containing only the clips whose names match a regex.

Usage: python tools/subset_lib.py --src data/bones_seed --dst data/bones_flips_rolls --clips "^(flip|side_roll)"
"""

import argparse
import os
import re

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--dst", required=True)
ap.add_argument("--clips", required=True)
a = ap.parse_args()
names = open(os.path.join(a.src, "names.txt")).read().splitlines()
lens = np.load(os.path.join(a.src, "clip_len.npy"))
starts = np.concatenate(([0], np.cumsum(lens)[:-1]))
keep = [i for i, n in enumerate(names) if re.search(a.clips, n)]
frames = np.concatenate([np.arange(starts[i], starts[i] + lens[i]) for i in keep])
os.makedirs(a.dst, exist_ok=True)
for f in ("root_p.npy", "root_quat.npy", "q.npy"):
    np.save(os.path.join(a.dst, f), np.load(os.path.join(a.src, f), mmap_mode="r")[frames])
np.save(os.path.join(a.dst, "clip_len.npy"), lens[keep])
open(os.path.join(a.dst, "names.txt"), "w").write("\n".join(names[i] for i in keep) + "\n")
print(f"{len(keep)} clips, {len(frames)} frames ({len(frames) / 50 / 60:.1f} min) -> {a.dst}")
