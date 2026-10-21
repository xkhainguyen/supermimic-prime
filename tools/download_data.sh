#!/bin/bash
# Download the 40 G1-retargeted LAFAN1 clips (30 fps CSV: root pos, root quat xyzw, 29 joints).
# LAFAN1 (Ubisoft La Forge) is CC BY-NC-ND 4.0. The robot model is mjlab's G1 (g1_29dof_rev_1_0).
set -euo pipefail
cd "$(dirname "$0")/.."
HF=https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset
API=https://huggingface.co/api/datasets/lvhaidong/LAFAN1_Retargeting_Dataset/tree/main
mkdir -p data/lafan1_g1
for f in $(curl -s "$API/g1" | python3 -c "import json,sys; [print(x['path']) for x in json.load(sys.stdin)]"); do
  curl -sfL -o "data/lafan1_g1/$(basename "$f")" "$HF/resolve/main/$f"
done
echo "motions: $(ls data/lafan1_g1 | wc -l) clips"
