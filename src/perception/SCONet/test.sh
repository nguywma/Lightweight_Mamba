#!/bin/bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "$DIR/network/test.py" --weights "$DIR/weights/weights_epoch_037.pth"  --dset_root /root/datasets/semantic_kitti --out_path "$DIR/prediction"
