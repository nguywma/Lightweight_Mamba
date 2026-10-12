#!/bin/bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "$DIR/test.py" --weights "$DIR/weights/weights_epoch_071.pth" --dset_root /root/datasets/semantic_kitti/dataset/sequences --out_path "$DIR/prediction"
