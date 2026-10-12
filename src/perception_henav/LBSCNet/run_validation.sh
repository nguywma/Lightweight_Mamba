#!/bin/bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "$DIR/validate.py" --weights "$DIR/weights/weights_epoch_065.pth" --dset_root /home/manh/drone/datasets/semantic_kitti/dataset/sequences
