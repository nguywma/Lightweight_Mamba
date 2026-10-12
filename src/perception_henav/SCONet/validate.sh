#!/bin/bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "$DIR/network/validate.py" --weights "$DIR/weights/weights_epoch_037.pth"  --dset_root /home/manh/drone/datasets/semantic_kitti
