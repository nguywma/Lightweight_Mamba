#!/bin/bash

# Paths
VEL_PATH="/home/manh/drone/HE-Nav/src/perception/raw_data/velodyne"
VOX_PATH="/home/manh/drone/HE-Nav/src/perception/raw_data/voxels"

# Count files
VEL_COUNT=$(ls -1 "$VEL_PATH"/*.bin 2>/dev/null | wc -l)
VOX_COUNT=$(ls -1 "$VOX_PATH"/*.bin 2>/dev/null | wc -l)

echo "Found $VEL_COUNT .bin files in $VEL_PATH"
echo "Found $VOX_COUNT .bin files in $VOX_PATH"

# Confirm before deleting
read -p "Do you really want to delete these files? (y/N): " confirm

if [[ "$confirm" =~ ^[Yy]$ ]]; then
    rm -rf "$VEL_PATH"/*.bin
    rm -rf "$VOX_PATH"/*.bin
    echo "Files deleted."
else
    echo "Aborted."
fi
