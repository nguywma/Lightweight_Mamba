#!/usr/bin/env python3
"""
inference_ros_lbscnet.py  -  LBSCNet inference node (no file I/O, optimised)

Pipeline:
  ROS PointCloud2  ->  np.frombuffer decode  ->  voxelization  ->  model  ->  publish

Performance notes
-----------------
The main slowdown vs the file-based pipeline was pc2.read_points() + list():
that materialises a Python generator into a Python list before converting to
numpy, which is O(N) Python object allocations.  The fast replacement reads
the raw message byte buffer directly with np.frombuffer -- zero Python-level
looping, comparable speed to np.fromfile used in the original pipeline.
"""

import os
import sys
import time
import threading
import queue

import numpy as np
import torch
import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float64MultiArray


# -- repo path setup -----------------------------------------------------------
repo_path, _ = os.path.split(os.path.realpath(__file__))
repo_path, _ = os.path.split(repo_path)
sys.path.append(repo_path)

from utils.seed import seed_all
from utils.config import CFG
from utils.model import get_model
from utils.logger import get_logger
import utils.checkpoint as checkpoint


# ==============================================================================
# Constants  (match point2vox.compute_voxel_params hardcoded values)
# ==============================================================================

VOXEL_ORIGIN = np.array([-10.0, -10.0, -0.1], dtype=np.float32)
VOXEL_SIZE   = 0.1
GRID_SIZE    = (256, 256, 32)   # (X, Y, Z)

# ROS PointCloud2 field datatype -> numpy dtype
_ROS_DTYPE = {
    1: np.int8,  2: np.uint8,  3: np.int16, 4: np.uint16,
    5: np.int32, 6: np.uint32, 7: np.float32, 8: np.float64,
}
_ROS_TYPE_SIZE = {1:1, 2:1, 3:2, 4:2, 5:4, 6:4, 7:4, 8:8}


# ==============================================================================
# Fast PointCloud2 decoder  (replaces pc2.read_points + list())
# ==============================================================================

def _build_point_dtype(ros_msg):
    """
    Build a numpy structured dtype that matches the binary layout of one point
    in the PointCloud2 message, including any padding at the end of each point.
    This lets np.frombuffer decode the whole buffer in one call.
    """
    dtype_list = []
    for field in ros_msg.fields:
        np_type = _ROS_DTYPE.get(field.datatype, np.uint8)
        dtype_list.append((field.name, np_type))

    # Pad up to point_step so numpy strides line up correctly
    last_field    = ros_msg.fields[-1]
    last_end      = last_field.offset + _ROS_TYPE_SIZE.get(last_field.datatype, 4)
    if last_end < ros_msg.point_step:
        dtype_list.append(('_pad', np.uint8, (ros_msg.point_step - last_end,)))

    return np.dtype(dtype_list)


def decode_pointcloud2(ros_msg):
    """
    Decode a ROS PointCloud2 message into a contiguous (N, 3) float32 xyz array.

    Uses np.frombuffer on the raw byte buffer -- no Python-level looping,
    equivalent performance to np.fromfile used in the original pipeline.
    """
    if ros_msg.point_step == 0 or len(ros_msg.data) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    dtype     = _build_point_dtype(ros_msg)
    cloud_arr = np.frombuffer(ros_msg.data, dtype=dtype)   # (N,) structured

    xs = cloud_arr['x'].astype(np.float32)
    ys = cloud_arr['y'].astype(np.float32)
    zs = cloud_arr['z'].astype(np.float32)
    points_xyz = np.column_stack([xs, ys, zs])             # (N, 3) contiguous

    # Drop NaN / Inf (skip_nans equivalent)
    finite_mask = np.isfinite(points_xyz).all(axis=1)
    return points_xyz[finite_mask]


# ==============================================================================
# In-memory conversion:  decoded xyz  ->  (N,4) points + flat voxel grid
# ==============================    ================================================

def pointcloud_msg_to_arrays(ros_msg):
    """
    Convert a ROS PointCloud2 message to the two arrays the model needs.

    Returns
    -------
    points_xyzr    : (N, 4) float32  -- xyz + remission (= distance/10)
    voxel_occupancy: (256*256*32,) float32  -- 1.0 occupied, 0.0 empty
    """
    points_xyz = decode_pointcloud2(ros_msg)
    if points_xyz.shape[0] == 0:
        return (np.zeros((0, 4), dtype=np.float32),
                np.zeros(int(np.prod(GRID_SIZE)), dtype=np.float32))

    # -- (N,4) with remission = distance/10  (matches pointcloud_listener) ----
    remission   = np.linalg.norm(points_xyz, axis=1, keepdims=True) / 10.0
    points_xyzr = np.hstack([points_xyz, remission])           # (N, 4)

    # -- voxelization (mirrors point2vox.voxelization exactly) ----------------
    voxel_coords = ((points_xyz - VOXEL_ORIGIN) / VOXEL_SIZE).astype(np.int32)
    valid_mask   = np.all(
        (voxel_coords >= 0) & (voxel_coords < np.array(GRID_SIZE, np.int32)),
        axis=1
    )
    voxel_coords = voxel_coords[valid_mask]

    voxel_grid = np.zeros(GRID_SIZE, dtype=np.float32)
    if voxel_coords.shape[0] > 0:
        # ravel_multi_index + unique: vectorised, handles duplicate coords
        flat_idx = np.ravel_multi_index(
            (voxel_coords[:, 0], voxel_coords[:, 1], voxel_coords[:, 2]),
            GRID_SIZE
        )
        voxel_grid.ravel()[np.unique(flat_idx)] = 1.0

    return points_xyzr, voxel_grid.flatten()


# ==============================================================================
# Build model input dict  (replicates SemanticKitti.get_data + collate_fn)
# ==============================================================================

def build_model_input(points_xyzr, voxel_occupancy):
    """
    Produce the dict that DSC.forward() / collate_fn expects.

    DSC.forward() does:
        occupancy = example['occupancy'].permute(0, 3, 2, 1)  # (B,X,Y,Z)->(B,Z,Y,X)

    So we must pass (B, X, Y, Z) = (1, 256, 256, 32).
    point2vox fills voxel_grid[x, y, z] with GRID_SIZE=(256,256,32), so the
    flat array already has index order [x,y,z] -> reshape to (256,256,32) = (X,Y,Z).

    collate_fn keeps 'points' as a list (variable-length per sample), NOT stacked.
    """
    occ = torch.from_numpy(voxel_occupancy.reshape(256, 256, 32)).unsqueeze(0)  # (1,256,256,32)
    pts = torch.from_numpy(points_xyzr)                                          # (N,4)

    return {
        'occupancy': occ,    # Tensor (1, 256, 256, 32)
        'points':    [pts],  # list[ Tensor(N,4) ]
    }


# ==============================================================================
# Publisher helper
# ==============================================================================

def publish_coordinates(coordinates, publisher):
    # REMOVED: coordinates = coordinates[:, [0, 2, 1]]
    msg = Float64MultiArray()
    for coord in coordinates:
        msg.data.extend(coord)
    publisher.publish(msg)

# ==============================================================================
# Inference node
# ==============================================================================

class InferenceNode:
    # Match the original pipeline's rospy.Rate(10) polling rate.
    # Raise if your planner can handle faster updates; lower if output is too dense.
    PUBLISH_RATE_HZ = 10.0

    def __init__(self, model, device, coordinates_publisher, logger, conf_threshold=0.6):
        self.model     = model
        self.device    = device
        self.publisher = coordinates_publisher
        self.logger    = logger

        self._queue      = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._min_interval = 1.0 / self.PUBLISH_RATE_HZ  # seconds between publishes
        self._last_publish = 0.0                          # timestamp of last publish
        self._conf_threshold = conf_threshold

        self._worker = threading.Thread(target=self._inference_loop, daemon=True)
        self._worker.start()

        # Subscribe to the sensor-only topic so the network never reads its
        # own predictions back as input, which would cause a compounding feedback loop.
        # The full prediction-augmented map (occupancy_inflate) is reserved for the planner.
        topic = "/grid_map/occupancy_inflate_raw"
        rospy.Subscriber(topic, PointCloud2, self._ros_callback, queue_size=1)
        self.logger.info(f"Subscribed to {topic} at {self.PUBLISH_RATE_HZ} Hz")

    def _ros_callback(self, msg):
        # Always keep only the latest message; drop stale ones
        try:
            self._queue.put_nowait(msg)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put_nowait(msg)

    def _inference_loop(self):
        self.model.eval()
        while not self._stop_event.is_set() and not rospy.is_shutdown():
            try:
                msg = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._process_frame(msg)
            except Exception as e:
                self.logger.error(f"Inference error: {e}")

    def _process_frame(self, msg):
        t0 = time.time()

        # 1. decode + voxelize
        points_xyzr, voxel_occupancy = pointcloud_msg_to_arrays(msg)

        # 2. build model input
        data = build_model_input(points_xyzr, voxel_occupancy)
        data['occupancy'] = data['occupancy'].to(self.device)
        data['points']    = [p.to(self.device) for p in data['points']]

        # 3. forward pass
        with torch.no_grad():
            scores = self.model(data)

        t1 = time.time()
        self.logger.info(f"Inference time: {t1 - t0:.4f}s")

        # 4. rate-limit publishing
        now = time.time()
        if (now - self._last_publish) < self._min_interval:
            return
        self._last_publish = now

        # 5. post-process and publish
        raw_logits = scores['pred_semantic_1_1']  # (1, nbr_classes, 256, 256, 32)

        # Confidence = P(any obstacle) = 1 - P(class 0 / free)
        # softmax over class dim (1), take free-class prob, subtract from 1
        prob_free     = torch.softmax(raw_logits, dim=1)[:, 0, ...]  # (1, 256, 256, 32)
        prob_occupied = 1.0 - prob_free                              # (1, 256, 256, 32)
        prob_np       = prob_occupied[0].data.cpu().numpy().ravel()  # (256*256*32,)

        score_mask = prob_np >= self._conf_threshold

        # Only publish voxels predicted as occupied AND not already seen by sensor
        voxel_occ_3d = voxel_occupancy.reshape(256, 256, 32)
        voxel_mask   = voxel_occ_3d.ravel() == 1

        non_intersection = np.logical_and(score_mask, np.logical_not(voxel_mask))

        n_total    = int(score_mask.sum())
        n_filtered = int(non_intersection.sum())
        self.logger.info(
            f"Predictions: {n_total} occupied (conf>={self._conf_threshold:.2f}) "
            f"→ {n_filtered} new voxels published"
        )

        non_intersection_coords = np.column_stack(
            np.nonzero(non_intersection.reshape(256, 256, 32))
        )
        publish_coordinates(non_intersection_coords, self.publisher)

    def stop(self):
        self._stop_event.set()
        self._worker.join()


# ==============================================================================
# Entry point
# ==============================================================================

def main():
    print('???')
    rospy.init_node("inference_node")

    coordinates_publisher = rospy.Publisher(
        '/non_intersection_coordinates', Float64MultiArray, queue_size=1000
    )

    torch.backends.cudnn.enabled = True
    seed_all(0)

    weights_f     = rospy.get_param('~weights_file')
    dataset_f     = rospy.get_param('~dataset_root')
    out_path_root = rospy.get_param('~output_path')

    assert os.path.isfile(weights_f), f'Weights file not found: {weights_f}'

    checkpoint_path = torch.load(weights_f)
    config_dict     = checkpoint_path.pop('config_dict')
    config_dict['DATASET']['ROOT_DIR'] = dataset_f

    _cfg = CFG()
    _cfg.from_dict(config_dict)
    _cfg.data = config_dict

    logger = get_logger(out_path_root, 'logs_test.log')
    logger.info(f'============ Weights: "{weights_f}" ============\n')

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logger.info(f'Using device: {device}')

    # get_model needs a dataset object for metadata only; DataLoader never iterated
    from utils.dataset import get_dataset
    _dummy_dataset = get_dataset(_cfg)['test'].dataset

    logger.info('=> Loading network architecture...')
    model = get_model(_cfg, _dummy_dataset)
    logger.info('=> Loading network weights...')
    model = checkpoint.load_model(model, weights_f, logger)
    model = model.to(device=device)

    conf_threshold = rospy.get_param('~conf_threshold', 0.6)
    logger.info(f'Confidence threshold: {conf_threshold}')

    node = InferenceNode(model, device, coordinates_publisher, logger,
                         conf_threshold=conf_threshold)
    rospy.spin()
    node.stop()

if __name__ == "__main__":
    main()