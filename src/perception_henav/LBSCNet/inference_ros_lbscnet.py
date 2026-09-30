#!/usr/bin/env python3
"""
inference_ros_lbscnet.py  —  SCONet/LBSCNet inference node

Coordinate-frame convention
---------------------------
The LBSCNet checkpoint is trained with SemanticKITTI LiDAR coordinates:

  x ∈ [0, 51.2] m, y ∈ [-25.6, 25.6] m, z ∈ [-2, 4.4] m

The ROS map cloud is published in ``world`` coordinates.  The node converts
it to a *virtual* KITTI LiDAR frame before voxelization:

  p_world = R_world_lidar @ p_lidar + t_world_lidar
  p_lidar = R_world_lidar.T @ (p_world - t_world_lidar)

The virtual LiDAR follows the drone's x, y and yaw, but is gravity-aligned
(no roll/pitch) and sits ``~virtual_lidar_height`` above ``~ground_z``, like
the car-mounted KITTI sensor.  1.40 m is where the ground's top face lies below
the sensor in the SemanticKITTI voxel labels the network was trained on (median
over sequence 08), not the nominal 1.73 m mounting height, which lifts every
prediction ~0.3-0.5 m.  Using the drone's full attitude and
real altitude tilts/shifts the scene away from the training distribution and
makes the network hallucinate large structures (e.g. building walls) that
block the local map.  Set ``~virtual_lidar_height`` < 0 to use the odometry
altitude instead.
"""

import os
import sys
import time
import threading
import queue

import numpy as np
import torch
import rospy

from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Float64MultiArray
from nav_msgs.msg import Odometry

# ---------------------------------------------------------------------------
# Repo setup
# ---------------------------------------------------------------------------

repo_path, _ = os.path.split(os.path.realpath(__file__))
repo_path, _ = os.path.split(repo_path)
sys.path.append(repo_path)

from utils.seed import seed_all
from utils.config import CFG
from utils.model import get_model
from utils.logger import get_logger
import utils.checkpoint as checkpoint

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# SemanticKITTI/LBSCNet voxel geometry.  Keep these values synchronized with
# LBSCNet/cfgs/DSC-Base.yaml and the checkpoint configuration.
KITTI_ORIGIN = np.array([0.0, -25.6, -2.0], dtype=np.float32)
KITTI_SIZE = np.array([51.2, 51.2, 6.4], dtype=np.float32)
VOXEL_SIZE = 0.2
GRID_SIZE = (256, 256, 32)

_ROS_DTYPE = {
    1: np.int8,
    2: np.uint8,
    3: np.int16,
    4: np.uint16,
    5: np.int32,
    6: np.uint32,
    7: np.float32,
    8: np.float64,
}

_ROS_TYPE_SIZE = {
    1: 1,
    2: 1,
    3: 2,
    4: 2,
    5: 4,
    6: 4,
    7: 4,
    8: 8,
}

# ---------------------------------------------------------------------------
# PointCloud2 decoder
# ---------------------------------------------------------------------------

def _build_point_dtype(ros_msg):
    if not ros_msg.fields:
        raise ValueError("PointCloud2 has no fields")

    byte_order = ">" if ros_msg.is_bigendian else "<"
    names = []
    formats = []
    offsets = []

    for field in ros_msg.fields:
        np_type = np.dtype(
            _ROS_DTYPE.get(field.datatype, np.uint8)
        ).newbyteorder(byte_order)
        names.append(field.name)
        formats.append(np_type)
        offsets.append(field.offset)

    return np.dtype({
        "names": names,
        "formats": formats,
        "offsets": offsets,
        "itemsize": ros_msg.point_step,
    })


def decode_pointcloud2(ros_msg):
    """Return (N, 3) float32 array of finite xyz points in world frame."""
    if ros_msg.point_step == 0 or len(ros_msg.data) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    if not all(name in {field.name for field in ros_msg.fields}
               for name in ("x", "y", "z")):
        raise ValueError("PointCloud2 must contain x, y and z fields")

    dtype = _build_point_dtype(ros_msg)
    height = max(int(ros_msg.height), 1)
    width = int(ros_msg.width)
    if width <= 0:
        width = len(ros_msg.data) // ros_msg.point_step

    cloud_arr = np.ndarray(
        shape=(height, width),
        dtype=dtype,
        buffer=ros_msg.data,
        strides=(ros_msg.row_step, ros_msg.point_step),
    ).reshape(-1)

    points_xyz = np.column_stack([
        cloud_arr['x'].astype(np.float32),
        cloud_arr['y'].astype(np.float32),
        cloud_arr['z'].astype(np.float32),
    ])

    return points_xyz[np.isfinite(points_xyz).all(axis=1)]


def configure_geometry(config_dict):
    """Use the exact voxel geometry stored in the checkpoint config."""
    dataset_config = config_dict.get("DATASET", {})
    limits = dataset_config.get("LIMS")
    sizes = dataset_config.get("SIZES")
    grid_meters = dataset_config.get("GRID_METERS")

    if limits is None or sizes is None or grid_meters is None:
        raise KeyError("Checkpoint config lacks LIMS, SIZES or GRID_METERS")

    if len(limits) != 3 or len(sizes) != 3 or len(grid_meters) != 3:
        raise ValueError("Expected three-dimensional LBSCNet geometry")

    expected_sizes = tuple(
        int(round((axis_limits[1] - axis_limits[0]) / voxel_size))
        for axis_limits, voxel_size in zip(limits, grid_meters)
    )
    actual_sizes = tuple(int(size) for size in sizes)
    if expected_sizes != actual_sizes:
        raise ValueError(
            f"Inconsistent checkpoint geometry: limits/grid imply "
            f"{expected_sizes}, config has {actual_sizes}"
        )

    global KITTI_ORIGIN, KITTI_SIZE, VOXEL_SIZE, GRID_SIZE
    KITTI_ORIGIN = np.array(
        [axis_limits[0] for axis_limits in limits], dtype=np.float32
    )
    KITTI_SIZE = np.array(
        [axis_limits[1] - axis_limits[0] for axis_limits in limits],
        dtype=np.float32,
    )
    VOXEL_SIZE = float(grid_meters[0])
    if not np.allclose(grid_meters, VOXEL_SIZE):
        raise ValueError("This inference node requires equal voxel size on axes")
    GRID_SIZE = actual_sizes

# ---------------------------------------------------------------------------
# LiDAR-frame point cloud → SemanticKITTI voxel grid
# ---------------------------------------------------------------------------

def pointcloud_to_kitti_input(points_lidar):
    """Build LBSCNet point features and occupancy in LiDAR coordinates."""
    if points_lidar.shape[0] == 0:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros(int(np.prod(GRID_SIZE)), dtype=np.float32),
        )

    points_lidar = np.asarray(points_lidar, dtype=np.float32)
    finite_mask = np.isfinite(points_lidar).all(axis=1)
    points_lidar = points_lidar[finite_mask]

    if points_lidar.shape[0] == 0:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros(int(np.prod(GRID_SIZE)), dtype=np.float32),
        )

    voxel_coords = np.floor(
        (points_lidar - KITTI_ORIGIN[None, :]) / VOXEL_SIZE
    ).astype(np.int32)

    valid_mask = np.all(
        (voxel_coords >= 0)
        & (voxel_coords < np.array(GRID_SIZE, dtype=np.int32)),
        axis=1,
    )
    points_lidar = points_lidar[valid_mask]
    voxel_coords = voxel_coords[valid_mask]

    remission = np.linalg.norm(points_lidar, axis=1, keepdims=True) / 10.0
    points_xyzr = np.hstack([points_lidar, remission]).astype(np.float32)

    # ── build binary occupancy grid ───────────────────────────────────────────
    voxel_grid = np.zeros(GRID_SIZE, dtype=np.float32)
    if voxel_coords.shape[0] > 0:
        voxel_grid[
            voxel_coords[:, 0],
            voxel_coords[:, 1],
            voxel_coords[:, 2],
        ] = 1.0

    return points_xyzr, voxel_grid.flatten()

# ---------------------------------------------------------------------------
# Voxel-index  →  world coordinates (inverse of the transform above)
# ---------------------------------------------------------------------------

def voxel_indices_to_world(voxel_indices, lidar_position_world, world_lidar_rotation):
    voxel_indices = voxel_indices.astype(np.float32)

    # Use voxel centers rather than voxel corners
    points_lidar = (KITTI_ORIGIN[None, :] + (voxel_indices + 0.5) * VOXEL_SIZE)

    points_world = (points_lidar @ world_lidar_rotation.T + lidar_position_world[None, :])

    return points_world.astype(np.float32)


# ---------------------------------------------------------------------------
# Build model input dict
# ---------------------------------------------------------------------------

def build_model_input(points_xyzr, voxel_occupancy):
    return {
        'occupancy': torch.from_numpy(
            voxel_occupancy.reshape(GRID_SIZE)
        ).unsqueeze(0),
        'points': [torch.from_numpy(points_xyzr)],
    }

# ---------------------------------------------------------------------------
# Publish helper
# ---------------------------------------------------------------------------

def publish_coordinates(world_coords, publisher):
    """Publish (M, 3) world-frame coordinates as a flat Float64MultiArray."""
    msg = Float64MultiArray()
    msg.data = world_coords.ravel().tolist()
    publisher.publish(msg)

# SemanticKITTI learning classes, for reading the ``intensity`` channel below.
KITTI_CLASSES = (
    'empty', 'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle', 'person',
    'bicyclist', 'motorcyclist', 'road', 'parking', 'sidewalk', 'other-ground',
    'building', 'fence', 'vegetation', 'trunk', 'terrain', 'pole', 'traffic-sign',
)

def make_prediction_cloud(points, classes, confidence, observed, stamp, frame_id="world"):
    """Full network output as PointCloud2 for RViz: x, y, z, intensity (argmax
    class id, see KITTI_CLASSES), confidence (1 - P(empty)) and observed
    (1 if the voxel was already occupied in the input)."""
    fields = [PointField(name, 4 * i, PointField.FLOAT32, 1)
              for i, name in enumerate(("x", "y", "z", "intensity", "confidence", "observed"))]
    data = np.empty(len(points), dtype=np.dtype([(f.name, np.float32) for f in fields]))
    data["x"], data["y"], data["z"] = points[:, 0], points[:, 1], points[:, 2]
    data["intensity"], data["confidence"], data["observed"] = classes, confidence, observed

    msg = PointCloud2()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height, msg.width = 1, len(points)
    msg.fields = fields
    msg.is_bigendian = False
    msg.point_step = 4 * len(fields)
    msg.row_step = msg.point_step * len(points)
    msg.is_dense = True
    msg.data = data.tobytes()
    return msg

def quaternion_to_rotation_matrix(x, y, z, w):
    norm = np.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-8:
        return np.eye(3, dtype=np.float32)

    x /= norm
    y /= norm
    z /= norm
    w /= norm

    return np.array([
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
        ],
        [
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
        ],
        [
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ], dtype=np.float32)

def yaw_only_rotation(rotation):
    """Keep only the heading of a world-from-body rotation (gravity-aligned)."""
    yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)

def world_to_lidar(points_world, lidar_position_world, world_lidar_rotation):
    if points_world.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)

    points_relative_world = points_world - lidar_position_world[None, :]

    # R_world_lidar maps LiDAR coordinates to world coordinates.
    # Therefore inverse transform is its transpose.
    points_lidar = points_relative_world @ world_lidar_rotation

    return points_lidar.astype(np.float32)

# ---------------------------------------------------------------------------
# Inference node
# ---------------------------------------------------------------------------

class InferenceNode:

    def __init__(
        self,
        model,
        device,
        coordinates_publisher,
        logger,
        conf_threshold=0.6,
        virtual_lidar_height=1.40,
        ground_z=0.0,
        prediction_cloud_publisher=None,
    ):
        self.model     = model
        self.device    = device
        self.publisher = coordinates_publisher
        self.cloud_publisher = prediction_cloud_publisher
        self.logger    = logger

        self._queue      = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()

        self._conf_threshold = conf_threshold
        self._virtual_lidar_height = virtual_lidar_height
        self._ground_z = ground_z

        # ── LiDAR pose state ─────────────────────────────────────────────────
        self._drone_pos = np.zeros(3, dtype=np.float32)
        self._world_body_rotation = np.eye(3, dtype=np.float32)
        self._has_odom = False
        self._drone_pos_lock = threading.Lock()

        # ── ROS ──────────────────────────────────────────────────────────────
        rospy.Subscriber(
            "/visual_slam/odom",
            Odometry,
            self._odom_callback,
            queue_size=1,
        )
        rospy.Subscriber(
            "/grid_map/occupancy_inflate_raw",
            PointCloud2,
            self._ros_callback,
            queue_size=1,
        )

        self.logger.info(
            "Inference node started (world cloud -> SemanticKITTI LiDAR frame)"
        )

        self._worker = threading.Thread(
            target=self._inference_loop,
            daemon=True,
        )
        self._worker.start()

    # -------------------------------------------------------------------------

    def _odom_callback(self, msg):
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation

        rotation_world_body = yaw_only_rotation(quaternion_to_rotation_matrix(
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w,
        ))

        lidar_z = position.z
        if self._virtual_lidar_height >= 0.0:
            lidar_z = self._ground_z + self._virtual_lidar_height

        with self._drone_pos_lock:
            self._drone_pos = np.array(
                [position.x, position.y, lidar_z],
                dtype=np.float32,
            )
            self._world_body_rotation = rotation_world_body
            self._has_odom = True

    # -------------------------------------------------------------------------

    def _ros_callback(self, msg):
        try:
            self._queue.put_nowait(msg)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put_nowait(msg)

    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------

    def _process_frame(self, msg):
        t0 = time.time()

        # ── snapshot LiDAR/body pose ──────────────────────────────────────────
        with self._drone_pos_lock:
            drone_pos = self._drone_pos.copy()
            world_body_rotation = self._world_body_rotation.copy()
            has_odom = self._has_odom

        if not has_odom:
            self.logger.warn("Skipping point cloud until first odometry message")
            return

        if msg.header.frame_id not in ("", "world"):
            self.logger.warn(
                f"Expected world-frame cloud, got '{msg.header.frame_id}'"
            )

        points_world = decode_pointcloud2(msg)

        points_lidar = world_to_lidar(
            points_world,
            drone_pos,
            world_body_rotation,
        )

        points_xyzr, voxel_occupancy = pointcloud_to_kitti_input(
            points_lidar
        )

        if points_world.shape[0] > 0:
            valid_ratio = points_xyzr.shape[0] / float(points_world.shape[0])
            self.logger.info(
                f"Points inside SemanticKITTI ROI: "
                f"{points_xyzr.shape[0]}/{points_world.shape[0]} "
                f"({100.0 * valid_ratio:.1f}%)"
            )

        if points_xyzr.shape[0] == 0:
            self.logger.warn("No points inside SemanticKITTI ROI; skipping frame")
            return

        self.logger.info(
            f"Input occupied voxels: {int(np.count_nonzero(voxel_occupancy))}"
        )

        # ── build model input ─────────────────────────────────────────────────
        data = build_model_input(points_xyzr, voxel_occupancy)
        data['occupancy'] = data['occupancy'].to(self.device)
        data['points']    = [p.to(self.device) for p in data['points']]

        # ── forward pass ──────────────────────────────────────────────────────
        with torch.no_grad():
            scores = self.model(data)

        self.logger.info(f"Inference time: {time.time() - t0:.4f}s")

        # No rate limiter: the size-1 queue already drops stale clouds, and a
        # limiter here would throw away a finished prediction.
        # ── postprocess: confidence threshold ────────────────────────────────
        raw_logits  = scores['pred_semantic_1_1']
        prob_occ    = (
            1.0 - torch.softmax(raw_logits, dim=1)[:, 0]
        )[0].data.cpu().numpy().ravel()

        score_mask  = prob_occ >= self._conf_threshold

        # non-intersection: predicted occupied AND not already scanned
        scanned_mask    = voxel_occupancy.ravel() == 1
        non_intersection = score_mask & ~scanned_mask

        self.logger.info(
            f"Predictions: {int(score_mask.sum())} occupied "
            f"(conf>={self._conf_threshold:.2f}) "
            f"→ {int(non_intersection.sum())} new voxels"
        )

        # ── convert predicted voxel indices → world frame ────────────────────
        # An empty result is still published so the mapper drops the previous
        # prediction immediately instead of waiting for its timeout.
        voxel_indices = np.column_stack(
            np.nonzero(non_intersection.reshape(GRID_SIZE))
        )                                                      # (M, 3) int

        world_coords = voxel_indices_to_world(
            voxel_indices,
            drone_pos,
            world_body_rotation,
        )

        publish_coordinates(world_coords, self.publisher)

        # ── full-output visualisation (only when someone is listening) ───────
        # Unlike /non_intersection_coordinates this keeps every predicted voxel,
        # including ones that overlap the input and ones outside the planning map.
        if self.cloud_publisher is not None and self.cloud_publisher.get_num_connections() > 0:
            classes = torch.argmax(raw_logits, dim=1)[0].cpu().numpy().ravel()
            all_indices = np.column_stack(np.nonzero(score_mask.reshape(GRID_SIZE)))
            self.cloud_publisher.publish(make_prediction_cloud(
                voxel_indices_to_world(all_indices, drone_pos, world_body_rotation),
                classes[score_mask].astype(np.float32),
                prob_occ[score_mask].astype(np.float32),
                scanned_mask[score_mask].astype(np.float32),
                msg.header.stamp if not msg.header.stamp.is_zero() else rospy.Time.now(),
            ))

        if not msg.header.stamp.is_zero():
            latency = (rospy.Time.now() - msg.header.stamp).to_sec()
            self.logger.info(f"End-to-end latency (sensor -> prediction): {latency:.3f}s")

    # -------------------------------------------------------------------------

    def stop(self):
        self._stop_event.set()
        self._worker.join()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    rospy.init_node("inference_node")

    coordinates_publisher = rospy.Publisher(
        '/non_intersection_coordinates',
        Float64MultiArray,
        queue_size=1000,
    )
    prediction_cloud_publisher = rospy.Publisher(
        '/lbscnet/prediction_cloud',
        PointCloud2,
        queue_size=1,
    )

    torch.backends.cudnn.enabled = True
    seed_all(0)

    weights_f       = rospy.get_param('~weights_file')
    dataset_f       = rospy.get_param('~dataset_root')
    out_path_root   = rospy.get_param('~output_path')
    conf_threshold  = rospy.get_param('~conf_threshold', 0.6)
    virtual_lidar_height = rospy.get_param('~virtual_lidar_height', 1.40)
    ground_z        = rospy.get_param('~ground_z', 0.0)

    assert os.path.isfile(weights_f), f"Weights not found: {weights_f}"

    device = (
        torch.device('cuda') if torch.cuda.is_available()
        else torch.device('cpu')
    )

    checkpoint_path = torch.load(weights_f, map_location=device)
    config_dict     = checkpoint_path['config_dict']
    config_dict['DATASET']['ROOT_DIR'] = dataset_f
    configure_geometry(config_dict)

    _cfg = CFG()
    _cfg.from_dict(config_dict)
    _cfg.data = config_dict

    logger = get_logger(out_path_root, 'logs_test.log')
    logger.info(f"Using device: {device}")

    model = get_model(_cfg._dict, phase='test')
    model = checkpoint.load_model(model, weights_f, logger)
    model = model.to(device=device)

    node = InferenceNode(
        model,
        device,
        coordinates_publisher,
        logger,
        conf_threshold=conf_threshold,
        virtual_lidar_height=virtual_lidar_height,
        ground_z=ground_z,
        prediction_cloud_publisher=prediction_cloud_publisher,
    )

    rospy.spin()
    node.stop()


if __name__ == "__main__":
    main()
