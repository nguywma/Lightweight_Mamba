#!/usr/bin/env python3
import rospy
import numpy as np
import os
import threading
import cv2

from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from point2vox import PointCloudVoxelization


class DepthImageListener:
    def __init__(self):
        self.input_folder = "/home/manh/drone/HE-Nav/src/perception/raw_data/velodyne"
        self.output_folder = "/home/manh/drone/HE-Nav/src/perception/raw_data/voxels"

        self.grid_size = (256, 256, 32)
        self.voxelizer = PointCloudVoxelization(self.input_folder, self.output_folder, self.grid_size)
        self.lock = threading.Lock()

        self.bridge = CvBridge()
        self.scan_count = 0

        # Camera intrinsics from /camera/depth/camera_info
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        # Latest RGB image
        self.rgb_image = None
        self.rgb_lock = threading.Lock()

        os.makedirs(self.input_folder, exist_ok=True)

    def camera_info_callback(self, msg):
        if self.fx is None:  # set once
            self.fx = msg.K[0]
            self.fy = msg.K[4]
            self.cx = msg.K[2]
            self.cy = msg.K[5]
            rospy.loginfo(f"Camera intrinsics loaded: fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}")

    def rgb_callback(self, data):
        try:
            rgb = self.bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
            with self.rgb_lock:
                self.rgb_image = rgb
        except Exception as e:
            rospy.logerr("RGB callback error: %s", str(e))

    def callback(self, data):
        try:
            if self.fx is None:
                rospy.logwarn("No camera intrinsics yet, skipping frame...")
                return

            depth_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")

            self.save_depth_as_bin(depth_image, self.scan_count)

            t = threading.Thread(target=self.process_voxelization)
            t.start()

            self.scan_count += 1

        except Exception as e:
            rospy.logerr("Depth callback error: %s", str(e))

    def save_depth_as_bin(self, depth_image, scan_count):
        file_name = f"{scan_count:06}.bin"
        file_path = os.path.join(self.input_folder, file_name)

        # Ensure depth is in meters
        if depth_image.dtype != np.float32:
            depth_image = depth_image.astype(np.float32) / 1000.0  # assume mm → m

        h, w = depth_image.shape
        i, j = np.meshgrid(np.arange(w), np.arange(h))

        z = depth_image
        x = (i - self.cx) * z / self.fx
        y = (j - self.cy) * z / self.fy

        # Keep only valid depth points
        mask = z > 0
        x = x[mask]
        y = y[mask]
        z = z[mask]

        # Add dummy intensity (all 1.0)
        intensity = np.ones_like(z, dtype=np.float32)

        points = np.stack((x, y, z, intensity), axis=-1).astype(np.float32)

        # Save as binary file (SemKITTI/KITTI format: x,y,z,intensity)
        points.tofile(file_path)

        rospy.loginfo(f"Saved point cloud {scan_count} -> {file_path}")

    def process_voxelization(self):
        with self.lock:
            self.voxelizer.voxelization()

    def listener(self):
        rospy.init_node("pointcloud_listener", anonymous=True)

        rospy.Subscriber("/camera/depth/camera_info", CameraInfo, self.camera_info_callback)
        rospy.Subscriber("/camera/depth/image_raw", Image, self.callback)
        rospy.Subscriber("/camera/color/image_raw", Image, self.rgb_callback)

        rospy.spin()


if __name__ == "__main__":
    listener = DepthImageListener()
    listener.listener()
