#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <pcl/search/impl/kdtree.hpp>
#include <limits>
#include <vector>

using namespace std;
using namespace Eigen;

ros::Publisher pub_cloud;

sensor_msgs::PointCloud2 local_map_pcl;
sensor_msgs::PointCloud2 local_depth_pcl;

ros::Subscriber odom_sub;
ros::Subscriber global_map_sub, local_map_sub;

ros::Timer local_sensing_timer;

bool has_global_map(false);
bool has_local_map(false);
bool has_odom(false);

nav_msgs::Odometry _odom;

double sensing_horizon, sensing_rate, estimation_rate;
double _x_size, _y_size, _z_size;
double _gl_xl, _gl_yl, _gl_zl;
double _resolution, _inv_resolution;
int _GLX_SIZE, _GLY_SIZE, _GLZ_SIZE;

ros::Time last_odom_stamp = ros::TIME_MAX;

inline Eigen::Vector3d gridIndex2coord(const Eigen::Vector3i& index) {
  Eigen::Vector3d pt;
  pt(0) = ((double)index(0) + 0.5) * _resolution + _gl_xl;
  pt(1) = ((double)index(1) + 0.5) * _resolution + _gl_yl;
  pt(2) = ((double)index(2) + 0.5) * _resolution + _gl_zl;

  return pt;
};

inline Eigen::Vector3i coord2gridIndex(const Eigen::Vector3d& pt) {
  Eigen::Vector3i idx;
  idx(0) = std::min(std::max(int((pt(0) - _gl_xl) * _inv_resolution), 0),
                    _GLX_SIZE - 1);
  idx(1) = std::min(std::max(int((pt(1) - _gl_yl) * _inv_resolution), 0),
                    _GLY_SIZE - 1);
  idx(2) = std::min(std::max(int((pt(2) - _gl_zl) * _inv_resolution), 0),
                    _GLZ_SIZE - 1);

  return idx;
};

void rcvOdometryCallbck(const nav_msgs::Odometry& odom) {
  /*if(!has_global_map)
    return;*/
  has_odom = true;
  _odom = odom;
}

pcl::PointCloud<pcl::PointXYZ> _cloud_all_map, _local_map;
pcl::VoxelGrid<pcl::PointXYZ> _voxel_sampler;
sensor_msgs::PointCloud2 _local_map_pcd;

pcl::search::KdTree<pcl::PointXYZ> _kdtreeLocalMap;
vector<int> _pointIdxRadiusSearch;
vector<float> _pointRadiusSquaredDistance;

void rcvGlobalPointCloudCallBack(
    const sensor_msgs::PointCloud2& pointcloud_map) {
  if (has_global_map) return;

  ROS_WARN("Global Pointcloud received..");

  pcl::PointCloud<pcl::PointXYZ> cloud_input;
  pcl::fromROSMsg(pointcloud_map, cloud_input);

  _voxel_sampler.setLeafSize(0.1f, 0.1f, 0.1f);
  _voxel_sampler.setInputCloud(cloud_input.makeShared());
  _voxel_sampler.filter(_cloud_all_map);

  _kdtreeLocalMap.setInputCloud(_cloud_all_map.makeShared());

  has_global_map = true;
}

// Depth-camera model: pinhole intrinsics from camera.yaml, rendered at
// 1/downsample resolution with a z-buffer so only the nearest surface per pixel
// is returned (occlusion). Each map point is a solid 0.1 m voxel, splatted over
// its pixel footprint so it hides whatever lies behind it.
int cam_width, cam_height, cam_downsample;
double cam_fx, cam_fy, cam_cx, cam_cy, cam_min_range;
Eigen::Matrix3d R_body_cam;   // optical frame (z forward, x right, y down) -> body
Eigen::Vector3d t_body_cam;
std::vector<float> depth_buffer;

void renderSensedPoints(const ros::TimerEvent& event) {
  if (!has_global_map || !has_odom) return;

  Eigen::Quaterniond q;
  q.x() = _odom.pose.pose.orientation.x;
  q.y() = _odom.pose.pose.orientation.y;
  q.z() = _odom.pose.pose.orientation.z;
  q.w() = _odom.pose.pose.orientation.w;

  Eigen::Vector3d body_pos(_odom.pose.pose.position.x,
                           _odom.pose.pose.position.y,
                           _odom.pose.pose.position.z);
  Eigen::Matrix3d R_world_cam = q.toRotationMatrix() * R_body_cam;
  Eigen::Vector3d cam_pos = body_pos + q.toRotationMatrix() * t_body_cam;
  Eigen::Matrix3d R_cam_world = R_world_cam.transpose();

  const int W = cam_width / cam_downsample, H = cam_height / cam_downsample;
  const double fx = cam_fx / cam_downsample, fy = cam_fy / cam_downsample;
  const double cx = cam_cx / cam_downsample, cy = cam_cy / cam_downsample;
  const float kInf = std::numeric_limits<float>::infinity();
  depth_buffer.assign(W * H, kInf);

  _local_map.points.clear();
  pcl::PointXYZ searchPoint(cam_pos(0), cam_pos(1), cam_pos(2));
  _pointIdxRadiusSearch.clear();
  _pointRadiusSquaredDistance.clear();

  if (_kdtreeLocalMap.radiusSearch(searchPoint, sensing_horizon + _resolution,
                                   _pointIdxRadiusSearch,
                                   _pointRadiusSquaredDistance) <= 0)
    return;

  for (size_t i = 0; i < _pointIdxRadiusSearch.size(); ++i) {
    const pcl::PointXYZ& pt = _cloud_all_map.points[_pointIdxRadiusSearch[i]];
    Eigen::Vector3d pc = R_cam_world * (Eigen::Vector3d(pt.x, pt.y, pt.z) - cam_pos);
    if (pc(2) < cam_min_range) continue;

    const double u = fx * pc(0) / pc(2) + cx;
    const double v = fy * pc(1) / pc(2) + cy;
    // Half footprint of a voxel in pixels (at least one pixel).
    const int hu = std::max(0, (int)ceil(0.5 * fx * _resolution / pc(2)));
    const int hv = std::max(0, (int)ceil(0.5 * fy * _resolution / pc(2)));
    const int u0 = std::max(0, (int)floor(u) - hu), u1 = std::min(W - 1, (int)floor(u) + hu);
    const int v0 = std::max(0, (int)floor(v) - hv), v1 = std::min(H - 1, (int)floor(v) + hv);
    const float z = pc(2);
    for (int r = v0; r <= v1; ++r)
      for (int c = u0; c <= u1; ++c) {
        float& d = depth_buffer[r * W + c];
        if (z < d) d = z;
      }
  }

  // Back-project every valid depth pixel, as a real depth camera does.
  for (int r = 0; r < H; ++r)
    for (int c = 0; c < W; ++c) {
      const float z = depth_buffer[r * W + c];
      if (z == kInf || z > sensing_horizon) continue;
      Eigen::Vector3d pc(((c + 0.5) - cx) / fx * z, ((r + 0.5) - cy) / fy * z, z);
      Eigen::Vector3d pw = R_world_cam * pc + cam_pos;
      _local_map.points.push_back(pcl::PointXYZ(pw(0), pw(1), pw(2)));
    }

  _local_map.width = _local_map.points.size();
  _local_map.height = 1;
  _local_map.is_dense = true;

  pcl::toROSMsg(_local_map, _local_map_pcd);
  _local_map_pcd.header.frame_id = "map";
  _local_map_pcd.header.stamp = _odom.header.stamp;

  pub_cloud.publish(_local_map_pcd);
}

void rcvLocalPointCloudCallBack(
    const sensor_msgs::PointCloud2& pointcloud_map) {
  // do nothing, fix later
}

int main(int argc, char** argv) {
  ros::init(argc, argv, "pcl_render");
  ros::NodeHandle nh("~");

  nh.getParam("sensing_horizon", sensing_horizon);
  nh.getParam("sensing_rate", sensing_rate);
  nh.getParam("estimation_rate", estimation_rate);

  nh.getParam("map/x_size", _x_size);
  nh.getParam("map/y_size", _y_size);
  nh.getParam("map/z_size", _z_size);

  nh.param("cam_width", cam_width, 640);
  nh.param("cam_height", cam_height, 480);
  nh.param("cam_fx", cam_fx, 387.229248046875);
  nh.param("cam_fy", cam_fy, 387.229248046875);
  nh.param("cam_cx", cam_cx, 321.04638671875);
  nh.param("cam_cy", cam_cy, 243.44969177246094);
  nh.param("cam_downsample", cam_downsample, 4);
  nh.param("cam_min_range", cam_min_range, 0.1);
  // Same extrinsic as GridMap::cam2body_: camera looks along body +x.
  R_body_cam << 0.0, 0.0, 1.0,
               -1.0, 0.0, 0.0,
                0.0, -1.0, 0.0;
  t_body_cam << 0.0, 0.0, -0.02;

  // subscribe point cloud
  global_map_sub = nh.subscribe("global_map", 1, rcvGlobalPointCloudCallBack);
  local_map_sub = nh.subscribe("local_map", 1, rcvLocalPointCloudCallBack);
  odom_sub = nh.subscribe("odometry", 50, rcvOdometryCallbck);

  // publisher depth image and color image
  pub_cloud =
      nh.advertise<sensor_msgs::PointCloud2>("/pcl_render_node/cloud", 10);

  double sensing_duration = 1.0 / sensing_rate * 2.5;

  local_sensing_timer =
      nh.createTimer(ros::Duration(sensing_duration), renderSensedPoints);

  _resolution = 0.1;  // leaf size of _voxel_sampler
  _inv_resolution = 1.0 / _resolution;

  _gl_xl = -_x_size / 2.0;
  _gl_yl = -_y_size / 2.0;
  _gl_zl = 0.0;

  _GLX_SIZE = (int)(_x_size * _inv_resolution);
  _GLY_SIZE = (int)(_y_size * _inv_resolution);
  _GLZ_SIZE = (int)(_z_size * _inv_resolution);

  ros::Rate rate(100);
  bool status = ros::ok();
  while (status) {
    ros::spinOnce();
    status = ros::ok();
    rate.sleep();
  }
}
