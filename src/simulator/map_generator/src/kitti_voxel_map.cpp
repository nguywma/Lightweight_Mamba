/**
 * kitti_voxel_map_publisher.cpp
 *
 * ROS node: SemanticKITTI voxel map publisher for HE-Nav simulation.
 *
 * PURPOSE
 * -------
 * Replaces random_forest_sensing.cpp (the procedural pillar+ring generator) with a
 * publisher that loads a real SemanticKITTI voxel grid from a .bin file and publishes
 * it on /map_generator/global_cloud.  All downstream nodes (pcl_render_node, GridMap,
 * EGO-Planner) consume that topic unchanged — no modifications required there.
 *
 * VOXEL FORMAT (SemanticKITTI v1.1)
 * -----------------------------------
 * Each .bin file stores 256 × 256 × 32 = 2,097,152 uint8_t occupancy bytes.
 * A voxel is occupied when its byte value == 1 (free == 0, invalid/occluded == 255).
 * The dataset grid layout (from semantic-kitti.yaml):
 *   X ∈ [0,   51.2] m  →  256 voxels  @  0.2 m/voxel
 *   Y ∈ [-25.6, 25.6] m →  256 voxels  @  0.2 m/voxel
 *   Z ∈ [-2.0,  4.4] m  →  32  voxels  @  0.2 m/voxel
 * Flat index: idx = vx * 256 * 32 + vy * 32 + vz  (C-order, X-major)
 *
 * COORDINATE REMAPPING
 * ---------------------
 * The KITTI scene is centred and scaled to fit the simulation arena
 * [-20, +20] × [-20, +20] × [-0.1, +3.0] m.
 *
 * Raw metric position (before remapping):
 *   kx = vx * 0.2          (X ∈ [0, 51.2])
 *   ky = vy * 0.2 - 25.6   (Y ∈ [-25.6, +25.6])
 *   kz = vz * 0.2 - 2.0    (Z ∈ [-2.0, +4.4])
 *
 * Arena remapping (shifts and clips to drone arena bounds):
 *   wx = kx - 25.6 + ARENA_OFFSET_X   → roughly centred on X=0
 *   wy = ky + ARENA_OFFSET_Y           → Y already centred
 *   wz = kz + 2.0 + ARENA_OFFSET_Z    → lift so ground == 0
 *
 * Default offset values place the usable KITTI scene within [-20,+20]^2 × [0,3] m,
 * with the drone spawn at (-18, 0, 1) m inside free space.
 *
 * ROS PARAMETERS
 * ---------------
 *   ~voxel_file      (string) — absolute path to the .bin voxel file to load
 *   ~sequence        (string) — SemanticKITTI sequence folder (e.g. "00")
 *   ~frame_id        (string) — "world"  (default: world)
 *   ~publish_rate    (double) — Hz at which the cloud is re-published (default: 1.0)
 *   ~occupied_only   (bool)   — if true only publish value==1; false includes all non-zero (default: true)
 *   ~arena_offset_x  (double) — X-shift applied after KITTI→metric conversion (default: -5.6)
 *   ~arena_offset_y  (double) — Y-shift (default: 0.0)
 *   ~arena_offset_z  (double) — Z-shift (default: 0.0)
 *   ~min_z           (double) — clip points below this world-Z (default: -0.1)
 *   ~max_z           (double) — clip points above this world-Z (default:  3.0)
 *
 * PUBLISHED TOPICS
 *   /map_generator/global_cloud  (sensor_msgs/PointCloud2, frame: world)
 *
 * ORIGINAL FILE (backed up as random_forest_sensing.cpp.bak)
 * ---------------------------------------------------------------
 * random_forest_sensing.cpp generates a procedural random environment
 * of cylindrical pillars and circular rings seeded by an integer.
 * It is kept as-is and can be re-enabled by reverting CMakeLists.txt.
 */

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include <fstream>
#include <string>
#include <vector>
#include <cstdint>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <algorithm>

// ── SemanticKITTI voxel grid constants ──────────────────────────────────────
static constexpr int    KITTI_NX    = 256;
static constexpr int    KITTI_NY    = 256;
static constexpr int    KITTI_NZ    = 32;
static constexpr size_t KITTI_TOTAL = KITTI_NX * KITTI_NY * KITTI_NZ;  // 2,097,152 bytes
static constexpr double KITTI_RES   = 0.2;   // metres per voxel (all axes)
static constexpr double KITTI_X0    = 0.0;   // X origin (metres)
static constexpr double KITTI_Y0    = -25.6; // Y origin (metres)
static constexpr double KITTI_Z0    = -2.0;  // Z origin (metres)

// ── Inline flat-index accessor ───────────────────────────────────────────────
// SemanticKITTI stores voxels in X-major (C) order: idx = vx*(NY*NZ) + vy*NZ + vz
inline size_t kittiIdx(int vx, int vy, int vz) {
    return static_cast<size_t>(vx) * (KITTI_NY * KITTI_NZ)
         + static_cast<size_t>(vy) * KITTI_NZ
         + static_cast<size_t>(vz);
}

// ── Global state ─────────────────────────────────────────────────────────────
pcl::PointCloud<pcl::PointXYZ> g_cloud;
sensor_msgs::PointCloud2       g_cloud_msg;
ros::Publisher                 g_pub;

// ── Load voxel file and build point cloud ────────────────────────────────────
bool loadVoxelFile(const std::string& path,
                   double arena_offset_x,
                   double arena_offset_y,
                   double arena_offset_z,
                   double min_z,
                   double max_z,
                   bool   occupied_only,
                   bool   auto_ground,
                   double ground_top_z)
{
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) {
        ROS_ERROR("[KittiVoxelMap] Cannot open voxel file: %s", path.c_str());
        return false;
    }

    // Get file size
    ifs.seekg(0, std::ios::end);
    std::streamsize file_size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);

    std::vector<uint8_t> raw_data(file_size);
    ifs.read(reinterpret_cast<char*>(raw_data.data()), file_size);
    ifs.close();

    std::vector<uint8_t> data(KITTI_TOTAL, 0);
    std::vector<uint16_t> semantic;   // raw SemanticKITTI ids, only for .label input

    if (static_cast<size_t>(file_size) == KITTI_TOTAL / 8) {
        // Bit-packed format (.bin sparse input)
        for (size_t i = 0; i < raw_data.size(); ++i) {
            uint8_t b = raw_data[i];
            data[i * 8 + 0] = (b >> 7) & 1;
            data[i * 8 + 1] = (b >> 6) & 1;
            data[i * 8 + 2] = (b >> 5) & 1;
            data[i * 8 + 3] = (b >> 4) & 1;
            data[i * 8 + 4] = (b >> 3) & 1;
            data[i * 8 + 5] = (b >> 2) & 1;
            data[i * 8 + 6] = (b >> 1) & 1;
            data[i * 8 + 7] = (b >> 0) & 1;
        }
    } else if (static_cast<size_t>(file_size) == KITTI_TOTAL * 2) {
        // 16-bit labels (.label dense ground truth completion)
        const uint16_t* labels = reinterpret_cast<const uint16_t*>(raw_data.data());
        semantic.assign(labels, labels + KITTI_TOTAL);
        for (size_t i = 0; i < KITTI_TOTAL; ++i) {
            uint16_t semantic_label = labels[i] & 0xFFFF;
            // 0 is free, 255 is invalid/unknown. Anything else is an obstacle.
            if (semantic_label > 0 && semantic_label != 255) {
                data[i] = 1;
            } else {
                data[i] = 0;
            }
        }
    } else if (static_cast<size_t>(file_size) == KITTI_TOTAL) {
        // Byte-per-voxel format
        data = std::move(raw_data);
    } else {
        ROS_WARN("[KittiVoxelMap] Unexpected file size: %zu. Proceeding as raw bytes.", static_cast<size_t>(file_size));
        size_t copy_size = std::min(static_cast<size_t>(file_size), KITTI_TOTAL);
        std::copy(raw_data.begin(), raw_data.begin() + copy_size, data.begin());
    }

    // ── Vertical placement ───────────────────────────────────────────────────
    // With labels, put the top face of the ground (road, parking, sidewalk,
    // other-ground, lane-marking, terrain) at ground_top_z: the median over
    // columns of the highest ground voxel. Otherwise use the fixed offset.
    double z_shift = 2.0 + arena_offset_z;
    if (auto_ground && !semantic.empty()) {
        std::vector<int> tops;
        for (int vx = 0; vx < KITTI_NX; ++vx)
            for (int vy = 0; vy < KITTI_NY; ++vy)
                for (int vz = KITTI_NZ - 1; vz >= 0; --vz) {
                    const uint16_t l = semantic[kittiIdx(vx, vy, vz)];
                    if (l == 40 || l == 44 || l == 48 || l == 49 || l == 60 || l == 72) {
                        tops.push_back(vz);
                        break;
                    }
                }
        if (!tops.empty()) {
            std::nth_element(tops.begin(), tops.begin() + tops.size() / 2, tops.end());
            const double ground_top_k = KITTI_Z0 + (tops[tops.size() / 2] + 1) * KITTI_RES;
            z_shift = ground_top_z - ground_top_k;
            ROS_INFO("[KittiVoxelMap] Ground top at KITTI z=%.2f m -> world z=%.2f m",
                     ground_top_k, ground_top_z);
        } else {
            ROS_WARN("[KittiVoxelMap] No ground labels found; using arena_offset_z.");
        }
    }

    // ── Voxels → points ─────────────────────────────────────────────────────
    // Each 0.2 m voxel is emitted as its 8 sub-voxel centres at 0.1 m, the
    // resolution pcl_render_node splats with, so surfaces render solid
    // (a single point per voxel would leave see-through gaps).
    g_cloud.clear();
    int occupied_count = 0;
    const double sub = 0.5 * KITTI_RES;
    for (int vx = 0; vx < KITTI_NX; ++vx) {
        for (int vy = 0; vy < KITTI_NY; ++vy) {
            for (int vz = 0; vz < KITTI_NZ; ++vz) {
                const uint8_t val = data[kittiIdx(vx, vy, vz)];
                if (val == 0 || val == 255) continue;
                if (occupied_only && val != 1) continue;

                const double kx = KITTI_X0 + vx * KITTI_RES;
                const double ky = KITTI_Y0 + vy * KITTI_RES;
                const double kz = KITTI_Z0 + vz * KITTI_RES;
                ++occupied_count;
                for (int a = 0; a < 2; ++a)
                    for (int b = 0; b < 2; ++b)
                        for (int c = 0; c < 2; ++c) {
                            const double wx = kx + (a + 0.5) * sub - 25.6 + arena_offset_x;
                            const double wy = ky + (b + 0.5) * sub + arena_offset_y;
                            const double wz = kz + (c + 0.5) * sub + z_shift;
                            if (wz < min_z || wz > max_z) continue;
                            g_cloud.push_back(pcl::PointXYZ(wx, wy, wz));
                        }
            }
        }
    }

    g_cloud.width    = static_cast<uint32_t>(g_cloud.points.size());
    g_cloud.height   = 1;
    g_cloud.is_dense = true;

    pcl::toROSMsg(g_cloud, g_cloud_msg);
    g_cloud_msg.header.frame_id = "world";

    ROS_INFO("[KittiVoxelMap] Loaded %s — %d occupied voxels → %zu cloud points",
             path.c_str(), occupied_count, g_cloud.points.size());

    return !g_cloud.points.empty();
}

// ── Timer callback: republish the static cloud ───────────────────────────────
void publishCallback(const ros::TimerEvent&) {
    g_cloud_msg.header.stamp = ros::Time::now();
    g_pub.publish(g_cloud_msg);
}

// ── Main ─────────────────────────────────────────────────────────────────────
int main(int argc, char** argv)
{
    ros::init(argc, argv, "kitti_voxel_map_publisher");
    ros::NodeHandle nh("~");

    // ── Parameters ──────────────────────────────────────────────────────────
    std::string voxel_file;
    double publish_rate   = 1.0;
    double arena_offset_x = -5.6;   // shifts KITTI X centre into [-20,+20] range
    double arena_offset_y =  0.0;
    double arena_offset_z =  0.0;
    double min_z          = -0.1;
    double max_z          =  3.0;
    bool   occupied_only  = true;
    bool   auto_ground    = true;
    double ground_top_z   = -0.05;

    int frame_num = -1;
    std::string sequence = "08";
    std::string kitti_root = "/home/manh/dataset_voxel_map/dataset_voxel_map/dataset/sequences";

    nh.param<std::string>("voxel_file",     voxel_file,     "");
    nh.param<int>("frame_num",              frame_num,      frame_num);
    nh.param<std::string>("sequence",       sequence,       sequence);
    nh.param<std::string>("kitti_root",     kitti_root,     kitti_root);
    nh.param("publish_rate",   publish_rate,   publish_rate);
    nh.param("arena_offset_x", arena_offset_x, arena_offset_x);
    nh.param("arena_offset_y", arena_offset_y, arena_offset_y);
    nh.param("arena_offset_z", arena_offset_z, arena_offset_z);
    nh.param("min_z",          min_z,          min_z);
    nh.param("max_z",          max_z,          max_z);
    nh.param("occupied_only",  occupied_only,  occupied_only);
    nh.param("auto_ground",    auto_ground,    auto_ground);
    nh.param("ground_top_z",   ground_top_z,   ground_top_z);

    if (voxel_file.empty()) {
        if (frame_num >= 0) {
            char buf[64];
            snprintf(buf, sizeof(buf), "%06d.label", frame_num);
            voxel_file = kitti_root + "/" + sequence + "/voxels/" + std::string(buf);
            ROS_INFO("[KittiVoxelMap] Auto-constructed path: %s", voxel_file.c_str());
        } else {
            ROS_FATAL("[KittiVoxelMap] Must set either ~voxel_file (string) or ~frame_num (int).");
            return 1;
        }
    }

    // ── Publisher ────────────────────────────────────────────────────────────
    g_pub = nh.advertise<sensor_msgs::PointCloud2>("/map_generator/global_cloud", 1, /*latch=*/true);

    // ── Load the voxel file ──────────────────────────────────────────────────
    ros::Duration(0.5).sleep();   // brief wait for subscriber connections

    if (!loadVoxelFile(voxel_file, arena_offset_x, arena_offset_y, arena_offset_z,
                       min_z, max_z, occupied_only, auto_ground, ground_top_z)) {
        ROS_FATAL("[KittiVoxelMap] Failed to load voxel file. Exiting.");
        return 1;
    }

    // ── Publish immediately, then repeat at publish_rate ─────────────────────
    g_cloud_msg.header.stamp = ros::Time::now();
    g_pub.publish(g_cloud_msg);

    ros::Timer timer = nh.createTimer(ros::Duration(1.0 / publish_rate), publishCallback);

    ROS_INFO("[KittiVoxelMap] Publishing on /map_generator/global_cloud at %.1f Hz", publish_rate);

    ros::spin();
    return 0;
}
