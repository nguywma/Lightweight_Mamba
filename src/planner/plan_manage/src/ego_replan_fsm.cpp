
#include <plan_manage/ego_replan_fsm.h>

#include <limits>
#include <cmath>
#include <pcl_conversions/pcl_conversions.h>

namespace ego_planner
{

  void EGOReplanFSM::init(ros::NodeHandle &nh)
  {
    current_wp_ = 0;
    exec_state_ = FSM_EXEC_STATE::INIT;
    have_target_ = false;
    have_odom_ = false;

    /*  fsm param  */
    nh.param("fsm/flight_type", target_type_, -1);
    nh.param("fsm/thresh_replan", replan_thresh_, -1.0);
    nh.param("fsm/thresh_no_replan", no_replan_thresh_, -1.0);
    nh.param("fsm/planning_horizon", planning_horizen_, -1.0);
    nh.param("fsm/planning_horizen_time", planning_horizen_time_, -1.0);
    nh.param("fsm/emergency_time_", emergency_time_, 1.0);

    /* metric params (all overridable from launch) */
    nh.param("fsm/drone_radius", drone_radius_, 0.2);
    nh.param("fsm/safety_clearance", safety_clearance_thresh_, 0.5);
    nh.param("fsm/mission_timeout", mission_timeout_, 60.0);
    std::string gt_cloud_topic;
    nh.param<std::string>("fsm/ground_truth_cloud_topic", gt_cloud_topic, "/map_generator/global_cloud");

    nh.param("fsm/waypoint_num", waypoint_num_, -1);
    for (int i = 0; i < waypoint_num_; i++)
    {
      nh.param("fsm/waypoint" + to_string(i) + "_x", waypoints_[i][0], -1.0);
      nh.param("fsm/waypoint" + to_string(i) + "_y", waypoints_[i][1], -1.0);
      nh.param("fsm/waypoint" + to_string(i) + "_z", waypoints_[i][2], -1.0);
    }

    /* initialize main modules */
    visualization_.reset(new PlanningVisualization(nh));
    planner_manager_.reset(new EGOPlannerManager);
    planner_manager_->initPlanModules(nh, visualization_);

    /* callback */
    exec_timer_ = nh.createTimer(ros::Duration(0.01), &EGOReplanFSM::execFSMCallback, this);
    safety_timer_ = nh.createTimer(ros::Duration(0.05), &EGOReplanFSM::checkCollisionCallback, this);

    odom_sub_ = nh.subscribe("/odom_world", 1, &EGOReplanFSM::odometryCallback, this);

    /* ground-truth obstacle cloud for clearance/collision metrics (subscribe
       before the preset-target wait below so a single-shot latched publisher
       is not missed) */
    gt_cloud_sub_ = nh.subscribe(gt_cloud_topic, 1, &EGOReplanFSM::groundTruthMapCallback, this);

    bspline_pub_ = nh.advertise<ego_planner::Bspline>("/planning/bspline", 10);
    data_disp_pub_ = nh.advertise<ego_planner::DataDisp>("/planning/data_display", 100);

    if (target_type_ == TARGET_TYPE::MANUAL_TARGET)
      waypoint_sub_ = nh.subscribe("/waypoint_generator/waypoints", 1, &EGOReplanFSM::waypointCallback, this);
    else if (target_type_ == TARGET_TYPE::PRESET_TARGET)
    {
      ros::Duration(1.0).sleep();
      while (ros::ok() && !have_odom_)
        ros::spinOnce();
      planGlobalTrajbyGivenWps();
    }
    else
      cout << "Wrong target_type_ value! target_type_=" << target_type_ << endl;
  }

  void EGOReplanFSM::groundTruthMapCallback(const sensor_msgs::PointCloud2ConstPtr &msg)
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*msg, *cloud);
    if (cloud->points.empty())
      return;
    gt_kdtree_.setInputCloud(cloud);
    gt_map_ready_ = true;
  }

  void EGOReplanFSM::startMissionMetrics()
  {
    total_missions_++;
    mission_ongoing_ = true;
    had_collision_ = false;
    gt_collision_ = false;
    mission_timed_out_ = false;
    mission_start_time_ = ros::Time::now();
    mission_start_pos_ = odom_pos_;
    straight_line_dist_ = (end_pt_ - mission_start_pos_).norm();
    mission_plan_time_ = 0.0;
    mission_plan_calls_ = 0;
    prediction_opportunities_ = 0;
    prediction_avoiding_replans_ = 0;
    prediction_only_contacts_ = 0;
    emergency_stops_ = 0;
    mission_prediction_messages_ = planner_manager_->grid_map_ ? planner_manager_->grid_map_->predictionMessageCount() : 0;
    mission_prediction_voxels_ = planner_manager_->grid_map_ ? planner_manager_->grid_map_->predictionVoxelCount() : 0;
    max_vel_ = 0.0;
    total_dist_ = 0.0;
    energy_jerk_ = 0.0;
    min_clearance_ = std::numeric_limits<double>::infinity();
    clearance_time_weighted_sum_ = 0.0;
    clearance_time_total_ = 0.0;
    time_in_danger_ = 0.0;
    is_first_odom_in_mission_ = true;
  }

  void EGOReplanFSM::planGlobalTrajbyGivenWps()
  {
    std::vector<Eigen::Vector3d> wps(waypoint_num_);
    for (int i = 0; i < waypoint_num_; i++)
    {
      wps[i](0) = waypoints_[i][0];
      wps[i](1) = waypoints_[i][1];
      wps[i](2) = waypoints_[i][2];

      end_pt_ = wps.back();
    }

    startMissionMetrics();
    cout << "\033[1;33m[Metrics]: Mission Started (Preset Waypoints)\033[0m" << endl;

    bool success = planner_manager_->planGlobalTrajWaypoints(odom_pos_, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), wps, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());

    for (size_t i = 0; i < (size_t)waypoint_num_; i++)
    {
      visualization_->displayGoalPoint(wps[i], Eigen::Vector4d(0, 0.5, 0.5, 1), 0.3, i);
      ros::Duration(0.001).sleep();
    }

    if (success)
    {

      /*** display ***/
      constexpr double step_size_t = 0.1;
      int i_end = floor(planner_manager_->global_data_.global_duration_ / step_size_t);
      std::vector<Eigen::Vector3d> gloabl_traj(i_end);
      for (int i = 0; i < i_end; i++)
      {
        gloabl_traj[i] = planner_manager_->global_data_.global_traj_.evaluate(i * step_size_t);
      }

      end_vel_.setZero();
      have_target_ = true;
      have_new_target_ = true;

      /*** FSM ***/
      // if (exec_state_ == WAIT_TARGET)
      changeFSMExecState(GEN_NEW_TRAJ, "TRIG");
      // else if (exec_state_ == EXEC_TRAJ)
      //   changeFSMExecState(REPLAN_TRAJ, "TRIG");

      // visualization_->displayGoalPoint(end_pt_, Eigen::Vector4d(1, 0, 0, 1), 0.3, 0);
      ros::Duration(0.001).sleep();
      visualization_->displayGlobalPathList(gloabl_traj, 0.1, 0);
      ros::Duration(0.001).sleep();
    }
    else
    {
      ROS_ERROR("Unable to generate global trajectory!");
    }
  }

  void EGOReplanFSM::waypointCallback(const nav_msgs::PathConstPtr &msg)
  {
    if (msg->poses[0].pose.position.z < -0.1)
      return;

    cout << "Triggered!" << endl;
    trigger_ = true;
    init_pt_ = odom_pos_;

    bool success = false;
    end_pt_ << msg->poses[0].pose.position.x, msg->poses[0].pose.position.y, 1.0;
    
    startMissionMetrics();
    cout << "\033[1;33m[Metrics]: Mission Started (Manual Goal)\033[0m" << endl;

    success = planner_manager_->planGlobalTraj(odom_pos_, odom_vel_, Eigen::Vector3d::Zero(), end_pt_, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());

    visualization_->displayGoalPoint(end_pt_, Eigen::Vector4d(0, 0.5, 0.5, 1), 0.3, 0);

    if (success)
    {

      /*** display ***/
      constexpr double step_size_t = 0.1;
      int i_end = floor(planner_manager_->global_data_.global_duration_ / step_size_t);
      vector<Eigen::Vector3d> gloabl_traj(i_end);
      for (int i = 0; i < i_end; i++)
      {
        gloabl_traj[i] = planner_manager_->global_data_.global_traj_.evaluate(i * step_size_t);
      }

      end_vel_.setZero();
      have_target_ = true;
      have_new_target_ = true;

      /*** FSM ***/
      if (exec_state_ == WAIT_TARGET)
        changeFSMExecState(GEN_NEW_TRAJ, "TRIG");
      else if (exec_state_ == EXEC_TRAJ)
        changeFSMExecState(REPLAN_TRAJ, "TRIG");

      // visualization_->displayGoalPoint(end_pt_, Eigen::Vector4d(1, 0, 0, 1), 0.3, 0);
      visualization_->displayGlobalPathList(gloabl_traj, 0.1, 0);
    }
    else
    {
      ROS_ERROR("Unable to generate global trajectory!");
    }
  }

  void EGOReplanFSM::odometryCallback(const nav_msgs::OdometryConstPtr &msg)
  {
    odom_pos_(0) = msg->pose.pose.position.x;
    odom_pos_(1) = msg->pose.pose.position.y;
    odom_pos_(2) = msg->pose.pose.position.z;

    odom_vel_(0) = msg->twist.twist.linear.x;
    odom_vel_(1) = msg->twist.twist.linear.y;
    odom_vel_(2) = msg->twist.twist.linear.z;

    //odom_acc_ = estimateAcc( msg );

    odom_orient_.w() = msg->pose.pose.orientation.w;
    odom_orient_.x() = msg->pose.pose.orientation.x;
    odom_orient_.y() = msg->pose.pose.orientation.y;
    odom_orient_.z() = msg->pose.pose.orientation.z;

    if (mission_ongoing_)
    {
      /* Collision metric: scored on the path the drone actually flew, against
         the sensed map only (getInflateOccupancyRaw). Using the planning map
         here would score the predictions against themselves — a hallucinated
         voxel would be counted as a collision, and the planner is already
         steering away from every voxel in that same buffer. */
      if (!had_collision_ && planner_manager_ && planner_manager_->grid_map_ &&
          planner_manager_->grid_map_->getInflateOccupancyRaw(odom_pos_) == 1)
      {
        had_collision_ = true;
        ROS_ERROR("[Metrics]: collision with a sensed obstacle at (%.2f, %.2f, %.2f)",
                  odom_pos_(0), odom_pos_(1), odom_pos_(2));
        /* A collision is terminal: score the mission now so the benchmark
           moves on instead of waiting for the mission timeout. */
        printMetrics(false, "COLLISION");
        mission_ongoing_ = false;
        return;
      }
      if (planner_manager_ && planner_manager_->grid_map_ &&
          planner_manager_->grid_map_->getPredictionOnlyOccupancy(odom_pos_))
        ++prediction_only_contacts_;

      /* Ground-truth clearance: nearest distance from the drone to any real
         obstacle point. Neutral referee shared by baseline and prediction. */
      double clearance = -1.0;
      if (gt_map_ready_)
      {
        pcl::PointXYZ q(odom_pos_(0), odom_pos_(1), odom_pos_(2));
        std::vector<int> nn_idx(1);
        std::vector<float> nn_d2(1);
        if (gt_kdtree_.nearestKSearch(q, 1, nn_idx, nn_d2) > 0)
        {
          clearance = std::sqrt((double)nn_d2[0]);
          if (clearance < min_clearance_) min_clearance_ = clearance;
          if (clearance < drone_radius_) gt_collision_ = true;
        }
      }

      ros::Time now = ros::Time::now();
      if (is_first_odom_in_mission_)
      {
        last_odom_vel_ = odom_vel_;
        last_odom_acc_ = Eigen::Vector3d::Zero();
        last_odom_time_ = now;
        is_first_odom_in_mission_ = false;
      }
      else
      {
        double dt = (now - last_odom_time_).toSec();
        if (dt > 1e-4)
        {
          Eigen::Vector3d acc = (odom_vel_ - last_odom_vel_) / dt;
          Eigen::Vector3d jerk = (acc - last_odom_acc_) / dt;

          energy_jerk_ += jerk.squaredNorm() * dt;
          total_dist_ += odom_vel_.norm() * dt;
          if (odom_vel_.norm() > max_vel_) max_vel_ = odom_vel_.norm();

          /* time-weighted clearance (robust to variable odom rate) */
          if (clearance >= 0.0)
          {
            clearance_time_weighted_sum_ += clearance * dt;
            clearance_time_total_ += dt;
            if (clearance < safety_clearance_thresh_) time_in_danger_ += dt;
          }

          last_odom_vel_ = odom_vel_;
          last_odom_acc_ = acc;
          last_odom_time_ = now;
        }
      }
    }

    have_odom_ = true;
  }

  void EGOReplanFSM::changeFSMExecState(FSM_EXEC_STATE new_state, string pos_call)
  {

    if (new_state == exec_state_)
      continously_called_times_++;
    else
      continously_called_times_ = 1;

    static string state_str[7] = {"INIT", "WAIT_TARGET", "GEN_NEW_TRAJ", "REPLAN_TRAJ", "EXEC_TRAJ", "EMERGENCY_STOP"};
    int pre_s = int(exec_state_);
    exec_state_ = new_state;
    cout << "[" + pos_call + "]: from " + state_str[pre_s] + " to " + state_str[int(new_state)] << endl;

    /* An emergency stop is a recoverable event: the FSM brakes, then goes back
       to GEN_NEW_TRAJ and keeps flying. Count it instead of ending the mission;
       the mission is still scored by REACHED / COLLISION / TIMEOUT. */
    if (new_state == EMERGENCY_STOP && mission_ongoing_ && pre_s != EMERGENCY_STOP)
    {
      ++emergency_stops_;
    }
    else if (new_state == WAIT_TARGET && mission_ongoing_)
    {
      if (!had_collision_)
        success_missions_++;
      printMetrics(!had_collision_, had_collision_ ? "COLLISION" : "REACHED");
      mission_ongoing_ = false;
    }
  }

  std::pair<int, EGOReplanFSM::FSM_EXEC_STATE> EGOReplanFSM::timesOfConsecutiveStateCalls()
  {
    return std::pair<int, FSM_EXEC_STATE>(continously_called_times_, exec_state_);
  }

  void EGOReplanFSM::printFSMExecState()
  {
    static string state_str[7] = {"INIT", "WAIT_TARGET", "GEN_NEW_TRAJ", "REPLAN_TRAJ", "EXEC_TRAJ", "EMERGENCY_STOP"};

    cout << "[FSM]: state: " + state_str[int(exec_state_)] << endl;
  }

  void EGOReplanFSM::execFSMCallback(const ros::TimerEvent &e)
  {

    static int fsm_num = 0;
    fsm_num++;
    if (fsm_num == 100)
    {
      printFSMExecState();
      if (!have_odom_)
        cout << "no odom." << endl;
      if (!trigger_)
        cout << "wait for goal." << endl;
      fsm_num = 0;
    }

    /* mission watchdog: a run that neither reaches the goal nor emergency-stops
       (e.g. frozen / deadlocked) is scored as TIMEOUT rather than hanging. */
    if (mission_ongoing_ && !mission_timed_out_ &&
        (ros::Time::now() - mission_start_time_).toSec() > mission_timeout_)
    {
      mission_timed_out_ = true;
      ROS_ERROR("[Metrics]: mission TIMEOUT after %.1f s", mission_timeout_);
      printMetrics(false, had_collision_ ? "COLLISION" : "TIMEOUT");
      mission_ongoing_ = false; // set before EMERGENCY_STOP so we don't double-print
      have_target_ = false;
      changeFSMExecState(EMERGENCY_STOP, "TIMEOUT");
      return;
    }

    switch (exec_state_)
    {
    case INIT:
    {
      if (!have_odom_)
      {
        return;
      }
      if (!trigger_)
      {
        return;
      }
      changeFSMExecState(WAIT_TARGET, "FSM");
      break;
    }

    case WAIT_TARGET:
    {
      if (!have_target_)
        return;
      else
      {
        changeFSMExecState(GEN_NEW_TRAJ, "FSM");
      }
      break;
    }

    case GEN_NEW_TRAJ:
    {
      start_pt_ = odom_pos_;
      start_vel_ = odom_vel_;
      start_acc_.setZero();
      start_sample_time_ = ros::Time::now();

      // Eigen::Vector3d rot_x = odom_orient_.toRotationMatrix().block(0, 0, 3, 1);
      // start_yaw_(0)         = atan2(rot_x(1), rot_x(0));
      // start_yaw_(1) = start_yaw_(2) = 0.0;

      bool flag_random_poly_init;
      if (timesOfConsecutiveStateCalls().first == 1)
        flag_random_poly_init = false;
      else
        flag_random_poly_init = true;

      bool success = callReboundReplan(true, flag_random_poly_init);
      if (success)
      {

        changeFSMExecState(EXEC_TRAJ, "FSM");
        flag_escape_emergency_ = true;
      }
      else
      {
        changeFSMExecState(GEN_NEW_TRAJ, "FSM");
      }
      break;
    }

    case REPLAN_TRAJ:
    {

      if (planFromCurrentTraj())
      {
        changeFSMExecState(EXEC_TRAJ, "FSM");
      }
      else
      {
        changeFSMExecState(REPLAN_TRAJ, "FSM");
      }

      break;
    }

    case EXEC_TRAJ:
    {
      /* determine if need to replan */
      LocalTrajData *info = &planner_manager_->local_data_;
      ros::Time time_now = ros::Time::now();
      double t_cur = (time_now - info->start_time_).toSec();
      t_cur = min(info->duration_, t_cur);

      Eigen::Vector3d pos = info->position_traj_.evaluateDeBoorT(t_cur);

      /* && (end_pt_ - pos).norm() < 0.5 */
      if (t_cur > info->duration_ - 1e-2)
      {
        have_target_ = false;
        changeFSMExecState(WAIT_TARGET, "FSM");
        return;
      }
      else if ((end_pt_ - pos).norm() < no_replan_thresh_)
      {
        return;
      }
      else if ((info->start_pos_ - pos).norm() < replan_thresh_)
      {
        cout << "near start" << endl;
        return;
      }
      else
      {
        changeFSMExecState(REPLAN_TRAJ, "FSM");
      }
      break;
    }

    case EMERGENCY_STOP:
    {

      if (flag_escape_emergency_) // Avoiding repeated calls
      {
        /* Stop where the trajectory server is commanding the drone right now.
           odom_pos_ can be stale here: the node is single-threaded, so no odom
           arrives while replanning runs, and the drone keeps flying the old
           trajectory meanwhile. Stopping at odom_pos_ snapped it back. */
        LocalTrajData *info = &planner_manager_->local_data_;
        Eigen::Vector3d stop_pos = odom_pos_;
        if (info->start_time_.toSec() > 1e-5)
        {
          double t_cmd = std::max(0.0, std::min((ros::Time::now() - info->start_time_).toSec(), info->duration_));
          stop_pos = info->position_traj_.evaluateDeBoorT(t_cmd);
        }
        callEmergencyStop(stop_pos);
      }
      else
      {
        if (odom_vel_.norm() < 0.1)
          changeFSMExecState(GEN_NEW_TRAJ, "FSM");
      }

      flag_escape_emergency_ = false;
      break;
    }
    }

    data_disp_.header.stamp = ros::Time::now();
    data_disp_pub_.publish(data_disp_);
  }

  bool EGOReplanFSM::planFromCurrentTraj()
  {

    LocalTrajData *info = &planner_manager_->local_data_;
    ros::Time time_now = ros::Time::now();
    double t_cur = (time_now - info->start_time_).toSec();

    //cout << "info->velocity_traj_=" << info->velocity_traj_.get_control_points() << endl;

    start_pt_ = info->position_traj_.evaluateDeBoorT(t_cur);
    start_vel_ = info->velocity_traj_.evaluateDeBoorT(t_cur);
    start_acc_ = info->acceleration_traj_.evaluateDeBoorT(t_cur);
    start_sample_time_ = time_now;

    bool success = callReboundReplan(false, false);

    if (!success)
    {
      success = callReboundReplan(true, false);
      //changeFSMExecState(EXEC_TRAJ, "FSM");
      if (!success)
      {
        success = callReboundReplan(true, true);
        if (!success)
        {
          return false;
        }
      }
    }

    return true;
  }

  void EGOReplanFSM::checkCollisionCallback(const ros::TimerEvent &e)
  {
    LocalTrajData *info = &planner_manager_->local_data_;
    auto map = planner_manager_->grid_map_;

    if (exec_state_ == WAIT_TARGET || info->start_time_.toSec() < 1e-5)
      return;

    /* ---------- check trajectory ---------- */
    constexpr double time_step = 0.01;
    double t_cur = (ros::Time::now() - info->start_time_).toSec();
    double t_2_3 = info->duration_ * 2 / 3;
    for (double t = t_cur; t < info->duration_; t += time_step)
    {
      if (t_cur < t_2_3 && t >= t_2_3) // If t_cur < t_2_3, only the first 2/3 partition of the trajectory is considered valid and will get checked.
        break;

      const Eigen::Vector3d position = info->position_traj_.evaluateDeBoorT(t);

      // Planning-side check only: this map is sensed obstacles UNION network
      // predictions, so it is the right thing to replan against but the wrong
      // thing to score with. The collision metric lives in odometryCallback().
      if (map->getInflateOccupancy(position) > 0)
      {
        /* Prediction-effectiveness instrumentation.
           Every time the safety check finds the current trajectory blocked is
           an "opportunity" to react. If the blocking voxel is one the network
           predicted but the raw sensor has NOT yet observed
           (getPredictionOnlyOccupancy), then this reaction was triggered by the
           prediction — the sensor alone would not have caught it yet. The ratio
           avoiding_replans_/opportunities_ is the fraction of hazard reactions
           for which prediction provided early warning. */
        ++prediction_opportunities_;
        if (map->getPredictionOnlyOccupancy(position))
          ++prediction_avoiding_replans_;

        if (planFromCurrentTraj()) // Make a chance
        {
          changeFSMExecState(EXEC_TRAJ, "SAFETY");
          return;
        }
        else
        {
          if (t - t_cur < emergency_time_) // 0.8s of emergency time
          {
            ROS_WARN("Suddenly discovered obstacles. emergency stop! time=%f", t - t_cur);
            changeFSMExecState(EMERGENCY_STOP, "SAFETY");
          }
          else
          {
            ROS_WARN("current traj in collision, replan.");
            changeFSMExecState(REPLAN_TRAJ, "SAFETY");
          }
          return;
        }
        break;
      }
    }
  }

  bool EGOReplanFSM::callReboundReplan(bool flag_use_poly_init, bool flag_randomPolyTraj)
  {

    getLocalTarget();

    ros::Time t_p_start = ros::Time::now();
    bool plan_success =
        planner_manager_->reboundReplan(start_pt_, start_vel_, start_acc_, local_target_pt_, local_target_vel_, (have_new_target_ || flag_use_poly_init), flag_randomPolyTraj, start_sample_time_);
    double p_time = (ros::Time::now() - t_p_start).toSec();

    total_plan_time_ += p_time;
    total_plan_calls_++;
    mission_plan_time_ += p_time;
    mission_plan_calls_++;

    have_new_target_ = false;

    cout << "final_plan_success=" << plan_success << endl;

    if (plan_success)
    {

      auto info = &planner_manager_->local_data_;

      /* publish traj */
      ego_planner::Bspline bspline;
      bspline.order = 3;
      bspline.start_time = info->start_time_;
      bspline.traj_id = info->traj_id_;

      Eigen::MatrixXd pos_pts = info->position_traj_.getControlPoint();
      bspline.pos_pts.reserve(pos_pts.cols());
      for (int i = 0; i < pos_pts.cols(); ++i)
      {
        geometry_msgs::Point pt;
        pt.x = pos_pts(0, i);
        pt.y = pos_pts(1, i);
        pt.z = pos_pts(2, i);
        bspline.pos_pts.push_back(pt);
      }

      Eigen::VectorXd knots = info->position_traj_.getKnot();
      bspline.knots.reserve(knots.rows());
      for (int i = 0; i < knots.rows(); ++i)
      {
        bspline.knots.push_back(knots(i));
      }

      bspline_pub_.publish(bspline);

      visualization_->displayOptimalList(info->position_traj_.get_control_points(), 0);
    }

    return plan_success;
  }

  bool EGOReplanFSM::callEmergencyStop(Eigen::Vector3d stop_pos)
  {

    planner_manager_->EmergencyStop(stop_pos);

    auto info = &planner_manager_->local_data_;

    /* publish traj */
    ego_planner::Bspline bspline;
    bspline.order = 3;
    bspline.start_time = info->start_time_;
    bspline.traj_id = info->traj_id_;

    Eigen::MatrixXd pos_pts = info->position_traj_.getControlPoint();
    bspline.pos_pts.reserve(pos_pts.cols());
    for (int i = 0; i < pos_pts.cols(); ++i)
    {
      geometry_msgs::Point pt;
      pt.x = pos_pts(0, i);
      pt.y = pos_pts(1, i);
      pt.z = pos_pts(2, i);
      bspline.pos_pts.push_back(pt);
    }

    Eigen::VectorXd knots = info->position_traj_.getKnot();
    bspline.knots.reserve(knots.rows());
    for (int i = 0; i < knots.rows(); ++i)
    {
      bspline.knots.push_back(knots(i));
    }

    bspline_pub_.publish(bspline);

    return true;
  }

  void EGOReplanFSM::getLocalTarget()
  {
    double t;

    double t_step = planning_horizen_ / 20 / planner_manager_->pp_.max_vel_;
    double dist_min = 9999, dist_min_t = 0.0;

    // Recover the progress cursor if the vehicle is no longer near its expected
    // point on the global trajectory (for example after a tracking deviation).
    double progress_t = std::max(0.0, std::min(planner_manager_->global_data_.last_progress_time_,
                                               planner_manager_->global_data_.global_duration_));
    if ((planner_manager_->global_data_.getPosition(progress_t) - start_pt_).norm() > planning_horizen_)
    {
      double min_dist_sq = std::numeric_limits<double>::max();
      for (double candidate_t = 0.0; candidate_t <= planner_manager_->global_data_.global_duration_; candidate_t += t_step)
      {
        const double candidate_dist = (planner_manager_->global_data_.getPosition(candidate_t) - start_pt_).squaredNorm();
        if (candidate_dist < min_dist_sq)
        {
          min_dist_sq = candidate_dist;
          dist_min_t = candidate_t;
        }
      }
      planner_manager_->global_data_.last_progress_time_ = dist_min_t;
      dist_min = 9999;
    }

    for (t = planner_manager_->global_data_.last_progress_time_; t < planner_manager_->global_data_.global_duration_; t += t_step)
    {
      Eigen::Vector3d pos_t = planner_manager_->global_data_.getPosition(t);
      double dist = (pos_t - start_pt_).norm();
      if (dist < dist_min)
      {
        dist_min = dist;
        dist_min_t = t;
      }
      if (dist >= planning_horizen_)
      {
        local_target_pt_ = pos_t;
        planner_manager_->global_data_.last_progress_time_ = dist_min_t;
        break;
      }
    }
    if (t > planner_manager_->global_data_.global_duration_) // Last global point
    {
      local_target_pt_ = end_pt_;
    }

    if ((end_pt_ - local_target_pt_).norm() < (planner_manager_->pp_.max_vel_ * planner_manager_->pp_.max_vel_) / (2 * planner_manager_->pp_.max_acc_))
    {
      // local_target_vel_ = (end_pt_ - init_pt_).normalized() * planner_manager_->pp_.max_vel_ * (( end_pt_ - local_target_pt_ ).norm() / ((planner_manager_->pp_.max_vel_*planner_manager_->pp_.max_vel_)/(2*planner_manager_->pp_.max_acc_)));
      // cout << "A" << endl;
      local_target_vel_ = Eigen::Vector3d::Zero();
    }
    else
    {
      local_target_vel_ = planner_manager_->global_data_.getVelocity(t);
      // cout << "AA" << endl;
    }
  }

  void EGOReplanFSM::printMetrics(bool success, const std::string &outcome)
  {
    double f_time = (ros::Time::now() - mission_start_time_).toSec();
    if (success) accum_flight_time_ += f_time;

    cout << "\033[1;32m" << endl;
    cout << "========= Navigation Metrics =========" << endl;
    cout << "Mission #" << total_missions_ << ": " << (success ? "SUCCESS" : "FAIL") << " (" << outcome << ")" << endl;
    cout << "Outcome: " << outcome << endl;
    cout << "Flying Time: " << f_time << " s" << endl;
    cout << "Total Distance: " << total_dist_ << " m" << endl;
    cout << "Velocity - Max: " << max_vel_ << " m/s | Avg: " << (f_time > 0.1 ? total_dist_ / f_time : 0.0) << " m/s" << endl;
    cout << "Energy (Jerk Integral): " << energy_jerk_ << endl;
    cout << "Planning Time (this mission): " << mission_plan_time_ << " s | Avg: " << (mission_plan_calls_ > 0 ? mission_plan_time_ / mission_plan_calls_ * 1000.0 : 0.0) << " ms" << endl;
    cout << "--------------------------------------" << endl;
    cout << "Overall Success Rate: " << success_missions_ << "/" << total_missions_ << " (" << (double)success_missions_ / total_missions_ * 100.0 << "%)" << endl;
    if (success_missions_ > 0)
      cout << "Avg Flying Time (success): " << accum_flight_time_ / success_missions_ << " s" << endl;
    cout << "Total Planning Time: " << total_plan_time_ << " s | Avg: " << (total_plan_calls_ > 0 ? total_plan_time_ / total_plan_calls_ * 1000.0 : 0.0) << " ms" << endl;
    /* ---- Prediction-effectiveness instrumentation ----
       messages/voxels are cumulative counters in the grid map, so report the
       delta accrued during THIS mission. age_s is the staleness of the newest
       prediction at mission end (large/absent => prediction was off or stale). */
    uint64_t pred_msgs = 0, pred_vox = 0;
    double pred_age = 0.0;
    if (planner_manager_ && planner_manager_->grid_map_)
    {
      pred_msgs = planner_manager_->grid_map_->predictionMessageCount() - mission_prediction_messages_;
      pred_vox  = planner_manager_->grid_map_->predictionVoxelCount() - mission_prediction_voxels_;
      pred_age  = planner_manager_->grid_map_->predictionAge();
    }
    /* ---- Ground-truth safety metrics (graded; the key lever for showing that
       prediction keeps the drone further from obstacles even when neither
       method actually collides). -1 => no ground-truth cloud was received. */
    double min_clear_out = std::isinf(min_clearance_) ? -1.0 : min_clearance_;
    double mean_clear_out = clearance_time_total_ > 0.0 ? clearance_time_weighted_sum_ / clearance_time_total_ : -1.0;
    double path_eff = straight_line_dist_ > 1e-6 ? total_dist_ / straight_line_dist_ : -1.0;
    cout << "Collisions: " << (had_collision_ ? 1 : 0) << endl;
    cout << "Emergency Stops: " << emergency_stops_ << endl;
    cout << "Safety Metrics: gt_collision=" << (gt_collision_ ? 1 : 0)
         << " min_clearance_m=" << min_clear_out
         << " mean_clearance_m=" << mean_clear_out
         << " time_in_danger_s=" << time_in_danger_
         << " clearance_valid=" << (gt_map_ready_ ? 1 : 0) << endl;
    cout << "Path Metrics: straight_line_m=" << straight_line_dist_
         << " path_efficiency=" << path_eff << endl;
    cout << "Prediction Metrics: messages=" << pred_msgs
         << " voxels=" << pred_vox
         << " opportunities=" << prediction_opportunities_
         << " avoiding_replans=" << prediction_avoiding_replans_
         << " prediction_only_contacts=" << prediction_only_contacts_
         << " age_s=" << pred_age << endl;
    cout << "======================================" << endl;
    cout << "\033[0m" << endl;
  }

} // namespace ego_planner
