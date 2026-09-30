#!/bin/bash

LOG_FILE="benchmark_perception_results.txt"
RAW_OUTPUT="raw_ros_logs_perception.tmp"

echo "Seed, Success, FlyingTime, Distance, MaxVel, AvgVel, Jerk, PlanningTime" > $LOG_FILE

# Source workspace
source ~/drone/HE-Nav/devel/setup.bash

echo "Starting perception node..."
roslaunch perception inference.launch > perception.log 2>&1 &
PERCEPTION_PID=$!

echo "Waiting 30s for model to load..."
sleep 30

for seed in {10..20}
do
    echo "Running Experiment with Seed: $seed..."
    
    timeout 200s roslaunch ego_planner simple_run.launch map_seed:=$seed > $RAW_OUTPUT 2>&1
    
    # Extract Metrics
    SUCCESS=$(grep -c "Mission #.*: SUCCESS" $RAW_OUTPUT)
    FTIME=$(grep "Flying Time:" $RAW_OUTPUT | awk '{print $3}')
    DIST=$(grep "Total Distance:" $RAW_OUTPUT | awk '{print $3}')
    MVEL=$(grep "Velocity - Max:" $RAW_OUTPUT | awk '{print $4}')
    AVEL=$(grep "Velocity - Max:" $RAW_OUTPUT | awk '{print $7}')
    JERK=$(grep "Energy (Jerk Integral):" $RAW_OUTPUT | awk '{print $4}')
    PTIME=$(grep "Planning Time (this mission):" $RAW_OUTPUT | awk '{print $5}')

    echo "$seed, $SUCCESS, ${FTIME:-0}, ${DIST:-0}, ${MVEL:-0}, ${AVEL:-0}, ${JERK:-0}, ${PTIME:-0}" >> $LOG_FILE
    
    # Clean up simulation nodes ONLY (keep perception alive)
    killall -9 rosmaster gzserver gzclient 2>/dev/null
    sleep 2
done

echo "Stopping perception node..."
kill -9 $PERCEPTION_PID 2>/dev/null

echo "--------------------------------------"
echo "Benchmark Complete. Results saved to $LOG_FILE"

# Calculate Final Averages
awk -F', ' 'NR>1 {
    count++; succ+=$2; ftime+=$3; dist+=$4; jerk+=$7
} 
END {
    print "Final Success Rate: "(succ/count)*100"%"; 
    print "Avg Flying Time: "ftime/count"s"; 
    print "Avg Distance: "dist/count"m";
    print "Avg Jerk: "jerk/count
}' $LOG_FILE