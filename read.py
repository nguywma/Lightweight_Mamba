import os

def analyze_replan_logs(log_folder_path):
    # The two specific target strings we are looking for
    target_safety = "[SAFETY]: from EXEC_TRAJ to REPLAN_TRAJ"
    target_fsm = "[FSM]: from EXEC_TRAJ to REPLAN_TRAJ"

    total_safety_count = 0
    total_fsm_count = 0
    file_count = 0

    # Make sure the directory exists
    if not os.path.exists(log_folder_path):
        print(f"Error: The folder '{log_folder_path}' does not exist.")
        return

    print(f"Scanning logs in: {log_folder_path}...\n")

    # Iterate through every file in the target folder
    for filename in os.listdir(log_folder_path):
        filepath = os.path.join(log_folder_path, filename)

        # Ensure we are only reading files, not sub-folders
        if os.path.isfile(filepath):
            file_count += 1
            
            # errors='ignore' prevents crashes if a log file has weird binary characters
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
                for line in file:
                    if target_safety in line:
                        total_safety_count += 1
                    elif target_fsm in line:
                        total_fsm_count += 1

    # Guard against dividing by zero if the folder is empty
    if file_count == 0:
        print("No files found in the specified directory.")
        return

    # Calculate averages
    avg_safety = total_safety_count / file_count
    avg_fsm = total_fsm_count / file_count

    # Print out the formatted metrics
    print("-" * 40)
    print(f"Total Log Files Processed: {file_count}")
    print("-" * 40)
    print("1. Obstacle Avoidance Replans ([SAFETY])")
    print(f"   Total Count : {total_safety_count}")
    print(f"   Avg per file: {avg_safety:.2f}")
    print()
    print("2. Normal Continual Replans ([FSM])")
    print(f"   Total Count : {total_fsm_count}")
    print(f"   Avg per file: {avg_fsm:.2f}")
    print("-" * 40)

# --- Run the script ---
# Replace './my_log_folder' with the actual path to your 100 log files
log_dirs = ["result_no_predict/logs", "result_mamba/logs", "result_lbsc/logs", "results/logs"]
for dir in log_dirs:
    analyze_replan_logs(dir)
