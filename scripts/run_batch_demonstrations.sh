#!/bin/bash
# Run gen_demonstration_metaworld.sh for a list of task configurations
#
# Usage:
#   bash scripts/run_batch_demonstrations.sh scripts/task_list.txt
#
# The task list file should contain one task name per line, e.g.:
#   basketball
#   pick-place
#   pick-place-wall
#   push
#   reach

# Check if task list file is provided
if [ -z "$1" ]; then
    echo "Usage: bash scripts/run_batch_demonstrations.sh <task_list_file>"
    echo "Example: bash scripts/run_batch_demonstrations.sh scripts/task_list.txt"
    exit 1
fi

TASK_LIST_FILE=$1

# Check if task list file exists
if [ ! -f "$TASK_LIST_FILE" ]; then
    echo "Error: Task list file '$TASK_LIST_FILE' not found!"
    exit 1
fi

echo "=========================================="
echo "Running batch demonstrations"
echo "Task list file: $TASK_LIST_FILE"
echo "=========================================="

# Count total tasks
TOTAL_TASKS=$(grep -v '^#' "$TASK_LIST_FILE" | grep -v '^$' | wc -l)
CURRENT_TASK=0

# Read task list file line by line
while IFS= read -r task_name || [ -n "$task_name" ]; do
    # Skip empty lines and comments
    if [ -z "$task_name" ] || [[ "$task_name" == \#* ]]; then
        continue
    fi
    
    CURRENT_TASK=$((CURRENT_TASK + 1))
    
    echo ""
    echo "=========================================="
    echo "[$CURRENT_TASK/$TOTAL_TASKS] Running task: $task_name"
    echo "=========================================="
    
    # Run the demonstration generation script
    bash scripts/gen_demonstration_metaworld.sh "$task_name"
    
    # Check if the script succeeded
    if [ $? -eq 0 ]; then
        echo "[$CURRENT_TASK/$TOTAL_TASKS] Task '$task_name' completed successfully!"
    else
        echo "[$CURRENT_TASK/$TOTAL_TASKS] Task '$task_name' failed with exit code $?"
    fi
    
done < "$TASK_LIST_FILE"

echo ""
echo "=========================================="
echo "Batch demonstration generation complete!"
echo "Processed $CURRENT_TASK tasks"
echo "=========================================="

