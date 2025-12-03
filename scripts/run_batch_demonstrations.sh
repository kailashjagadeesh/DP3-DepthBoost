#!/bin/bash
# Run gen_demonstration_metaworld.sh for a list of task configurations in parallel

if [ -z "$1" ]; then
    echo "Usage: bash scripts/run_batch_demonstrations.sh <task_list_file>"
    exit 1
fi

TASK_LIST_FILE=$1

if [ ! -f "$TASK_LIST_FILE" ]; then
    echo "Error: Task list file '$TASK_LIST_FILE' not found!"
    exit 1
fi

MAX_JOBS=5  # <--- change this to control parallelism

echo "=========================================="
echo "Running batch demonstrations in parallel (max $MAX_JOBS jobs)"
echo "Task list file: $TASK_LIST_FILE"
echo "=========================================="

TOTAL_TASKS=$(grep -v '^#' "$TASK_LIST_FILE" | grep -v '^$' | wc -l)
CURRENT_TASK=0

run_task() {
    local idx="$1"
    local name="$2"

    echo ""
    echo "=========================================="
    echo "[$idx/$TOTAL_TASKS] Running task: $name"
    echo "=========================================="

    bash scripts/gen_demonstration_metaworld.sh "$name"
    local exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo "[$idx/$TOTAL_TASKS] Task '$name' completed successfully!"
    else
        echo "[$idx/$TOTAL_TASKS] Task '$name' failed with exit code $exit_code"
    fi
}

# Export function if you ever use GNU parallel/xargs, but not needed for pure bash loop:
# export -f run_task

while IFS= read -r task_name || [ -n "$task_name" ]; do
    # Skip empty lines and comments
    if [ -z "$task_name" ] || [[ "$task_name" == \#* ]]; then
        continue
    fi

    CURRENT_TASK=$((CURRENT_TASK + 1))

    # Start the task in the background
    run_task "$CURRENT_TASK" "$task_name" &

    # If we already have MAX_JOBS running, wait for at least one to finish
    while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do
        wait -n  # wait for any one job to complete (Bash 4.3+)
    done
done < "$TASK_LIST_FILE"

# Wait for all remaining jobs to finish
wait

echo ""
echo "=========================================="
echo "Batch demonstration generation complete!"
echo "Processed $CURRENT_TASK tasks"
echo "=========================================="
