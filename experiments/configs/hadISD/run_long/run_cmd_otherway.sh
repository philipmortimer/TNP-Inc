#!/bin/bash
# Script to run a list of commands on gpu one after another for a long amount of time
COMMAND_FILE="/homes/pm846/Thesis/Code/TNP-Inc/experiments/configs/hadISD/run_long/commands_otherway.txt"

# Does command file actually exist
if [ ! -f "$COMMAND_FILE" ]; then
    echo "CMD file doesnt exist change path '$COMMAND_FILE'"
    exit 1
fi

# Reads each line skipping empty ones and eexcuting the command
while IFS= read -r cmd || [[ -n "$cmd" ]]; do
    if [[ -z "$cmd" ]]; then
        continue
    fi
    echo "----------"
    echo "Running: $cmd"

    output=$(eval "$cmd" 2>&1)
    exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo "Command finished: $cmd"
    else
        echo "Command failed check run log for: $cmd"
    fi

done < "$COMMAND_FILE"
echo "----------"
echo "Run all commands."