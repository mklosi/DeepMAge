#!/bin/bash
set -e

# Script to kill all optuna processes.

process_name="python3.10.14-optuna"

echo "Killing all processes named '$process_name'..."
pkill -f "$process_name"
echo "Processes terminated."
