#!/bin/bash

BASE_DIR="./dataset/" 
TARGET_FILE="create_dataset.py"

for dir in "$BASE_DIR"*/ ; do
    if [ -d "$dir" ] && [ -f "$dir$TARGET_FILE" ]; then
        echo "$dir$TARGET_FILE"
        python "$dir$TARGET_FILE"
    fi
done