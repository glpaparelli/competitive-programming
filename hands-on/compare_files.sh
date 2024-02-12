#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <directory1> <directory2>"
    exit 1
fi

dir1="$1"
dir2="$2"

# Check if the directories exist
if [ ! -d "$dir1" ] || [ ! -d "$dir2" ]; then
    echo "Error: Both directories must exist."
    exit 1
fi

# Iterate through files in the first directory
for file1 in "$dir1"/*; do
    # Extract file name without path
    filename=$(basename "$file1")

    # Check if the corresponding file exists in the second directory
    file2="$dir2/$filename"
    if [ -e "$file2" ]; then
        # Compare the files using diff
        if diff "$file1" "$file2" &> /dev/null; then
            echo "Files $file1 and $file2 are equal."
        else
            echo "Files $file1 and $file2 are not equal."
        fi
    else
        echo "File $filename does not exist in $dir2."
    fi
done
