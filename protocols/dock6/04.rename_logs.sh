#!/bin/bash

# This script safely renames all *.log files in the pose_logs directory
# by inserting '_dock6_b2' before the .log extension.
# It uses a shell loop to avoid the "Argument list too long" error.

for f in separated_logs/*.log; do
    # Check if the file exists (important if the wildcard expands to nothing)
    if [ -e "$f" ]; then
        # Extract the base filename without the directory part
        FILENAME=$(basename "$f")
        
        # Get the part before the extension (e.g., ZINC00002486_pose10)
        # We use a shell parameter expansion for speed: ${FILENAME%.log}
        BASE_NAME="${FILENAME%.log}"
        
        # Construct the new filename (e.g., ZINC00002486_pose10_ad4_a1.log)
        NEW_FILENAME="${BASE_NAME}_dock6_b2.log"
        
        # Execute the rename (mv) command, ensuring we keep it in the pose_logs directory
        mv "$f" "separated_logs/$NEW_FILENAME"
        
        # Optional: uncomment the line below to see which files are renamed
        # echo "Renamed $FILENAME to $NEW_FILENAME"
    fi
done

echo "Renaming complete in the separated_logs directory."
