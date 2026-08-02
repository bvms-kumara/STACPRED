#!/bin/bash

# This script safely renames all *.mol2 files in the poses_mol2 directory
# by inserting '_vina_b2' before the .mol2 extension.
# It uses a shell loop to avoid the "Argument list too long" error.

for f in poses_mol2/*.mol2; do
    # Check if the file exists (important if the wildcard expands to nothing)
    if [ -e "$f" ]; then
        # Extract the base filename without the directory part
        FILENAME=$(basename "$f")
        
        # Get the part before the extension (e.g., ZINC00002486_pose10)
        # We use a shell parameter expansion for speed: ${FILENAME%.mol2}
        BASE_NAME="${FILENAME%.mol2}"
        
        # Construct the new filename (e.g., ZINC00002486_pose10_ad4_a1.mol2)
        NEW_FILENAME="${BASE_NAME}_vina_b2.mol2"
        
        # Execute the rename (mv) command, ensuring we keep it in the poses_mol2 directory
        mv "$f" "poses_mol2/$NEW_FILENAME"
        
        # Optional: uncomment the line below to see which files are renamed
        # echo "Renamed $FILENAME to $NEW_FILENAME"
    fi
done

echo "Renaming complete in the poses_mol2 directory."
