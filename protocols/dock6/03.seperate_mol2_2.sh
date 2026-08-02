#!/bin/bash

# Input directory
INPUT_DIR="./collected_mol2"

# Output directories
POSE_DIR="./separated_mol2"
LOG_DIR="./separated_logs"

mkdir -p "$POSE_DIR"
mkdir -p "$LOG_DIR"

# Total files
total_files=$(ls "$INPUT_DIR"/*.mol2 | wc -l)
file_count=0

# Loop through all MOL2 files
for file in "$INPUT_DIR"/*.mol2; do
    file_count=$((file_count+1))
    base=$(basename "$file" .mol2)
    
    # Count number of poses in this file
    num_poses=$(grep -c "##########.*Grid_Score:" "$file")
    
    echo "Processing ($file_count/$total_files): $base with $num_poses poses..."
    
    # Read the file and split by each pose
    awk -v base="$base" -v pose_dir="$POSE_DIR" -v log_dir="$LOG_DIR" '
    BEGIN {
        pose_num=1
        in_pose=0
    }
    /^##########.*Grid_Score:/ {
        if (in_pose) { close(mol_file); close(log_file) }

        split($0,a,"Grid_Score:"); score=a[2]+0

        mol_file = pose_dir "/" base "_pose" pose_num ".mol2"
        log_file = log_dir "/" base "_pose" pose_num ".log"

        print score > log_file

        in_pose=1
        pose_num++
        next
    }
    {
        if (in_pose) print $0 >> mol_file
    }
    ' "$file"
done

echo "✅ Pose separation complete!"
echo "MOL2 poses saved in: $POSE_DIR"
echo "Pose energies saved in: $LOG_DIR"
