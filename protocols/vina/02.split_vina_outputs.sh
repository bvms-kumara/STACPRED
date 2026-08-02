#!/bin/bash
# ---------------------------------------------------------------
# Split Vina docking outputs (_out.pdbqt) into per-pose files
# and create corresponding .log files containing binding energies.
# ---------------------------------------------------------------

input_pdbqt_dir="./docked_outputs"
input_log_dir="./docking_logs"
out_pdbqt_dir="./poses_pdbqt"
out_log_dir="./pose_logs"

mkdir -p "$out_pdbqt_dir" "$out_log_dir"

# Loop through each Vina output PDBQT
for pdbqt in "$input_pdbqt_dir"/*.pdbqt; do
    base=$(basename "$pdbqt" .pdbqt)
    log_file="$input_log_dir/$base.log"
    echo "Processing $base ..."

    # --- Extract binding energies from the log file ---
    mapfile -t energies < <(awk '/^[[:space:]]*[0-9]+[[:space:]]*-/{print $2}' "$log_file")

    pose=0
    current_pose_file=""

    # --- Split the PDBQT into per-pose files ---
    while IFS= read -r line; do
        if [[ $line == MODEL* ]]; then
            pose=$(echo "$line" | awk '{print $2}')
            current_pose_file="$out_pdbqt_dir/${base}_pose${pose}.pdbqt"
            echo "$line" > "$current_pose_file"
            continue
        fi

        if [[ $line == ENDMDL* ]]; then
            echo "$line" >> "$current_pose_file"
            # Write matching energy to log file
            if [[ -n ${energies[$((pose-1))]} ]]; then
                echo "${energies[$((pose-1))]}" > "$out_log_dir/${base}_pose${pose}.log"
            else
                echo "N/A" > "$out_log_dir/${base}_pose${pose}.log"
            fi
            current_pose_file=""
            continue
        fi

        if [[ -n $current_pose_file ]]; then
            echo "$line" >> "$current_pose_file"
        fi
    done < "$pdbqt"

done

echo "✅ All poses extracted successfully!"
echo "Pose PDBQT files → $out_pdbqt_dir/"
echo "Pose energy logs → $out_log_dir/"
