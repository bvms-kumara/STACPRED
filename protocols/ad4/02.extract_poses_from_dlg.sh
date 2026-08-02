#!/bin/bash
# Extract individual poses and energies from AutoDock .dlg files
# Usage: ./extract_poses_from_dlg.sh

# --- directories ---
input_dir="./results"
out_pdbqt="./poses_pdbqt"
out_logs="./pose_logs"

mkdir -p "$out_pdbqt" "$out_logs"

# --- loop through all dlg files ---
for dlg in "$input_dir"/*.dlg; do
    base=$(basename "$dlg" .dlg)
    echo "Processing $base ..."
    
    # extract each MODEL separately
    awk -v base="$base" -v outp="$out_pdbqt" -v outl="$out_logs" '
        /^DOCKED: MODEL/ { 
            model=$3; 
            outfile=sprintf("%s/%s_pose%d.pdbqt", outp, base, model); 
            logfile=sprintf("%s/%s_pose%d.log", outl, base, model);
            in_model=1; 
            next;
        }
        /^DOCKED: ENDMDL/ && in_model {
            print "ENDMDL" >> outfile;
            in_model=0;
            next;
        }
        in_model && /^DOCKED:/ {
            # write only relevant docking content (ATOM, ROOT, etc.)
            if ($2 ~ /^(ROOT|ATOM|BRANCH|ENDBRANCH|TER|ENDROOT|TORSDOF)$/) {
                sub(/^DOCKED: /,"");
                print >> outfile;
            }
        }
        /Estimated Free Energy of Binding/ {
            # extract binding energy
            match($0, /= *(-?[0-9.]+)/, arr);
            if (arr[1] != "") {
                printf("%s\n", arr[1]) > logfile;
            }
        }
    ' "$dlg"
done

echo "✅ Extraction complete!"
echo "Pose structures saved in: $out_pdbqt/"
echo "Binding energy logs saved in: $out_logs/"
