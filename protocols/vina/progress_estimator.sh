#!/bin/bash

# --- User settings ---
input_dir="./all_prepared_ligand"
output_dir="./docked_outputs"
sleep_time=60   # seconds between checks

# --- Count total ligands ---
total=$(find "$input_dir" -type f | wc -l)
echo "Total ligands to process: $total"

# --- First check ---
prev_done=$(find "$output_dir" -type f | wc -l)
echo "Initial completed outputs: $prev_done"
echo "Monitoring progress every $sleep_time seconds..."
echo

while true; do
    sleep "$sleep_time"
    now_done=$(find "$output_dir" -type f | wc -l)
    new_done=$((now_done - prev_done))

    if [ "$new_done" -gt 0 ]; then
        rate=$(echo "scale=2; $new_done / ($sleep_time / 60)" | bc)
        remaining=$((total - now_done))
        est_minutes=$(echo "scale=2; $remaining / $rate" | bc)
        est_hours=$(echo "scale=2; $est_minutes / 60" | bc)

        echo "[$(date '+%H:%M:%S')] Completed: $now_done/$total"
        echo "  Rate: $rate ligands/min"
        echo "  Estimated time left: ~${est_hours} hr (${est_minutes} min)"
        echo
    else
        echo "[$(date '+%H:%M:%S')] No new outputs yet..."
    fi

    prev_done=$now_done
done
