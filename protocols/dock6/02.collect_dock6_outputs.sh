#!/bin/bash
# --------------------------------------------------------------
# Collect all .mol2 and .out files from subdirectories under
# all_outputs/ into separate directories.
# --------------------------------------------------------------

base_dir="./all_outputs"
mol2_dir="./collected_mol2"
out_dir="./collected_out"

mkdir -p "$mol2_dir" "$out_dir"

echo "Collecting files from $base_dir ..."

# Loop through all subdirectories
find "$base_dir" -type f -name "*.mol2" -exec cp -v {} "$mol2_dir" \;
find "$base_dir" -type f -name "*.out" -exec cp -v {} "$out_dir" \;

echo "✅ Collection complete!"
echo "MOL2 files saved in: $mol2_dir/"
echo "OUT files saved in:  $out_dir/"
