#!/bin/bash
# Run AutoDock-GPU for all ligands in ligands_pdbqt/

LIG_DIR="ligands_prepared_hs"
MAP_FILE="dpf_gpf_files/REC.maps.fld"
OUT_DIR="results"

# create results directory if it doesn't exist
mkdir -p "$OUT_DIR"

for lig in "$LIG_DIR"/*.pdbqt; do
    base=$(basename "$lig" .pdbqt)
    echo "=== Running docking for $base ==="

    autodock_gpu_128wi \
        --lfile "$lig" \
        --ffile "$MAP_FILE" \
        --nrun 10 \
        --resnam "$OUT_DIR/$base" \
        --dlgoutput 1
done

echo "✅ All dockings finished. Results saved in $OUT_DIR/"
