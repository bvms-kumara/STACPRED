#!/bin/bash

# Directories
LIGAND_DIR="all_prepared_ligand"
OUTPUT_PDBQT_DIR="docked_outputs"
OUTPUT_LOG_DIR="docking_logs"
RECEPTOR="REC.pdbqt"

# Create output directories if they don't exist
mkdir -p "$OUTPUT_PDBQT_DIR"
mkdir -p "$OUTPUT_LOG_DIR"

# Grid box parameters
CENTER_X=65.623
CENTER_Y=29.982
CENTER_Z=43.044
SIZE_X=33.75
SIZE_Y=33.75
SIZE_Z=33.75
THREADS=2000
SEARCH_DEPTH=100

# Get total number of ligands for progress
TOTAL=$(ls "$LIGAND_DIR"/*.pdbqt | wc -l)
COUNT=0

# Loop through ligands
for LIGAND in "$LIGAND_DIR"/*.pdbqt; do
    COUNT=$((COUNT + 1))
    BASENAME=$(basename "$LIGAND")
    echo "Docking ligand $COUNT/$TOTAL: $BASENAME"

    ~/vina_gpu/Vina-GPU-2.1/AutoDock-Vina-GPU-2.1/AutoDock-Vina-GPU-2-1 \
      --receptor "$RECEPTOR" \
      --ligand "$LIGAND" \
      --center_x $CENTER_X \
      --center_y $CENTER_Y \
      --center_z $CENTER_Z \
      --size_x $SIZE_X \
      --size_y $SIZE_Y \
      --size_z $SIZE_Z \
      --out "$OUTPUT_PDBQT_DIR/${BASENAME%.pdbqt}_out.pdbqt" \
      --log "$OUTPUT_LOG_DIR/${BASENAME%.pdbqt}.log" \
      --thread $THREADS \
      --search_depth $SEARCH_DEPTH
done

echo "Batch docking complete!"
