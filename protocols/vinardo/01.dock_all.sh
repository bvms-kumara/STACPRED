#!/bin/bash

# Directories
LIG_DIR="all_prepared_ligand"
OUT_DIR="vina_outputs"
LOG_DIR="vina_logs"

# Create output directories if they don't exist
mkdir -p "$OUT_DIR"
mkdir -p "$LOG_DIR"

# Maximum number of simultaneous jobs
MAX_JOBS=8

# Counter for progress
TOTAL=$(ls $LIG_DIR/*.pdbqt | wc -l)
COUNT=0

# Function to run vina for one ligand
run_vina() {
    LIG=$1
    BASENAME=$(basename $LIG .pdbqt)
    vina --receptor REC.pdbqt \
         --ligand "$LIG" \
         --scoring vinardo \
         --center_x 65.623 --center_y 29.928 --center_z 43.044 \
         --size_x 90 --size_y 90 --size_z 90 \
         --spacing 0.375 \
         --out "$OUT_DIR/${BASENAME}_out.pdbqt" \
         --cpu 1 \
         --exhaustiveness 8 > "$LOG_DIR/${BASENAME}.log" 2>&1
}

export -f run_vina
export OUT_DIR LOG_DIR

# Loop over ligands and run with GNU parallel
ls $LIG_DIR/*.pdbqt | parallel -j $MAX_JOBS --bar run_vina {}
