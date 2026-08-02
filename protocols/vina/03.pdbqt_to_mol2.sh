#!/bin/bash

# Input and output directories
INPUT_DIR="./poses_pdbqt"
OUTPUT_DIR="./poses_mol2"

mkdir -p "$OUTPUT_DIR"

# Number of cores to use
CORES=10

# Convert function
convert_file() {
    local pdbqt_file="$1"
    local base=$(basename "$pdbqt_file" .pdbqt)
    obabel "$pdbqt_file" -O "$OUTPUT_DIR/${base}.mol2" >/dev/null 2>&1
    echo "Converted: $base.pdbqt -> $base.mol2"
}

export -f convert_file
export OUTPUT_DIR

# Use parallel to process files
find "$INPUT_DIR" -maxdepth 1 -name "*.pdbqt" | parallel -j $CORES convert_file {}
