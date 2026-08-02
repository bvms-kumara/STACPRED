# -*- coding: utf-8 -*-
import os
import subprocess

# ===== CONFIGURATION =====
mgltools_base = "/home/malaka-sandaruwan/newdirectory/mgltools_x86_64Linux2_1.5.7"
input_folder = "pdb_outputs"
output_folder = "prepared_ligands"

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# ===== FIND prepare_ligand4.py =====
prepare_script = os.path.join(
    mgltools_base, "MGLToolsPckgs", "AutoDockTools", "Utilities24", "prepare_ligand4.py"
)

if not os.path.isfile(prepare_script):
    print("❌ Could not find prepare_ligand4.py at: %s" % prepare_script)
    exit(1)
else:
    print("✅ Found prepare_ligand4.py at: %s" % prepare_script)

# ===== LIST ALL pdb FILES =====
ligand_files = [f for f in os.listdir(input_folder) if f.endswith(".pdb")]
if not ligand_files:
    print("❌ No pdb files found in %s" % input_folder)
    exit(1)

print("Found %d ligands to process..." % len(ligand_files))

# ===== PROCESS EACH LIGAND =====
pythonsh = os.path.join(mgltools_base, "bin", "pythonsh")

for ligand_file in ligand_files:
    input_path = os.path.abspath(os.path.join(input_folder, ligand_file))
    base_name = os.path.splitext(ligand_file)[0]
    output_path = os.path.abspath(os.path.join(output_folder, base_name + ".pdbqt"))
    print("Preparing ligand: %s -> %s" % (ligand_file, output_path))

    cmd = [pythonsh, prepare_script, "-l", input_path, "-o", output_path]
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        print("❌ Error processing %s: %s" % (ligand_file, e))
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
        exit(1)

print("\n✅ All ligands processed. Prepared ligands are in %s" % output_folder)

