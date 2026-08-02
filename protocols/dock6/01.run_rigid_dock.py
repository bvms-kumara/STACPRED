#!/usr/bin/env python3
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

# CONFIG
dock_bin = "/home/kumara/malakafahs/dock/dock6/dock6/bin/dock6"   # adjust path if needed
template_in = "/home/kumara/malakafahs/dock/allosteric/dock6/doc6_2/005_rigid_dock/rigid.in"
ligand_dir = "/home/kumara/malakafahs/dock/allosteric/dock6/ligands_mol2"
output_root = "/home/kumara/malakafahs/dock/allosteric/dock6/all_outputs"
max_workers = 10   # number of parallel jobs

os.makedirs(output_root, exist_ok=True)

def run_dock(ligfile):
    ligbase = os.path.splitext(os.path.basename(ligfile))[0]
    outdir = os.path.join(output_root, ligbase)
    os.makedirs(outdir, exist_ok=True)

    # make a ligand-specific input file
    infile = os.path.join(outdir, f"{ligbase}_rigid.in")
    outfile = os.path.join(outdir, f"{ligbase}_rigid.out")

    with open(template_in) as f:
        lines = f.readlines()

    with open(infile, "w") as f:
        for line in lines:
            if line.strip().startswith("ligand_atom_file"):
                f.write(f"ligand_atom_file {os.path.abspath(ligfile)}\n")
            elif line.strip().startswith("ligand_outfile_prefix"):
                f.write(f"ligand_outfile_prefix {os.path.join(outdir, ligbase)}\n")
            else:
                f.write(line)

    # run dock
    cmd = [dock_bin, "-i", infile, "-o", outfile]
    try:
        subprocess.run(cmd, check=True)
        return f"[OK] {ligbase}"
    except subprocess.CalledProcessError:
        return f"[FAILED] {ligbase}"

def main():
    ligands = [os.path.join(ligand_dir, f) for f in os.listdir(ligand_dir) if f.endswith(".mol2")]
    total = len(ligands)
    print(f"Found {total} ligands. Running with {max_workers} workers...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_dock, lig): lig for lig in ligands}
        done_count = 0
        for future in as_completed(futures):
            lig = futures[future]
            result = future.result()
            done_count += 1
            print(f"[{done_count}/{total}] {result}")

if __name__ == "__main__":
    main()
