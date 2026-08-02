#!/usr/bin/env python3
"""
sdf_to_pdbqt_by_id.py
Convert each molecule in merged_unique.sdf to a separate pdbqt file.
- Keeps ZINC ID or molecule name as the filename.
- Uses OpenBabel (obabel CLI) for the conversion.
- Does NOT add hydrogens.
"""

import os
import re
import sys
import tempfile
import subprocess
from rdkit import Chem

def sanitize(name):
    # keep only safe characters
    name = re.sub(r'[^A-Za-z0-9_.-]', '_', name)
    return name[:180]

def find_zinc_or_name(mol, idx):
    # try to find ZINC ID or use molecule name
    if mol.HasProp('_Name'):
        name = mol.GetProp('_Name').strip()
        if name:
            m = re.search(r'(ZINC\d+)', name, flags=re.IGNORECASE)
            if m:
                return m.group(1).upper()
            return name
    for prop in mol.GetPropNames():
        v = mol.GetProp(prop)
        if v:
            m = re.search(r'(ZINC\d+)', v, flags=re.IGNORECASE)
            if m:
                return m.group(1).upper()
    return f"mol_{idx+1:06d}"

def convert_with_obabel(tmp_sdf, out_pdbqt):
    # obabel command without adding hydrogens
    cmd = ['obabel', tmp_sdf, '-O', out_pdbqt]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        return False, "obabel not found. Install it with conda or apt."
    if p.returncode == 0 and os.path.exists(out_pdbqt):
        return True, "ok"
    return False, p.stderr.strip()

def main(infile, outdir):
    if not os.path.exists(infile):
        print("Input file not found:", infile)
        sys.exit(1)
    os.makedirs(outdir, exist_ok=True)
    suppl = Chem.SDMolSupplier(infile)
    if not suppl:
        print("Failed to open SDF file.")
        sys.exit(1)

    seen = set()
    converted = 0
    errors = []

    for i, mol in enumerate(suppl):
        if mol is None:
            continue
        name = find_zinc_or_name(mol, i)
        base = sanitize(name)
        fname = base
        suffix = 1
        while fname in seen:
            suffix += 1
            fname = f"{base}_{suffix}"
        seen.add(fname)
        out_pdbqt = os.path.join(outdir, fname + ".pdbqt")

        if os.path.exists(out_pdbqt):
            print(f"[skip] {fname} exists")
            continue

        tmp = tempfile.NamedTemporaryFile(suffix=".sdf", delete=False)
        tmpname = tmp.name
        tmp.close()
        try:
            w = Chem.SDWriter(tmpname)
            w.write(mol)
            w.close()
        except Exception as e:
            errors.append((fname, f"tmp SDF write failed: {e}"))
            continue

        ok, msg = convert_with_obabel(tmpname, out_pdbqt)
        os.unlink(tmpname)
        if ok:
            converted += 1
            if converted % 100 == 0:
                print(f"Converted {converted} molecules (last: {fname})")
        else:
            errors.append((fname, msg))

    print("✅ Done.")
    print("Converted:", converted)
    print("Total unique molecules:", len(seen))
    if errors:
        print("⚠️ Errors:", len(errors))
        for e in errors[:10]:
            print("  ", e[0], "->", e[1])

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 sdf_to_pdbqt_by_id.py merged_unique.sdf pdbqt_outputs")
        sys.exit(1)
    infile = sys.argv[1]
    outdir = sys.argv[2]
    main(infile, outdir)
