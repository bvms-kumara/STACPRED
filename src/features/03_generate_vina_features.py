#!/usr/bin/env python3
import os
import subprocess
import pandas as pd
import numpy as np
import re
import sys
import tempfile
import multiprocessing
from rdkit import RDLogger
from tqdm import tqdm

# RDKit Imports
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Descriptors3D, AllChem, MACCSkeys
    from rdkit.Chem.rdmolops import GetFormalCharge
    from rdkit.DataStructs import TanimotoSimilarity
    from rdkit.Chem import rdFMCS, inchi
except ImportError:
    print("FATAL ERROR: RDKit is not installed. Please install RDKit to run this script.")
    sys.exit(1)

# ======================================================================
# --- SUPPRESS RDKIT MESSAGES (CRITICAL FIX) ---
RDLogger.DisableLog('rdApp.*')
# ======================================================================

# ======================================================================
# --- CONFIGURATION (ADJUSTED FOR VINA GPU) ---
# ======================================================================
MOL2_DIR = "/home/kumara/malakafahs/dock/allosteric/vina_gpu_2/poses_mol2"
LOG_DIR = "/home/kumara/malakafahs/dock/allosteric/vina_gpu_2/pose_logs"
LSALIGN_EXE = "./LSalign"
REFERENCE_MOL2 = "4tq_reference.mol2"
OUTPUT_CSV = "b2_vina_final_results_2.csv"
NUM_CORES = 16
MOLECULE_LIMIT = 0
FP_BITS = 2048
FP_RADIUS = 2

if not hasattr(np, "float"):
    np.float = float

core_2d = [
    "MolWt", "MolLogP", "TPSA",
    ("NumHDonors", "HBD"), ("NumHAcceptors", "HBA"),
    ("NumRotatableBonds", "RotBonds"),
    "HeavyAtomCount", "FractionCSP3",
    ("NumAromaticRings", "AromaticRings"),
    "FormalCharge"
]

core_3d = [
    "Asphericity", "Eccentricity", "RadiusOfGyration", "SpherocityIndex",
    "InertialShapeFactor", "PMI1", "PMI2", "PMI3", "NPR1", "NPR2"
]

FINAL_ORDER_BASE = [
    "Full_Name", "Canonical_SMILES", "Standard_InChIKey", "Docking_Algorithm", "Binding_Energy",
    "tan_morgan", "macs", "mcs_fraction", "rmsd", "com_distance",
    "MolWt", "MolLogP", "TPSA", "HBD", "HBA", "RotBonds", "AromaticRings", "FormalCharge",
    "HeavyAtomCount", "FractionCSP3",
    "Asphericity", "Eccentricity", "RadiusOfGyration", "SpherocityIndex",
    "InertialShapeFactor", "PMI1", "PMI2", "PMI3", "NPR1", "NPR2",
    "PC_scoreQ", "PC_scoreT", "Pval_PC8Q", "Pval_PC8T", "JaccardR", "QUERY_SIZE", "TEMPL_SIZE",
    "LSalign_Error" 
]

# ----------------------------------------------------------------------
# --- Utility Functions ---
# ----------------------------------------------------------------------

def repair_mol_for_sanitization(mol):
    if mol is None:
        return None
    flexible_flags = (
        Chem.SanitizeFlags.SANITIZE_ALL &
        ~Chem.SanitizeFlags.SANITIZE_KEKULIZE &
        ~Chem.SanitizeFlags.SANITIZE_SETVALENCES
    )
    try:
        Chem.SanitizeMol(mol, catchErrors=True)
        return mol
    except Exception:
        pass
    try:
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        Chem.SanitizeMol(mol, sanitizeOps=flexible_flags, catchErrors=True)
        mol.UpdatePropertyCache(strict=False)
        return mol
    except Exception:
        return None


def calculate_robust_distance(mol_pose, mol_ref):
    """Calculates CoM/CoG distance using RDKit's conformer data."""
    if mol_pose is None or mol_ref is None or mol_pose.GetNumConformers() == 0 or mol_ref.GetNumConformers() == 0:
        return None
    try:
        conf1 = mol_pose.GetConformer(0)
        conf2 = mol_ref.GetConformer(0)
        com1 = AllChem.ComputeCenterOfMass(mol_pose, conf1)
        com2 = AllChem.ComputeCenterOfMass(mol_ref, conf2)
        return float(np.linalg.norm(np.array(com1) - np.array(com2)))
    except Exception:
        pass
    try:
        pos1 = mol_pose.GetConformer(0).GetPositions()
        pos2 = mol_ref.GetConformer(0).GetPositions()
        cog1 = np.mean(pos1, axis=0)
        cog2 = np.mean(pos2, axis=0)
        return float(np.linalg.norm(cog1 - cog2))
    except Exception:
        pass
    return None


def calculate_com_distance_from_mol2(mol2_file, ref_mol):
    """Backup method: compute COM distance directly from raw mol2 file atoms."""
    try:
        with open(mol2_file, 'r') as f:
            lines = f.readlines()
        atom_coords = []
        in_atom_section = False
        for line in lines:
            if line.startswith("@<TRIPOS>ATOM"):
                in_atom_section = True
                continue
            if line.startswith("@<TRIPOS>BOND"):
                break
            if in_atom_section:
                parts = line.split()
                if len(parts) >= 6:
                    x, y, z = map(float, parts[2:5])
                    atom_coords.append([x, y, z])
        if not atom_coords:
            return None
        
        ligand_center = np.mean(np.array(atom_coords), axis=0)
        ref_conf = ref_mol.GetConformer(0)
        ref_positions = np.array(ref_conf.GetPositions())
        ref_center = np.mean(ref_positions, axis=0)
        
        return float(np.linalg.norm(ligand_center - ref_center))
    except Exception:
        return None


def extract_docking_algorithm(file_name):
    match = re.search(r'_(dock6|vina|vinardo|ad4)(?:_|\.)', file_name, re.IGNORECASE)
    return match.group(1).lower() if match else 'unknown'


def run_lsalign_and_parse(ligand_mol2, reference_mol2, lsalign_exe):
    # Initialize with None and an error message holder
    rmsd_row = {k: None for k in ["PC_scoreQ", "PC_scoreT", "Pval_PC8Q", "Pval_PC8T",
                                 "JaccardR", "rmsd", "QUERY_SIZE", "TEMPL_SIZE", "LSalign_Error"]}
    
    cmd = [lsalign_exe, ligand_mol2, reference_mol2, "-H", "1"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False)
        stdout_output = result.stdout
        
        if result.returncode != 0 and "QUERY_NAME" not in stdout_output and "QEURY_NAME" not in stdout_output:
            rmsd_row["LSalign_Error"] = f"LSalign exited with code {result.returncode}. Stderr: {result.stderr.strip()}"
            return rmsd_row
            
        # --- CRITICAL FIX: Robust Regex to handle varying Filename/Path formats (e.g., 'p/ZINC...' or '999146...') ---
        # The pattern captures the two ID columns (allowing for path characters like /) and the 8 numeric columns.
        
        regex_pattern = re.compile(
            r'^\s*([a-zA-Z0-9_/.\-]+)\s+([\*a-zA-Z0-9_/.\-]+)\s+([-]?\d+\.\d+)\s+([-]?\d+\.\d+)\s+([-]?\d+\.\d+)\s+([-]?\d+\.\d+)\s+([-]?\d+\.\d+)\s+([-]?\d+\.\d+)\s+(\d+)\s+(\d+)\s*$', 
            re.MULTILINE
        )
        
        match = regex_pattern.search(stdout_output)

        if match:
            # The groups are: (ID1, ID2, PCQ, PCT, PvalQ, PvalT, JaccardR, RMSD, QSize, TSize)
            data_parts = match.groups()
            
            # The data we want starts at index 2 (PC-score8Q)
            if len(data_parts) >= 10:
                try:
                    rmsd_row["PC_scoreQ"] = float(data_parts[2])
                    rmsd_row["PC_scoreT"] = float(data_parts[3])
                    rmsd_row["Pval_PC8Q"] = float(data_parts[4])
                    rmsd_row["Pval_PC8T"] = float(data_parts[5])
                    rmsd_row["JaccardR"] = float(data_parts[6])
                    rmsd_row["rmsd"] = float(data_parts[7])
                    rmsd_row["QUERY_SIZE"] = int(data_parts[8])
                    rmsd_row["TEMPL_SIZE"] = int(data_parts[9])
                    rmsd_row["LSalign_Error"] = None # Success
                except ValueError as e:
                    rmsd_row["LSalign_Error"] = f"Error converting LSalign output to float/int. Captured data: {data_parts}. Error: {e}"
            else:
                rmsd_row["LSalign_Error"] = f"Regex captured insufficient columns ({len(data_parts)})."
        else:
            rmsd_row["LSalign_Error"] = "Could not find LSalign summary data line using robust regex search."
            
    except subprocess.TimeoutExpired:
        rmsd_row["LSalign_Error"] = "LSalign command timed out (60 seconds)."
    except Exception as e:
        rmsd_row["LSalign_Error"] = f"Unexpected error during LSalign execution: {e}"
        
    return rmsd_row


def find_file_groups(mol2_dir, log_dir):
    mol2_files = {re.sub(r'\.mol2$', '', f): f for f in os.listdir(mol2_dir) if f.endswith('.mol2')}
    log_files = {re.sub(r'\.(log|txt)$', '', f): f for f in os.listdir(log_dir) if f.endswith(('.log', '.txt'))}
    return {base: {'mol2': os.path.join(mol2_dir, mol2_files[base]),
                   'log': os.path.join(log_dir, log_files[base])}
            for base in mol2_files if base in log_files}


def generate_ecfp_from_mol(mol, fp_bits, fp_radius):
    if mol is None:
        return pd.Series([0] * fp_bits, index=[f'ECFP_{i}' for i in range(fp_bits)])
    try:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=fp_radius, nBits=fp_bits)
        return pd.Series(list(map(int, list(fp.ToBitString()))), index=[f'ECFP_{i}' for i in range(fp_bits)])
    except Exception:
        return pd.Series([0] * fp_bits, index=[f'ECFP_{i}' for i in range(fp_bits)])


def extract_binding_energy(log_path):
    try:
        with open(log_path, 'r') as f:
            content = f.read().strip()
        match = re.search(r'[-+]?\d*\.\d+|\d+', content)
        return float(match.group(0)) if match else None
    except Exception:
        return None

# ----------------------------------------------------------------------
# --- Worker Function ---
# ----------------------------------------------------------------------

def worker_process_molecule(args):
    base_name, files, REFERENCE_MOL2_PATH, LSALIGN_EXE, FP_BITS, FP_RADIUS = args

    # Load reference molecule
    ref_mol = None
    try:
        ref_mol = Chem.MolFromMol2File(REFERENCE_MOL2_PATH, removeHs=False, sanitize=False)
        if not ref_mol:
            return None
        Chem.AssignStereochemistry(ref_mol, cleanIt=True, force=True)
        if ref_mol.GetNumConformers() == 0:
            AllChem.EmbedMolecule(ref_mol, AllChem.ETKDG())
    except Exception:
        return None

    ligand_mol2 = files['mol2']
    log_file = files['log']
    mol_3d = None
    rdkit_success = False

    # Load and sanitize ligand molecule
    try:
        mol_3d = Chem.MolFromMol2File(ligand_mol2, removeHs=False, sanitize=True)
        if not mol_3d:
            mol_unsanitized = Chem.MolFromMol2File(ligand_mol2, removeHs=False, sanitize=False)
            mol_3d = repair_mol_for_sanitization(mol_unsanitized)
        if mol_3d and mol_3d.GetNumConformers() > 0:
            rdkit_success = True
    except Exception:
        mol_3d = None

    # Compute COM distance (RDKit method + fallback to raw mol2)
    com_distance = calculate_robust_distance(mol_3d, ref_mol)
    
    com_fallback_used = False
    if com_distance is None:
        com_distance = calculate_com_distance_from_mol2(ligand_mol2, ref_mol)
        if com_distance is not None:
             com_fallback_used = True

    # Prepare result row
    row = {"Full_Name": base_name, "Docking_Algorithm": extract_docking_algorithm(base_name),
           "Binding_Energy": extract_binding_energy(log_file), "com_distance": com_distance}
    
    # Initialize RDKit error message
    lsalign_error_message = None
    if com_fallback_used:
        lsalign_error_message = "RDKit MolFromMol2File failed, used raw mol2 fallback for CoM distance."
    elif not rdkit_success:
         lsalign_error_message = "RDKit MolFromMol2File failed, descriptors/similarity skipped."

    # Process RDKit-dependent properties only if mol_3d load/sanitize was successful
    if rdkit_success:
        try:
            row["Canonical_SMILES"] = Chem.MolToSmiles(mol_3d, isomericSmiles=True)
            row["Standard_InChIKey"] = inchi.MolToInchiKey(mol_3d)
        except Exception:
            row["Canonical_SMILES"], row["Standard_InChIKey"] = None, None

        for d_name in core_2d:
            func_name, col_name = (d_name if isinstance(d_name, tuple) else (d_name, d_name))
            try:
                row[col_name] = GetFormalCharge(mol_3d) if col_name == "FormalCharge" else getattr(Descriptors, func_name)(mol_3d)
            except Exception:
                row[col_name] = None
        
        for d in core_3d:
            try:
                row[d] = getattr(Descriptors3D, d)(mol_3d)
            except Exception:
                row[d] = None

        try:
            fp_macs = MACCSkeys.GenMACCSKeys(mol_3d)
            row["macs"] = fp_macs.GetNumOnBits()
        except Exception:
            row["macs"] = None

        try:
            fp_mol = AllChem.GetMorganFingerprint(mol_3d, 2)
            fp_ref = AllChem.GetMorganFingerprint(ref_mol, 2)
            row["tan_morgan"] = TanimotoSimilarity(fp_mol, fp_ref)
        except Exception:
            row["tan_morgan"] = None

        try:
            mcs_result = rdFMCS.FindMCS([mol_3d, ref_mol])
            mcs_mol = Chem.MolFromSmarts(mcs_result.smartsString)
            row["mcs_fraction"] = (2 * mcs_mol.GetNumHeavyAtoms()) / (mol_3d.GetNumHeavyAtoms() + ref_mol.GetNumHeavyAtoms())
        except Exception:
            row["mcs_fraction"] = None
    else:
         # Populate RDKit-dependent fields with None if RDKit failed
        for col in ["Canonical_SMILES", "Standard_InChIKey", "macs", "tan_morgan", "mcs_fraction"] + [c[1] if isinstance(c, tuple) else c for c in core_2d] + core_3d:
             if col not in row:
                 row[col] = None
                 
    # Run LSalign and update the row
    lsalign_results = run_lsalign_and_parse(ligand_mol2, REFERENCE_MOL2_PATH, LSALIGN_EXE)
    
    # Update the row with LSalign results (including the 'rmsd' column)
    row.update(lsalign_results)
    
    # Finalize the error message, prioritizing LSalign's error over RDKit's
    if row["LSalign_Error"] is None:
        row["LSalign_Error"] = lsalign_error_message

    fp_series = generate_ecfp_from_mol(mol_3d, FP_BITS, FP_RADIUS)
    row.update(fp_series.to_dict())

    return row

# ----------------------------------------------------------------------
# --- Main Function ---
# ----------------------------------------------------------------------

def process_data_and_save(mol2_dir, log_dir, lsalign_exe, reference_mol2, output_csv, limit):
    if not os.path.exists(reference_mol2):
        print(f"FATAL: Reference molecule not found: {reference_mol2}")
        sys.exit(1)

    file_groups_all = find_file_groups(mol2_dir, log_dir)
    file_groups = dict(list(file_groups_all.items())[:limit]) if limit > 0 else file_groups_all
    total_molecules = len(file_groups)
    print(f"Found {total_molecules} molecules.")

    ecfp_cols = [f'ECFP_{i}' for i in range(FP_BITS)]
    final_cols = FINAL_ORDER_BASE + ecfp_cols
    
    # Write the header
    pd.DataFrame(columns=final_cols).to_csv(output_csv, index=False, header=True, mode='w')

    tasks = [(bn, f, reference_mol2, lsalign_exe, FP_BITS, FP_RADIUS) for bn, f in file_groups.items()]
    processed = 0
    with multiprocessing.Pool(processes=NUM_CORES) as pool:
        for result in tqdm(pool.imap_unordered(worker_process_molecule, tasks), total=total_molecules, desc="Processing"):
            if result:
                # Append data row, ensuring correct column order
                pd.DataFrame([result], columns=final_cols).to_csv(output_csv, index=False, header=False, mode='a')
                processed += 1
    print(f"✅ Finished. {processed} molecules processed.")


if __name__ == "__main__":
    try:
        subprocess.run([LSALIGN_EXE, "-h"], capture_output=True, check=True)
    except FileNotFoundError:
        print(f"FATAL ERROR: LSalign not found at {LSALIGN_EXE}")
        sys.exit(1)
    except subprocess.CalledProcessError:
         pass
    except Exception as e:
        print(f"FATAL ERROR checking LSalign: {e}")
        sys.exit(1)
        
    process_data_and_save(MOL2_DIR, LOG_DIR, LSALIGN_EXE, REFERENCE_MOL2, OUTPUT_CSV, MOLECULE_LIMIT)
