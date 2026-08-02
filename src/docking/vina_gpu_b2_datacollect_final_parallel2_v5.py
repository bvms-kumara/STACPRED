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
    from rdkit.DataStructs import TanimotoSimilarity
except ImportError:
    print("FATAL ERROR: RDKit is not installed. Please install RDKit to run this script.")
    sys.exit(1)


# ======================================================================
# --- SUPPRESS RDKIT MESSAGES (CRITICAL FIX) ---
RDLogger.DisableLog('rdApp.*')
# ======================================================================

# ======================================================================
# --- CONFIGURATION (EDIT THESE 5 LINES FOR EACH OF THE 8 RUNS) ---
# ======================================================================

# 1. Directory containing the docked pose MOL2 files.
MOL2_DIR = "/home/.........../poses_mol2"

# 2. Directory containing the matching log files with binding affinity.
LOG_DIR = "/home/.........../pose_logs"

# 3. Path to the LSalign executable (e.g., './LSalign').
LSALIGN_EXE = "./LSalign"

# 4. Path to the reference MOL2 file for RMSD/Tanimoto calculations.
REFERENCE_MOL2 = "reference.mol2"

# 5. The final name and path for the output CSV file.
OUTPUT_CSV = "b2_vina_gpu_results.csv" # TEST OUTPUT

# ======================================================================
# --- PARALLEL CONFIGURATION ---
# The number of CPU cores to use for multiprocessing.
NUM_CORES = 16 
# ======================================================================
# --- TEST MODE CONFIGURATION (TEMPORARY LIMIT) ---
# Set this to 100 to limit the run size. Set to 0 for full run.
MOLECULE_LIMIT = 100
# ======================================================================
# --- CONFIGURATION FOR FINGERPRINTS ---
FP_BITS = 2048  # Number of bits for the Morgan Fingerprint (ECFP)
FP_RADIUS = 2  # Radius for the Morgan Fingerprint (ECFP4)
# ======================================================================

# --- CONSTANT FOR FAILED RDKIT DATA ---
# Using None here allows Pandas to treat it as NaN for numbers, 
# but we will handle the strings explicitly to ensure no empty cells.
RDKIT_FAIL_FLAG = "RDKIT_FAIL"

# --- Fix numpy.float deprecation (Keeping for compatibility) ---
if not hasattr(np, "float"):
    np.float = float

# --- Descriptor sets (Kept as is) ---
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

# Define desired final column order (for header consistency)
FINAL_ORDER_BASE = [
    "Full_Name", "Canonical_SMILES", "Standard_InChIKey", "Docking_Algorithm", "Binding_Energy",
    "tan_morgan", "macs", "mcs_fraction", "rmsd", "com_distance",
    "MolWt", "MolLogP", "TPSA", "HBD", "HBA", "RotBonds", "AromaticRings", "FormalCharge",
    "HeavyAtomCount", "FractionCSP3", # Adding un-tupled 2D descriptors missed in original logic
    "Asphericity", "Eccentricity", "RadiusOfGyration", "SpherocityIndex",
    "InertialShapeFactor", "PMI1", "PMI2", "PMI3", "NPR1", "NPR2",
    "PC_scoreQ", "PC_scoreT", "Pval_PC8Q", "Pval_PC8T", "JaccardR", "QUERY_SIZE", "TEMPL_SIZE",
    "Processing_Error" 
]

# ----------------------------------------------------------------------
# --- Utility Functions (Mostly Unchanged) ---
# ----------------------------------------------------------------------

def repair_mol_for_sanitization(mol):
    """
    Attempts robust sanitization, falling back to skipping Kekulize
    and Valence checks if necessary to maximize data retention.
    """
    if mol is None:
        return None

    flexible_sanit_flags = (
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
        Chem.SanitizeMol(mol,
                         sanitizeOps=flexible_sanit_flags,
                         catchErrors=True)
        mol.UpdatePropertyCache(strict=False)
        return mol
    except Exception:
        return None

def calculate_robust_distance(mol_pose, mol_ref):
    """Calculates distance using a multi-layer fallback strategy (CoM > CoG > 1st Atom)."""
    if mol_pose is None or mol_ref is None or mol_pose.GetNumConformers() == 0 or mol_ref.GetNumConformers() == 0:
        return None

    # 1. Attempt Center of Mass (CoM) Distance
    try:
        conf1 = mol_pose.GetConformer(0)
        conf2 = mol_ref.GetConformer(0)
        com1 = AllChem.ComputeCenterOfMass(mol_pose, conf1)
        com2 = AllChem.ComputeCenterOfMass(mol_ref, conf2)
        return float(np.linalg.norm(np.array(com1) - np.array(com2)))
    except Exception:
        pass

    # 2. Attempt Center of Geometry (CoG) Distance
    try:
        pos1 = mol_pose.GetConformer(0).GetPositions()
        pos2 = mol_ref.GetConformer(0).GetPositions()

        heavy_pos1 = np.array([pos1[i] for i, atom in enumerate(mol_pose.GetAtoms()) if atom.GetAtomicNum() > 1])
        heavy_pos2 = np.array([pos2[i] for i, atom in enumerate(mol_ref.GetAtoms()) if atom.GetAtomicNum() > 1])

        if len(heavy_pos1) == 0 or len(heavy_pos2) == 0:
            raise Exception("No heavy atoms found.")

        cog1 = np.mean(heavy_pos1, axis=0)
        cog2 = np.mean(heavy_pos2, axis=0)

        return float(np.linalg.norm(cog1 - cog2))
    except Exception:
        pass

    # 3. Final Fallback: Distance between first heavy atoms
    try:
        heavy_atoms_pose = [i for i, atom in enumerate(mol_pose.GetAtoms()) if atom.GetAtomicNum() > 1]
        heavy_atoms_ref = [i for i, atom in enumerate(mol_ref.GetAtoms()) if atom.GetAtomicNum() > 1]

        if heavy_atoms_pose and heavy_atoms_ref:
            coord_pose = mol_pose.GetConformer(0).GetAtomPosition(heavy_atoms_pose[0])
            coord_ref = mol_ref.GetConformer(0).GetAtomPosition(heavy_atoms_ref[0])

            p1 = np.array([coord_pose.x, coord_pose.y, coord_pose.z])
            p2 = np.array([coord_ref.x, coord_ref.y, coord_ref.z])

            return float(np.linalg.norm(p1 - p2))
    except Exception:
        return None
    return None

def extract_docking_algorithm(file_name):
    """Extracts the docking algorithm name from the file name."""
    match = re.search(r'_(dock6|vina|vinardo|ad4)(?:_|\.)', file_name, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    return 'unknown'

def run_lsalign_and_parse(ligand_mol2, reference_mol2, lsalign_exe):
    """Runs LSalign and parses the output for various scores and RMSD."""
    rmsd_row = {k: None for k in ["PC_scoreQ", "PC_scoreT", "Pval_PC8Q", "Pval_PC8T",
                                  "JaccardR", "rmsd", "QUERY_SIZE", "TEMPL_SIZE"]}
    cmd = [lsalign_exe, ligand_mol2, reference_mol2, "-H", "1"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False)
        stdout_output = result.stdout
        data_line = None
        lines = stdout_output.splitlines()
        for i, line in enumerate(lines):
            if line.strip().startswith("QEURY_NAME") or line.strip().startswith("QUERY_NAME"):
                for j in range(i + 2, len(lines)):
                    potential_data = lines[j].strip()
                    if potential_data and potential_data[0].isalpha():
                        data_line = potential_data
                        break
                break
        if data_line:
            parts = data_line.split()
            if len(parts) >= 10:
                rmsd_row["PC_scoreQ"] = float(parts[2])
                rmsd_row["PC_scoreT"] = float(parts[3])
                rmsd_row["Pval_PC8Q"] = float(parts[4])
                rmsd_row["Pval_PC8T"] = float(parts[5])
                rmsd_row["JaccardR"] = float(parts[6])
                rmsd_row["rmsd"] = float(parts[7])
                rmsd_row["QUERY_SIZE"] = int(parts[8])
                rmsd_row["TEMPL_SIZE"] = int(parts[9])
    except Exception:
        pass
    return rmsd_row

def find_file_groups(mol2_dir, log_dir):
    """Matches MOL2 and log files by base name."""
    mol2_files = {re.sub(r'\.mol2$', '', f): f for f in os.listdir(mol2_dir) if f.endswith('.mol2')}
    log_files = {re.sub(r'\.(log|txt)$', '', f): f for f in os.listdir(log_dir) if f.endswith(('.log', '.txt'))}
    file_groups = {}
    for base_name, mol2_file in mol2_files.items():
        if base_name in log_files:
            log_file = log_files[base_name]
            file_groups[base_name] = {'mol2': os.path.join(mol2_dir, mol2_file), 'log': os.path.join(log_dir, log_file)}
    return file_groups

def generate_ecfp_from_mol(mol, fp_bits, fp_radius):
    """Generates Morgan Fingerprint (ECFP) as a Pandas Series."""
    if mol is None:
        fp_series = pd.Series([np.nan] * fp_bits, index=[f'ECFP_{i}' for i in range(fp_bits)])
        return fp_series
    try:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=fp_radius, nBits=fp_bits)
        fp_array = np.array(list(fp.ToBitString())).astype(int)
        fp_df = pd.Series(fp_array, index=[f'ECFP_{i}' for i in range(fp_bits)])
    except Exception:
        fp_df = pd.Series([np.nan] * fp_bits, index=[f'ECFP_{i}' for i in range(fp_bits)])
    return fp_df

def extract_binding_energy(log_path):
    """Extracts the binding energy (the first number) from the log file."""
    try:
        with open(log_path, 'r') as f:
            content = f.read().strip()
            # Finds the first floating point number in the file
            match = re.search(r'[-+]?\d*\.\d+|\d+', content)
            if match:
                return float(match.group(0))
            return None
    except FileNotFoundError:
        return None
    except Exception:
        return None

# ----------------------------------------------------------------------
# --- PARALLEL WORKER FUNCTION ---
# ----------------------------------------------------------------------

def worker_process_molecule(args):
    """
    Processes a single molecule/log file pair and returns the results dictionary.
    """
    base_name, files, REFERENCE_MOL2_PATH, LSALIGN_EXE, FP_BITS, FP_RADIUS = args
    
    processing_error = None
    
    # --- Data Aggregation (Base/Text Data) ---
    row = {}
    row["Full_Name"] = base_name
    row["Docking_Algorithm"] = extract_docking_algorithm(base_name)
    row["Binding_Energy"] = extract_binding_energy(files['log'])

    # LSalign is text-based and independent of RDKit's mol object
    rmsd_row = run_lsalign_and_parse(files['mol2'], REFERENCE_MOL2_PATH, LSALIGN_EXE)
    row.update(rmsd_row)
    
    # Reload reference molecule
    try:
        ref_mol = Chem.MolFromMol2File(REFERENCE_MOL2_PATH, removeHs=False, sanitize=False)
        if not ref_mol: return None 
        Chem.AssignStereochemistry(ref_mol, cleanIt=True, force=True)
        if ref_mol.GetNumConformers() == 0: AllChem.EmbedMolecule(ref_mol, AllChem.ETKDG())

    except Exception:
        return None # Critical fail: Reference could not be loaded
        
    # --- RDKit Molecule Processing ---
    mol_3d = None
    mol_fp = None
    
    # --- 3.1. Primary Load Attempt (Strict) ---
    try:
        mol_3d = Chem.MolFromMol2File(files['mol2'], removeHs=False, sanitize=True)
        if mol_3d:
            Chem.AssignStereochemistry(mol_3d, cleanIt=True, force=True)
            mol_fp = mol_3d
        else:
            raise Exception("Strict load failed, trying robust load.")
    except Exception:
        mol_3d = None

    # --- 3.2. Robust Fallback Load (Relaxed Sanitization) ---
    if mol_3d is None:
        try:
            mol_unsanitized = Chem.MolFromMol2File(files['mol2'], removeHs=False, sanitize=False)
            mol_repaired = repair_mol_for_sanitization(mol_unsanitized)
            if mol_repaired:
                mol_3d = mol_repaired
                mol_fp = mol_repaired
                processing_error = "MOL2 Load Success via Robust Repair"
            else:
                raise Exception("Robust repair failed.")
        except Exception:
            mol_3d = None
            processing_error = "MOL2 Load Failed: Both Strict and Robust"

    # --- 3.3. CRITICAL CONFORMER CHECK + 3D Fallback (Forcing 3D) ---
    smi_from_mol = None

    if mol_3d is not None and mol_3d.GetNumConformers() == 0:
        # Mol graph loaded, but 3D coordinates are missing/invalid.
        
        # Capture SMILES from the broken mol
        try:
            smi_from_mol = Chem.MolToSmiles(mol_3d, isomericSmiles=True)
            # Re-embed 3D
            cid = AllChem.EmbedMolecule(mol_3d, AllChem.ETKDGv3())
            if cid == -1:
                raise Exception("ETKDGv3 Conformer Generation failed.")
            
            processing_error = "Original Conformer Missing, 3D structure REGENERATED"

        except Exception as e:
            processing_error = f"2D Graph OK, 3D Embed Failed: {type(e).__name__}"
            mol_3d = None
            mol_fp = None 
            
    # --- 3.4. FINAL RESCUE: Load from SMILES if 2D graph failed but SMILES could be extracted (Less likely here) ---
    if mol_3d is None:
        # Last attempt to get SMILES using the least-strict load and force-embed
        temp_mol = Chem.MolFromMol2File(files['mol2'], removeHs=True, sanitize=False)
        if temp_mol:
            try:
                # Capture SMILES
                smi_from_mol = Chem.MolToSmiles(temp_mol, isomericSmiles=True)
                
                # Rebuild from SMILES (SMILES will sanitize the graph)
                rebuilt_mol = Chem.MolFromSmiles(smi_from_mol)
                if rebuilt_mol:
                    # Force embed 3D
                    rebuilt_mol = Chem.AddHs(rebuilt_mol)
                    cid = AllChem.EmbedMolecule(rebuilt_mol, AllChem.ETKDGv3())
                    if cid != -1:
                         mol_3d = rebuilt_mol
                         mol_fp = rebuilt_mol
                         processing_error = "MOL2 FAILED, REBUILT 3D from SMILES"
                    else:
                        raise Exception("SMILES rebuild failed 3D embed.")
                else:
                    raise Exception("SMILES load failed.")
            except Exception:
                # Still failed, no choice but to mark as RDKIT_FAIL
                pass
    
    # Update final error status
    row["Processing_Error"] = processing_error 
    row["Canonical_SMILES"] = smi_from_mol if smi_from_mol else RDKIT_FAIL_FLAG
    
    # --- Set Fail Values or Run Standard Calculations ---
    if mol_3d is None:
        
        # Set all RDKit dependent data to Fail Flags
        row["Standard_InChIKey"] = RDKIT_FAIL_FLAG
        row["com_distance"] = RDKIT_FAIL_FLAG
        row["tan_morgan"] = RDKIT_FAIL_FLAG
        row["macs"] = RDKIT_FAIL_FLAG
        row["mcs_fraction"] = RDKIT_FAIL_FLAG

        for d_name in core_2d:
            col_name = d_name[1] if isinstance(d_name, tuple) else d_name
            row[col_name] = RDKIT_FAIL_FLAG
        for d_name in ["HeavyAtomCount", "FractionCSP3"]:
            row[d_name] = RDKIT_FAIL_FLAG
        for d in core_3d:
            row[d] = RDKIT_FAIL_FLAG

        row.update(generate_ecfp_from_mol(None, FP_BITS, FP_RADIUS).to_dict())

        return row # Return the row with the fail flags
    
    # --- If mol_3d IS VALID, perform standard calculations ---
    
    # 3.2. ECFP Fingerprint Generation 
    fp_series = generate_ecfp_from_mol(mol_fp, FP_BITS, FP_RADIUS) 
    row.update(fp_series.to_dict())

    # com_distance calculation 
    row["com_distance"] = calculate_robust_distance(mol_3d, ref_mol)

    # Identifiers (SMILES was already set, now InChIKey)
    try:
        row["Standard_InChIKey"] = inchi.MolToInchiKey(mol_3d)
    except Exception:
        row["Standard_InChIKey"] = "InChIKey_FAIL"

    # 2D Descriptors 
    for d_name in core_2d:
        func_name = d_name
        col_name = d_name
        if isinstance(d_name, tuple):
            func_name, col_name = d_name
        try:
            if col_name == "FormalCharge":
                row[col_name] = GetFormalCharge(mol_3d)
            else:
                row[col_name] = getattr(Descriptors, func_name)(mol_3d)
        except Exception:
            row[col_name] = None
    
    # Remaining single-string 2D descriptors
    for d_name in ["HeavyAtomCount", "FractionCSP3"]:
        try:
            row[d_name] = getattr(Descriptors, d_name)(mol_3d)
        except Exception:
            row[d_name] = None


    # 3D Descriptors 
    for d in core_3d:
        try:
            row[d] = getattr(Descriptors3D, d)(mol_3d)
        except Exception:
            row[d] = None

    # Similarity 
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
        mcs_atoms = mcs_mol.GetNumHeavyAtoms()
        row["mcs_fraction"] = (2 * mcs_atoms) / (mol_3d.GetNumHeavyAtoms() + ref_mol.GetNumHeavyAtoms())
    except Exception:
        row["mcs_fraction"] = None
    
    return row

# ----------------------------------------------------------------------
# --- MAIN ORCHESTRATOR FUNCTION (Handles parallel setup and streaming) ---
# ----------------------------------------------------------------------

def process_data_and_save(mol2_dir, log_dir, lsalign_exe, reference_mol2, output_csv, limit):

    # --- 1. Validation and Setup ---
    if not os.path.exists(reference_mol2):
        print(f"FATAL ERROR: Reference molecule file not found at {reference_mol2}")
        sys.exit(1)
    
    print(f"✅ Reference molecule file confirmed: {reference_mol2}")

    file_groups_all = find_file_groups(mol2_dir, log_dir)
    if not file_groups_all:
        print(f"FATAL ERROR: No complete sets of (.mol2, .log) files found matching between directories.")
        sys.exit(1)

    file_groups = dict(list(file_groups_all.items())[:limit]) if limit > 0 else file_groups_all
    total_molecules = len(file_groups)
    print(f"\nFound {total_molecules} molecule sets to process.")
    if limit > 0:
        print(f"--- RUNNING IN TEST MODE: PROCESSING ONLY THE FIRST {limit} MOLECULES ---")


    # --- 2. Determine Final Column Order and Write Header ---
    # We must ensure the header is written exactly once with the correct order.
    ecfp_cols = [f'ECFP_{i}' for i in range(FP_BITS)]
    final_cols = FINAL_ORDER_BASE + ecfp_cols
    
    # Write the header to the CSV file, overwriting any existing file
    pd.DataFrame(columns=final_cols).to_csv(output_csv, index=False, header=True, mode='w')
    print(f"✅ Initialized output CSV: {output_csv} with {len(final_cols)} columns.")


    # --- 3. Prepare Parallel Execution Configuration ---
    tasks = []
    for base_name, files in file_groups.items():
        # Pass all necessary constants and file paths to the worker
        tasks.append((base_name, files, reference_mol2, lsalign_exe, FP_BITS, FP_RADIUS))
    
    print(f"Starting parallel processing on {NUM_CORES} cores...")

    processed_count = 0
    
    # --- 4. Process each file group in parallel and stream results ---
    try:
        with multiprocessing.Pool(processes=NUM_CORES) as pool:
            
            # Use imap_unordered to process tasks and get results as they finish
            results_iterator = pool.imap_unordered(worker_process_molecule, tasks)
            
            for result in tqdm(results_iterator, desc="Processing Molecules (Streaming)", unit="mol", total=total_molecules):
                
                if result is not None:
                    # Convert RDKIT_FAIL_FLAG to np.nan for numerical columns 
                    # if they are set to RDKIT_FAIL_FLAG (to avoid string in numerical column)
                    for key, value in result.items():
                        if value == RDKIT_FAIL_FLAG and key not in ["Canonical_SMILES", "Standard_InChIKey", "Processing_Error"]:
                            result[key] = np.nan
                    
                    df_row = pd.DataFrame([result], columns=final_cols)
                    
                    # STREAM: Append the row to the CSV. Header=False, Mode='a' (append)
                    df_row.to_csv(output_csv, index=False, header=False, mode='a')
                    processed_count += 1
                
    except Exception as e:
        print(f"\nFATAL PARALLEL ERROR: An error occurred during parallel execution: {e}")
        # The CSV will contain partial results up to the point of failure.


    # --- 5. Final Output Summary ---
    print(f"\n=======================================================")
    print(f"✅ FINAL SUCCESS! All streaming data saved to {output_csv}")
    print(f"Successfully processed and saved {processed_count} molecule(s) to the CSV.")


# --- Script Entry Point ---
if __name__ == "__main__":
    # Ensure RDKit's global objects/flags are set before forking processes
    try:
        # Check if LSalign is executable
        subprocess.run([LSALIGN_EXE, "-h"], capture_output=True, text=True, check=True)
    except FileNotFoundError:
        print(f"FATAL ERROR: LSalign executable not found at '{LSALIGN_EXE}'. Please check the path and permissions.")
        sys.exit(1)
    except subprocess.CalledProcessError:
        # LSalign often returns non-zero code for help, which is fine
        pass
    except Exception as e:
        print(f"FATAL ERROR checking LSalign: {e}")
        sys.exit(1)
        
    process_data_and_save(MOL2_DIR, LOG_DIR, LSALIGN_EXE, REFERENCE_MOL2, OUTPUT_CSV, MOLECULE_LIMIT)
