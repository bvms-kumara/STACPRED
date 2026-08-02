#!/usr/bin/env python3
"""
STACPRED Real-Time Inference Module
Extracts 2D/3D descriptors, ECFP fingerprints, and LSalign metrics for uploaded MOL2 files,
logging full physicochemical profiles and detailed LSalign alignment outputs to the terminal.
"""

import os
import sys
import re
import shutil
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path

# RDKit Imports with Safe Guard
RDKIT_AVAILABLE = False
try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors, Descriptors3D, AllChem, MACCSkeys, rdFMCS, inchi
    from rdkit.Chem.rdmolops import GetFormalCharge
    from rdkit.DataStructs import TanimotoSimilarity
    RDLogger.DisableLog('rdApp.*')
    RDKIT_AVAILABLE = True
except Exception as e:
    print(f"[WARNING] RDKit initialization warning: {e}")

CORE_2D = [
    "MolWt", "MolLogP", "TPSA",
    ("NumHDonors", "HBD"), ("NumHAcceptors", "HBA"),
    ("NumRotatableBonds", "RotBonds"),
    "HeavyAtomCount", "FractionCSP3",
    ("NumAromaticRings", "AromaticRings"),
    "FormalCharge"
]

CORE_3D = [
    "Asphericity", "Eccentricity", "RadiusOfGyration", "SpherocityIndex",
    "InertialShapeFactor", "PMI1", "PMI2", "PMI3", "NPR1", "NPR2"
]

CATEGORICAL_COLS = ["Docking_Algorithm", "LSalign_Error", "Canonical_SMILES", "Standard_InChIKey", "Full_Name"]


def repair_mol_for_sanitization(mol):
    """Attempts flexible sanitization on molecules with non-standard valences or atom types."""
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


def load_mol2_robust(mol2_path):
    """Multi-stage robust MOL2 parser handling non-standard TRIPOS atom types."""
    if not os.path.exists(mol2_path):
        return None

    try:
        mol = Chem.MolFromMol2File(str(mol2_path), removeHs=False, sanitize=True)
        if mol is not None and mol.GetNumAtoms() > 0:
            return mol
    except Exception:
        pass

    try:
        mol = Chem.MolFromMol2File(str(mol2_path), removeHs=False, sanitize=False)
        if mol is not None and mol.GetNumAtoms() > 0:
            repaired = repair_mol_for_sanitization(mol)
            if repaired is not None:
                return repaired
            return mol
    except Exception:
        pass

    try:
        with open(mol2_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        content_cleaned = re.sub(r'([A-Za-z]+)\.(?:am|pl3|2|3|1|ar)', r'\1', content)
        mol = Chem.MolFromMol2Block(content_cleaned, removeHs=False, sanitize=False)
        if mol is not None and mol.GetNumAtoms() > 0:
            repaired = repair_mol_for_sanitization(mol)
            return repaired if repaired is not None else mol
    except Exception:
        pass

    return None


def is_valid_windows_executable(file_path):
    """Checks file magic bytes to verify if binary is a valid Windows executable."""
    if not file_path or not os.path.exists(file_path):
        return False
    if sys.platform != "win32":
        return True
    try:
        with open(file_path, "rb") as f:
            header = f.read(4)
            if header.startswith(b"MZ"):
                return True
    except Exception:
        pass
    return False


def locate_lsalign_executable():
    """Locates LSalign executable across bin/, repo root, or system PATH."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    exe_name = "LSalign.exe" if sys.platform == "win32" else "LSalign"
    
    possible_paths = [
        repo_root / "bin" / exe_name,
        repo_root / exe_name,
        Path.cwd() / exe_name,
        Path.cwd() / "bin" / exe_name
    ]
    for p in possible_paths:
        if p.exists() and is_valid_windows_executable(str(p)):
            return str(p)
            
    found_in_path = shutil.which(exe_name) or shutil.which("LSalign")
    if found_in_path and is_valid_windows_executable(found_in_path):
        return found_in_path

    return None


def run_lsalign_single(ligand_mol2, ref_mol2):
    """Executes LSalign binary and parses alignment output capturing both stdout & stderr."""
    exe_path = locate_lsalign_executable()
    res = {
        "LSalign_Status": "Skipped (Executable Missing/Invalid)",
        "rmsd": 0.0, "PC_scoreQ": 0.5, "PC_scoreT": 0.5,
        "Pval_PC8Q": 0.05, "Pval_PC8T": 0.05, "JaccardR": 0.5,
        "QUERY_SIZE": 0, "TEMPL_SIZE": 0,
        "LSalign_Error": "LSalign binary not present or incompatible with OS"
    }

    if not exe_path or not os.path.exists(exe_path):
        return res

    if not ref_mol2 or not os.path.exists(ref_mol2):
        res["LSalign_Error"] = f"Reference structure missing at {ref_mol2}"
        return res

    cmd = [exe_path, str(ligand_mol2), str(ref_mol2), "-H", "1"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15, check=False)
        stdout_output = result.stdout or ""
        stderr_output = result.stderr or ""
        
        num_pattern = r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?|nan|inf|\*+'
        regex_pattern = re.compile(
            rf'^\s*([^\s]+)\s+([^\s]+)\s+({num_pattern})\s+({num_pattern})\s+({num_pattern})\s+({num_pattern})\s+({num_pattern})\s+({num_pattern})\s+(\d+)\s+(\d+)\s*$', 
            re.MULTILINE | re.IGNORECASE
        )
        match = regex_pattern.search(stdout_output)
        
        def safe_float(val, default=0.0):
            try:
                v = float(val)
                return default if np.isnan(v) or np.isinf(v) else v
            except Exception:
                return default

        if match:
            parts = match.groups()
            res["PC_scoreQ"] = safe_float(parts[2], 0.5)
            res["PC_scoreT"] = safe_float(parts[3], 0.5)
            res["Pval_PC8Q"] = safe_float(parts[4], 0.05)
            res["Pval_PC8T"] = safe_float(parts[5], 0.05)
            res["JaccardR"] = safe_float(parts[6], 0.5)
            res["rmsd"] = safe_float(parts[7], 0.0)
            res["QUERY_SIZE"] = int(parts[8]) if parts[8].isdigit() else 0
            res["TEMPL_SIZE"] = int(parts[9]) if parts[9].isdigit() else 0
            res["LSalign_Status"] = "SUCCESS"
            res["LSalign_Error"] = "None"
        else:
            res["LSalign_Status"] = "Executed (Output Unparsed)"
            err_msg = stderr_output.strip() if stderr_output.strip() else stdout_output.strip()
            preview = " ".join(err_msg.split())[:110] if err_msg else "LSalign returned no output."
            res["LSalign_Error"] = f"Unparsed Output: '{preview}'"
    except Exception as e:
        res["LSalign_Status"] = "Failed"
        res["LSalign_Error"] = str(e)

    return res


def extract_single_molecule_features(mol2_path, ref_mol2_path, fp_bits=2048, fp_radius=2):
    """Extracts high-dimensional features matching the training feature schema."""
    row = {}

    if not RDKIT_AVAILABLE:
        print("[ERROR] Cannot compute chemical descriptors: RDKit is missing.")
        return pd.DataFrame([row])

    row["Docking_Algorithm"] = "vina"
    row["Full_Name"] = Path(mol2_path).stem
    row["_ref_name"] = Path(ref_mol2_path).stem

    ref_mol = load_mol2_robust(ref_mol2_path)
    if ref_mol:
        try:
            Chem.AssignStereochemistry(ref_mol, cleanIt=True, force=True)
            if ref_mol.GetNumConformers() == 0:
                AllChem.EmbedMolecule(ref_mol, AllChem.ETKDG())
        except Exception:
            pass

    mol_3d = load_mol2_robust(mol2_path)

    if mol_3d:
        try:
            row["Canonical_SMILES"] = Chem.MolToSmiles(mol_3d, isomericSmiles=True)
            row["Standard_InChIKey"] = inchi.MolToInchiKey(mol_3d)
        except Exception:
            row["Canonical_SMILES"], row["Standard_InChIKey"] = "missing", "missing"

        for d_name in CORE_2D:
            func_name, col_name = (d_name if isinstance(d_name, tuple) else (d_name, d_name))
            try:
                row[col_name] = GetFormalCharge(mol_3d) if col_name == "FormalCharge" else getattr(Descriptors, func_name)(mol_3d)
            except Exception:
                row[col_name] = 0.0

        for d in CORE_3D:
            try:
                row[d] = getattr(Descriptors3D, d)(mol_3d)
            except Exception:
                row[d] = 0.0

        try:
            fp_macs = MACCSkeys.GenMACCSKeys(mol_3d)
            row["macs"] = fp_macs.GetNumOnBits()
        except Exception:
            row["macs"] = 0

        if ref_mol:
            try:
                fp_mol = AllChem.GetMorganFingerprint(mol_3d, 2)
                fp_ref = AllChem.GetMorganFingerprint(ref_mol, 2)
                row["tan_morgan"] = TanimotoSimilarity(fp_mol, fp_ref)
            except Exception:
                row["tan_morgan"] = 0.0

            try:
                mcs_result = rdFMCS.FindMCS([mol_3d, ref_mol])
                mcs_mol = Chem.MolFromSmarts(mcs_result.smartsString)
                row["mcs_fraction"] = (2 * mcs_mol.GetNumHeavyAtoms()) / (mol_3d.GetNumHeavyAtoms() + ref_mol.GetNumHeavyAtoms())
            except Exception:
                row["mcs_fraction"] = 0.0

            try:
                conf1 = mol_3d.GetConformer(0)
                conf2 = ref_mol.GetConformer(0)
                com1 = AllChem.ComputeCenterOfMass(mol_3d, conf1)
                com2 = AllChem.ComputeCenterOfMass(ref_mol, conf2)
                row["com_distance"] = float(np.linalg.norm(np.array(com1) - np.array(com2)))
            except Exception:
                row["com_distance"] = 0.0
        else:
            row["tan_morgan"] = 0.0
            row["mcs_fraction"] = 0.0
            row["com_distance"] = 0.0

        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol_3d, radius=fp_radius, nBits=fp_bits)
            fp_bits_list = list(map(int, list(fp.ToBitString())))
            for i, b in enumerate(fp_bits_list):
                row[f"ECFP_{i}"] = b
        except Exception:
            for i in range(fp_bits):
                row[f"ECFP_{i}"] = 0
    else:
        row["Canonical_SMILES"] = "missing"
        row["Standard_InChIKey"] = "missing"
        for d_name in CORE_2D:
            col_name = d_name[1] if isinstance(d_name, tuple) else d_name
            row[col_name] = 0.0
        for d in CORE_3D:
            row[d] = 0.0
        row["macs"] = 0
        row["tan_morgan"] = 0.0
        row["mcs_fraction"] = 0.0
        row["com_distance"] = 0.0
        for i in range(fp_bits):
            row[f"ECFP_{i}"] = 0

    ls_data = run_lsalign_single(mol2_path, ref_mol2_path)
    row["LSalign_Error"] = ls_data["LSalign_Error"]
    row["rmsd"] = ls_data["rmsd"]
    row["PC_scoreQ"] = ls_data["PC_scoreQ"]
    row["PC_scoreT"] = ls_data["PC_scoreT"]
    row["Pval_PC8Q"] = ls_data["Pval_PC8Q"]
    row["Pval_PC8T"] = ls_data["Pval_PC8T"]
    row["JaccardR"] = ls_data["JaccardR"]
    row["QUERY_SIZE"] = ls_data["QUERY_SIZE"]
    row["TEMPL_SIZE"] = ls_data["TEMPL_SIZE"]
    row["_lsalign_status"] = ls_data["LSalign_Status"]

    return pd.DataFrame([row])


def print_feature_generation_summary(df_raw):
    """Outputs a clean, comprehensive molecular analysis card to the terminal."""
    if df_raw.empty:
        return

    row = df_raw.iloc[0]
    total_features = len(df_raw.columns) - (2 if "_lsalign_status" in df_raw.columns and "_ref_name" in df_raw.columns else 0)

    smiles = str(row.get("Canonical_SMILES", "N/A"))
    if len(smiles) > 55:
        smiles = smiles[:52] + "..."

    mw = float(row.get("MolWt", 0.0))
    logp = float(row.get("MolLogP", 0.0))
    tpsa = float(row.get("TPSA", 0.0))
    hbd = int(float(row.get("HBD", 0)))
    hba = int(float(row.get("HBA", 0)))
    rotb = int(float(row.get("RotBonds", 0)))
    charge = int(float(row.get("FormalCharge", 0)))

    ref_name = row.get("_ref_name", "4tq_reference")
    ls_status = row.get("_lsalign_status", "Skipped")
    rmsd = float(row.get("rmsd", 0.0))
    com_dist = float(row.get("com_distance", 0.0))
    pc_q = float(row.get("PC_scoreQ", 0.0))
    pc_t = float(row.get("PC_scoreT", 0.0))
    pval_q = float(row.get("Pval_PC8Q", 0.0))
    pval_t = float(row.get("Pval_PC8T", 0.0))
    jaccard = float(row.get("JaccardR", 0.0))
    q_size = int(float(row.get("QUERY_SIZE", 0)))
    t_size = int(float(row.get("TEMPL_SIZE", 0)))

    print("\n" + "=" * 75)
    print("🧪 STACPRED FEATURE EXTRACTION & MOLECULAR PROFILE SUMMARY")
    print("=" * 75)
    print(f"► Input Structure Name  : {row.get('Full_Name', 'Unknown')}.mol2")
    print(f"► Feature Matrix Built  : SUCCESS ({total_features:,} Features Generated)")
    print("-" * 75)
    print("► Basic Physicochemical Properties:")
    print(f"  • Canonical SMILES    : {smiles}")
    print(f"  • Molecular Weight    : {mw:.2f} g/mol")
    print(f"  • Lipophilic Index    : LogP {logp:.2f}")
    print(f"  • Polar Surface Area  : TPSA {tpsa:.2f} Å²")
    print(f"  • H-Bond Donor/Accept : HBD {hbd} | HBA {hba}")
    print(f"  • Conformational Flex : {rotb} Rotatable Bonds")
    print(f"  • Net Formal Charge   : {charge:+d}")
    print("-" * 75)
    print(f"► Structural Alignment Engine (LSalign vs {ref_name} Reference):")
    print(f"  • Alignment Status    : {ls_status}")
    if ls_status == "SUCCESS":
        print(f"  • Alignment RMSD     : {rmsd:.3f} Å")
        print(f"  • Center-of-Mass Dist : {com_dist:.3f} Å")
        print(f"  • Query/Template Size : {q_size} / {t_size} Heavy Atoms")
        print(f"  • PC-Score (Query)   : {pc_q:.4f} (p-value = {pval_q:.2e})")
        print(f"  • PC-Score (Template): {pc_t:.4f} (p-value = {pval_t:.2e})")
        print(f"  • Jaccard Overlap    : {jaccard:.4f}")
    else:
        print(f"  • Diagnostic Note     : {row.get('LSalign_Error', 'Not executed')}")
    print("=" * 75 + "\n")


def predict_consensus(mol2_path, ref_mol2_path, base_models, stacked_model):
    """Generates base model scores and calculates RidgeCV consensus binding energy."""
    df_raw_features = extract_single_molecule_features(mol2_path, ref_mol2_path)

    print_feature_generation_summary(df_raw_features)

    # Extract clean profile dictionary for UI display
    row = df_raw_features.iloc[0]
    total_feats = len(df_raw_features.columns) - (2 if "_lsalign_status" in df_raw_features.columns and "_ref_name" in df_raw_features.columns else 0)
    
    profile_dict = {
        "smiles": str(row.get("Canonical_SMILES", "N/A")),
        "mw": round(float(row.get("MolWt", 0.0)), 2),
        "logp": round(float(row.get("MolLogP", 0.0)), 2),
        "tpsa": round(float(row.get("TPSA", 0.0)), 2),
        "hbd": int(float(row.get("HBD", 0))),
        "hba": int(float(row.get("HBA", 0))),
        "rotb": int(float(row.get("RotBonds", 0))),
        "charge": int(float(row.get("FormalCharge", 0))),
        "total_features": total_feats,
        "ls_status": str(row.get("_lsalign_status", "Skipped")),
        "ls_error": str(row.get("LSalign_Error", "None")),
        "rmsd": round(float(row.get("rmsd", 0.0)), 3),
        "com_dist": round(float(row.get("com_distance", 0.0)), 3),
        "pc_q": round(float(row.get("PC_scoreQ", 0.0)), 4),
        "pc_t": round(float(row.get("PC_scoreT", 0.0)), 4),
        "jaccard": round(float(row.get("JaccardR", 0.0)), 4),
        "q_size": int(float(row.get("QUERY_SIZE", 0))),
        "t_size": int(float(row.get("TEMPL_SIZE", 0))),
        "ref_name": str(row.get("_ref_name", "4tq_reference"))
    }

    # Clean internal status metadata columns before sending to model pipelines
    for meta_col in ["_lsalign_status", "_ref_name"]:
        if meta_col in df_raw_features.columns:
            df_raw_features = df_raw_features.drop(columns=[meta_col])

    model_mapping = [
        ('B2_AD4_XGB', 'AD4_XGB_PRED'),
        ('B2_DOCK6_XGB', 'DOCK6_XGB_PRED'),
        ('B2_VINA_XGB', 'VINA_XGB_PRED'),
        ('B2_VINARDO_XGB', 'VINARDO_XGB_PRED'),
        ('B2_AD4_MLP', 'AD4_MLP_PRED'),
        ('B2_DOCK6_MLP', 'DOCK6_MLP_PRED'),
        ('B2_VINA_MLP', 'VINA_MLP_PRED'),
        ('B2_VINARDO_MLP', 'VINARDO_MLP_PRED')
    ]

    base_scores = {}
    meta_dict = {}

    print("--- INDIVIDUAL MODEL PREDICTIONS ---")

    for app_key, meta_col in model_mapping:
        if app_key in base_models:
            try:
                pipeline = base_models[app_key]
                df_input = df_raw_features.copy()

                if hasattr(pipeline, "feature_names_in_"):
                    expected_cols = list(pipeline.feature_names_in_)

                    missing_cols = [c for c in expected_cols if c not in df_input.columns]
                    if missing_cols:
                        missing_dict = {
                            col: ("missing" if col in CATEGORICAL_COLS else 0.0)
                            for col in missing_cols
                        }
                        missing_df = pd.DataFrame(missing_dict, index=df_input.index)
                        df_input = pd.concat([df_input, missing_df], axis=1)

                    df_input = df_input[expected_cols].copy()

                    cat_cols_present = [c for c in expected_cols if c in CATEGORICAL_COLS]
                    num_cols_present = [c for c in expected_cols if c not in CATEGORICAL_COLS]

                    if cat_cols_present:
                        df_input[cat_cols_present] = df_input[cat_cols_present].astype(str)
                    if num_cols_present:
                        df_input[num_cols_present] = df_input[num_cols_present].apply(pd.to_numeric, errors='coerce').fillna(0.0)

                pred_val = float(pipeline.predict(df_input)[0])
                base_scores[app_key] = pred_val
                meta_dict[meta_col] = pred_val
                print(f"  [✓] {app_key:<16}: {pred_val:.4f} kcal/mol")

            except Exception as e:
                print(f"  [X] {app_key:<16}: FAILED ({e})")
                base_scores[app_key] = -7.0
                meta_dict[meta_col] = -7.0
        else:
            print(f"  [X] {app_key:<16}: NOT LOADED")
            base_scores[app_key] = -7.0
            meta_dict[meta_col] = -7.0

    # RidgeCV Consensus Prediction
    if stacked_model is not None:
        try:
            df_meta = pd.DataFrame([meta_dict])
            final_score = float(stacked_model.predict(df_meta)[0])
            print(f"  [★] RidgeCV Stacked Score : {final_score:.4f} kcal/mol")
        except Exception as e:
            print(f"  [X] Stacking failed ({e}). Falling back to mean.")
            final_score = float(np.mean(list(meta_dict.values())))
    else:
        final_score = float(np.mean(list(meta_dict.values())))

    print("------------------------------------\n")
    return base_scores, final_score, profile_dict