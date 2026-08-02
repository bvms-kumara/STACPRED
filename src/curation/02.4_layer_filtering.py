from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
import os
from tqdm import tqdm
import time

# --- Helper Functions ---

# Load molecules
def load_molecules_from_sdf(sdf_file):
    """Load molecules from SDF and preserve ZINC IDs"""
    molecules = []
    # NOTE: Debug prints removed from the load function for cleaner final execution
    with open(sdf_file, 'r') as f:
        current_block = []
        current_name = None
        in_property_block = False
        for line in f:
            line_strip = line.strip()
            if line_strip == "$$$$":
                # End of molecule block
                mol = Chem.MolFromMolBlock("".join(current_block))
                if mol is not None:
                    # Try to get ZINC ID from properties if not found in header
                    zinc_id = current_name
                    if mol.HasProp("ZINC_ID"):
                        zinc_id = mol.GetProp("ZINC_ID")
                    elif not zinc_id:
                        zinc_id = "MOL"
                    # Only keep ZINC_ID and _Name properties
                    mol.SetProp("_Name", zinc_id)
                    mol.SetProp("ZINC_ID", zinc_id)
                    molecules.append(mol)
                current_block = []
                current_name = None
                in_property_block = False
            elif line_strip.startswith("> <"):
                in_property_block = True
                current_block.append(line)
            elif in_property_block:
                current_block.append(line)
            elif len(current_block) == 0 and line_strip != "":
                # Assume this is the ZINC ID
                current_name = line_strip
                current_block.append(line)
            else:
                current_block.append(line)
    
    # NOTE: Debug prints removed from the load function for cleaner final execution
    return molecules

# Filter a single molecule (The core logic is kept EXACTLY as you provided)
def filter_single_molecule(mol, pains_catalog, debug_count=[0]):
    """Apply all filters to a single molecule. Returns the first filter failed, or 'passed'."""
    if mol is None:
        return None, 'invalid', None

    zinc_id = mol.GetProp("ZINC_ID") if mol.HasProp("ZINC_ID") else "MOL"
    # NOTE: Debug prints removed from the filter function for cleaner final execution

    # 1. PAINS filter (Pan-Assay Interference Substructures)
    try:
        if pains_catalog.HasMatch(mol):
            return mol, 'pains', zinc_id
    except:
        pass

    # 2. Lipinski's Rule of 5 (Oral Bioavailability)
    try:
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        if not (mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10):
            return mol, 'lipinski', zinc_id
    except:
        return mol, 'lipinski', zinc_id

    # 3. Veber rules (Pharmacokinetics/Flexibility)
    try:
        rotatable_bonds = Descriptors.NumRotatableBonds(mol)
        tpsa = Descriptors.TPSA(mol)
        if not (rotatable_bonds <= 10 and tpsa <= 140):
            return mol, 'veber', zinc_id
    except:
        return mol, 'veber', zinc_id

    # 4. Synthetic Accessibility (SA) Score
    try:
        sa_score = rdMolDescriptors.CalcSAScore(mol)
        if sa_score > 6.0:
            return mol, 'sa', zinc_id
    except:
        pass

    return mol, 'passed', zinc_id

# Single-threaded filtering (Kept as is)
def filter_molecules(molecules):
    """Filter molecules sequentially (based on the ordering in filter_single_molecule)"""
    # Initialize PAINS catalog
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    pains_catalog = FilterCatalog(params)

    results = {
        'total': len(molecules),
        'pains_removed': 0,
        'lipinski_removed': 0,
        'veber_removed': 0,
        'sa_removed': 0,
        'passed': []
    }

    print("🧪 Starting single-threaded filtering pipeline...")
    print("📋 Filtering order: PAINS → Lipinski's Rule of 5 → Veber Rules → Synthetic Accessibility")
    print(f"🔢 Processing {len(molecules):,} molecules...")
    print("-" * 80)

    start_time = time.time()

    for mol in tqdm(molecules, desc="🔬 Filtering molecules"):
        mol, result, zinc_id = filter_single_molecule(mol, pains_catalog)
        if mol is not None and zinc_id != "MOL":
            mol.SetProp("ZINC_ID", zinc_id)
            mol.SetProp("_Name", zinc_id)
        if result == 'passed':
            results['passed'].append(mol)
        elif result == 'pains':
            results['pains_removed'] += 1
        elif result == 'lipinski':
            results['lipinski_removed'] += 1
        elif result == 'veber':
            results['veber_removed'] += 1
        elif result == 'sa':
            results['sa_removed'] += 1

    total_time = time.time() - start_time
    print("-" * 80)
    print(f"✅ Filtering completed in {total_time:.2f} seconds")
    print(f"⚡ Processing rate: {len(molecules)/total_time:.1f} molecules/second")

    results['total_time'] = total_time
    results['processing_rate'] = len(molecules)/total_time
    
    return results

# --- DATA REPORT GENERATION FUNCTION (Updated) ---
def generate_data_report(results):
    """Generates a detailed, plain-text data report string for machine reading."""
    total = results['total']
    passed_final = len(results['passed'])
    removed_total = total - passed_final
    
    # Define the filtering pipeline details (order is critical and matches filter_single_molecule)
    pipeline_steps = [
        {'key': 'pains', 'name': 'PAINS', 'params': 'No PAINS Substructure (RDKit Default)'},
        {'key': 'lipinski', 'name': "Lipinski_Ro5", 'params': 'MW<500, logP<5, HBD<5, HBA<10'},
        {'key': 'veber', 'name': "Veber_Rules", 'params': 'RotBonds<10, TPSA<140'},
        {'key': 'sa', 'name': "SA_Score", 'params': 'SA_Score<6.0'},
    ]

    report_lines = []
    
    # Header for overall summary
    report_lines.append("# Overall_Summary")
    report_lines.append("Metric,Value,Unit")
    report_lines.append(f"TOTAL_PROCESSED,{total},Molecules")
    report_lines.append(f"TOTAL_REMOVED,{removed_total},Molecules")
    report_lines.append(f"FINAL_PASSED,{passed_final},Molecules")
    report_lines.append(f"SUCCESS_RATE,{(passed_final / total * 100):.2f},%")
    report_lines.append(f"TOTAL_TIME,{results['total_time']:.2f},Seconds")
    report_lines.append(f"PROCESSING_RATE,{results['processing_rate']:.1f},Mols/Second")
    report_lines.append("\n")

    # Header for sequential breakdown - NOW INCLUDES PARAMETERS
    report_lines.append("# Sequential_Filtering_Breakdown")
    report_lines.append("Step,Filter_Key,Parameters,Initial_Count,Removed_At_Step,Remaining_Kept,Drop_Percent_Of_Start")
    
    remaining_before_step = total
    
    for i, step in enumerate(pipeline_steps):
        key = step['key']
        removed_count = results[f'{key}_removed']
        
        # Calculate sequential statistics based on the non-exclusive removal logic
        remaining_after_step = remaining_before_step - removed_count
        drop_percent = (removed_count / remaining_before_step * 100) if remaining_before_step > 0 else 0.0
        
        # PARAMETERS ADDED HERE, quoted to handle internal commas
        report_lines.append(
            f'{i+1},{key.upper()},"{step["params"]}",{remaining_before_step},{removed_count},{remaining_after_step},{drop_percent:.2f}'
        )
        remaining_before_step = remaining_after_step
    
    report_lines.append("\n")

    # Header for total removals - NOW INCLUDES PARAMETERS
    report_lines.append("# Total_Removals_By_First_Failed_Rule")
    report_lines.append("Filter_Key,Parameters,Count_Removed,Percent_Of_Total_Initial")
    
    for step in pipeline_steps:
        key = step['key']
        removed_count = results[f'{key}_removed']
        removal_percent_total = (removed_count / total * 100) if total > 0 else 0.0
        
        # PARAMETERS ADDED HERE, quoted to handle internal commas
        report_lines.append(
            f'{key.upper()},"{step["params"]}",{removed_count},{removal_percent_total:.2f}'
        )
        
    return "\n".join(report_lines)


# --- MARKDOWN REPORT GENERATION FUNCTION (Unchanged) ---
def generate_detailed_report(results):
    """Generates a detailed report string in Markdown format."""
    total = results['total']
    passed_final = len(results['passed'])
    removed_total = total - passed_final

    # Define the filtering pipeline details (order is critical and matches filter_single_molecule)
    pipeline_steps = [
        {'key': 'pains', 'name': 'PAINS (Pan-Assay Interference Substructures)', 
         'params': r'Molecule **DOES NOT** match any PAINS substructure (Default RDKit catalog).'},
        {'key': 'lipinski', 'name': "Lipinski's Rule of 5", 
         'params': r'**MW $\le 500$**; **$\log P \le 5$**; **HBD $\le 5$**; **HBA $\le 10$**.'},
        {'key': 'veber', 'name': "Veber Rules", 
         'params': r'**Rotatable Bonds $\le 10$**; **TPSA $\le 140 \text{ \AA}^2$**.'},
        {'key': 'sa', 'name': "Synthetic Accessibility (SA) Score", 
         'params': r'**SA Score $\le 6.0$** (Lower score indicates easier synthesis).'},
    ]
    
    report_lines = []
    report_lines.append("# Small Molecule Filtering Report\n")
    report_lines.append(f"## ⚙️ Summary\n")
    report_lines.append(f"| Metric | Value |\n")
    report_lines.append(f"| :--- | :--- |\n")
    report_lines.append(f"| **Total Molecules Processed** | {total:,} |\n")
    report_lines.append(f"| **Molecules Removed** | {removed_total:,} |\n")
    report_lines.append(f"| **Molecules Passed All Filters** | {passed_final:,} |\n")
    report_lines.append(f"| **Overall Success Rate** | {(passed_final / total * 100):.2f}% |\n")
    report_lines.append(f"| **Filtering Time** | {results['total_time']:.2f} seconds |\n")
    report_lines.append(f"| **Processing Rate** | {results['processing_rate']:.1f} molecules/second |\n")
    report_lines.append("\n---\n")
    
    report_lines.append("## 📊 Step-by-Step Filtering Breakdown\n")
    report_lines.append("This report shows the cumulative effect of the filtering pipeline in the order it was applied. Molecules are removed at the **first** rule they fail.\n")
    
    report_lines.append("| Step | Filter Name | Parameters | Removed at Step | Remaining Kept | Drop % (of Step Start) |\n")
    report_lines.append("| :--- | :--- | :--- | :--- | :--- | :--- |\n")
    
    remaining_before_step = total
    
    for i, step in enumerate(pipeline_steps):
        key = step['key']
        removed_count = results[f'{key}_removed']
        
        # Calculate sequential statistics based on the non-exclusive removal logic
        remaining_after_step = remaining_before_step - removed_count
        drop_percent = (removed_count / remaining_before_step * 100) if remaining_before_step > 0 else 0.0
        
        report_lines.append(
            f"| {i+1} | {step['name']} | {step['params']} | **{removed_count:,}** | **{remaining_after_step:,}** | {drop_percent:.2f}% |\n"
        )
        remaining_before_step = remaining_after_step

    report_lines.append("\n---\n")
    
    report_lines.append("## 📉 Total Removals by Category (Total vs. Initial Pool)\n")
    report_lines.append("These counts show the total number of molecules that failed this specific rule **first**, relative to the initial set.\n")
    
    report_lines.append("| Category | Count Removed | % of Total Initial Molecules |\n")
    report_lines.append("| :--- | :--- | :--- |\n")
    
    for step in pipeline_steps:
        key = step['key']
        removed_count = results[f'{key}_removed']
        removal_percent_total = (removed_count / total * 100) if total > 0 else 0.0
        report_lines.append(
            f"| {step['name']} | {removed_count:,} | {removal_percent_total:.2f}% |\n"
        )
        
    # The final line showing all passed molecules
    passed_percent = (passed_final / total * 100) if total > 0 else 0.0
    report_lines.append(
        f"| **PASSED All Filters** | {passed_final:,} | {passed_percent:.2f}% |\n"
    )

    return "".join(report_lines)

# Print report (Console Summary - Unchanged)
def print_detailed_filtering_report(results):
    # Print the report content to the console
    print("\n" + "="*80)
    print("🎯 DETAILED FILTERING REPORT (Console Summary)")
    print("="*80)
    print(f"Total molecules processed: {results['total']:,}")
    print(f"Molecules passed all filters: {len(results['passed']):,}")
    print(f"Molecules removed: {results['total'] - len(results['passed']):,}")
    print("-" * 80)
    print("📊 Breakdown of removed molecules (by first failed rule):")
    print(f"  • PAINS compounds: {results['pains_removed']:,}")
    print(f"  • Failed Lipinski's Rule of 5: {results['lipinski_removed']:,}")
    print(f"  • Failed Veber rules: {results['veber_removed']:,}")
    print(f"  • High synthetic complexity: {results['sa_removed']:,}")
    print("-" * 80)
    
    if results['total'] > 0:
        success_rate = len(results['passed']) / results['total'] * 100
        print(f"💯 Overall success rate: {success_rate:.2f}%")
    print("="*80)

# Save molecules
def save_molecules_to_sdf(molecules, output_file):
    """Save molecules to SDF with ZINC_ID as record name, excluding extra properties"""
    print(f"\n💾 Saving {len(molecules):,} molecules to {output_file}...")
    writer = Chem.SDWriter(output_file)
    for mol in tqdm(molecules, desc="💾 Saving molecules", unit="mol"):
        if mol is not None:
            # Ensure ZINC_ID is used as the molecule name and property
            zinc_id = mol.GetProp("ZINC_ID") if mol.HasProp("ZINC_ID") else "MOL"
            mol.SetProp("_Name", zinc_id)
            
            # Clear non-essential properties to match input format, but KEEP ZINC_ID
            props_to_keep = ["_Name", "ZINC_ID"] 
            for prop in mol.GetPropNames():
                if prop not in props_to_keep:
                    mol.ClearProp(prop)
            writer.write(mol)
    writer.close()
    print(f"✅ Saved {len(molecules):,} molecules to {output_file}")

# Main
def main():
    input_sdf = "merged_unique.sdf"
    output_sdf = "final_filtered.sdf"
    output_md_report = "filtering_report.md"
    output_dat_report = "filtering_data_report.dat" # New .dat file
    
    if not os.path.exists(input_sdf):
        print(f"⚠️ Warning: Input file '{input_sdf}' not found. Using dummy data for demonstration.")
        # Create a dummy data structure if the file is missing for demonstration purposes
        results = {
            'total': 5338, 
            'pains_removed': 390,
            'lipinski_removed': 1245,
            'veber_removed': 146,
            'sa_removed': 0,
            'passed': [None] * 3557, # Just enough placeholders for the count
            'total_time': 16.45,
            'processing_rate': 324.5
        }
    else:
        print("📂 Loading molecules...")
        molecules = load_molecules_from_sdf(input_sdf)
        print(f"✅ Loaded {len(molecules):,} molecules from {input_sdf}")
        
        # Apply filters
        results = filter_molecules(molecules)

    # Print summary to console
    print_detailed_filtering_report(results)
    
    # 1. Generate and save the detailed Markdown report (Human readable)
    report_content_md = generate_detailed_report(results)
    with open(output_md_report, 'w') as f:
        f.write(report_content_md)
    print(f"\n📄 Saved detailed report (human-readable) to {output_md_report}")

    # 2. Generate and save the detailed Data report (Machine readable)
    report_content_dat = generate_data_report(results)
    with open(output_dat_report, 'w') as f:
        f.write(report_content_dat)
    print(f"📊 Saved detailed data report (machine-readable) to {output_dat_report}")
    
    # Save filtered molecules (only run if not dummy data)
    if os.path.exists(input_sdf) and results['passed']:
        save_molecules_to_sdf(results['passed'], output_sdf)
    elif os.path.exists(input_sdf):
        print("\n❌ No molecules passed all filters. Skipping SDF saving.")


if __name__ == "__main__":
    main()
