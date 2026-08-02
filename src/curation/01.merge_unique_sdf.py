from rdkit import Chem
import glob

# Collect all SDF file paths
sdf_files = glob.glob("*.sdf")

unique_ids = set()
unique_mols = []

for file in sdf_files:
    suppl = Chem.SDMolSupplier(file, removeHs=False)
    for mol in suppl:
        if mol is None:
            continue
        # Try to extract ZINC ID (stored in title or property block)
        mol_id = mol.GetProp("_Name") if mol.HasProp("_Name") else None

        if not mol_id:
            # Sometimes ZINC ID appears in a property field like 'ZINC_ID'
            for prop in mol.GetPropNames():
                if "ZINC" in mol.GetProp(prop):
                    mol_id = mol.GetProp(prop)
                    break

        # Skip if still not found
        if not mol_id:
            continue

        if mol_id not in unique_ids:
            unique_ids.add(mol_id)
            unique_mols.append(mol)

print(f"✅ Collected {len(unique_mols)} unique molecules from {len(sdf_files)} files.")

# Save merged file
writer = Chem.SDWriter("merged_unique.sdf")
for m in unique_mols:
    writer.write(m)
writer.close()

print("💾 Saved to merged_unique.sdf")
