import pandas as pd
import os

# Original CSV files
csv_files = [
    'b2_ad4_final_results_2.csv',
    'b2_dock6_final_results_2.csv',
    'b2_vina_final_results_2.csv',
    'b2_vinardo_final_results_2.csv'
]

# Directory for cleaned CSVs
clean_dir = 'cleaned_csv'
os.makedirs(clean_dir, exist_ok=True)

# Summary report data
summary_data = []

for file in csv_files:
    df = pd.read_csv(file)
    total_rows = len(df)
    
    # Keep only rows where Canonical_SMILES is not empty
    df_clean = df[df['Canonical_SMILES'].notna() & (df['Canonical_SMILES'] != '')]
    cleaned_rows = len(df_clean)
    removed_rows = total_rows - cleaned_rows
    removed_percentage = (removed_rows / total_rows) * 100
    
    # Save cleaned CSV
    clean_file_path = os.path.join(clean_dir, file.replace('.csv', '_clean.csv'))
    df_clean.to_csv(clean_file_path, index=False)
    
    # Append to summary
    summary_data.append({
        'File': file,
        'Total_Rows': total_rows,
        'Cleaned_Rows': cleaned_rows,
        'Removed_Rows': removed_rows,
        'Removed_Percentage': removed_percentage
    })

# Save summary report
summary_df = pd.DataFrame(summary_data)
summary_df.to_csv('cleaning_summary_report.csv', index=False)
print("Cleaning complete. Summary saved to 'cleaning_summary_report.csv'")

