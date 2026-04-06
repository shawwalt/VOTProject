"""
Create Excel spreadsheet for benchmark results.

Usage:
    python create_benchmark_excel.py
"""

import pandas as pd
import os

# Output path
OUTPUT_DIR = "/mnt/disk2/Shawalt/SOT/OSTrack"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "OSTrack256_Benchmark_Results.xlsx")

# Training Configuration
train_config = {
    'Method': 'OSTrack256',
    'Batch Size (B)': 64,
    'Learning Rate (LR)': 0.0008,
    'Epoch': 300,
}

# Benchmark Results (with Method Name column)
results = {
    'Method': ['OSTrack256', 'OSTrack256', 'OSTrack256', 'OSTrack256', 'OSTrack256', 'OSTrack256', 'OSTrack256', 'OSTrack256', 'OSTrack256', 'OSTrack256', 'OSTrack256', 'OSTrack256'],
    'Dataset': ['TrackingNet', 'TrackingNet', 'TrackingNet', 'TrackingNet', 'GOT-10k', 'GOT-10k', 'GOT-10k', 'LaSOT', 'LaSOT', 'LaSOT', 'LaSOT', 'LaSOT'],
    'Metric': ['Coverage', 'Precision', 'Normalized Precision', 'Success', 'AO', 'SR0.5', 'SR0.75', 'AUC', 'OP50', 'OP75', 'Precision', 'Norm Precision'],
    'Value': [100.0, 80.24507577982098, 86.57879731434305, 81.81199744743084, 0.695, 0.778, 0.653, 66.31, 77.31, 65.00, 71.07, 75.54],
}

# Create DataFrames
df_config = pd.DataFrame([train_config])
df_results = pd.DataFrame(results)

# Write to Excel with formatting
with pd.ExcelWriter(OUTPUT_FILE, engine='openpyxl') as writer:
    # Write config and results in the same sheet
    df_config.to_excel(writer, sheet_name='Benchmark Results', index=False, startrow=0)
    df_results.to_excel(writer, sheet_name='Benchmark Results', index=False, startrow=3)

    # Auto-adjust column widths
    workbook = writer.book
    worksheet = workbook['Benchmark Results']
    for column in worksheet.columns:
        max_length = 0
        column_letter = column[0].column_letter
        for cell in column:
            try:
                if len(str(cell.value)) > max_length:
                    max_length = len(str(cell.value))
            except:
                pass
        adjusted_width = min(max_length + 2, 30)
        worksheet.column_dimensions[column_letter].width = adjusted_width

print(f"Created Excel file: {OUTPUT_FILE}")

# Also display the data
print("\n" + "="*60)
print("Training Configuration")
print("="*60)
for k, v in train_config.items():
    print(f"  {k}: {v}")

print("\n" + "="*60)
print("Benchmark Results")
print("="*60)
print(f"{'Method':<15} {'Dataset':<15} {'Metric':<25} {'Value':<15}")
print("-"*70)
for i, row in df_results.iterrows():
    print(f"{row['Method']:<15} {row['Dataset']:<15} {row['Metric']:<25} {row['Value']:<15.4f}")