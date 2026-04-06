"""
Convert TrackingNet tracking results to standard format.

Standard format:
- Values: x,y,w,h with scientific notation (e.g., 5.790000000000000000e+02)
- Delimiter: comma

Usage:
    python convert_trackingnet_to_standard.py
"""

import os

# Path to TrackingNet results
RESULTS_DIR = "/mnt/disk2/Shawalt/SOT/OSTrack/output/test/tracking_results/ostrack/vitb_256_mae_ce_32x4_ep300/trackingnet"

def convert_to_scientific_notation(value):
    """Convert a value to scientific notation string with 18 decimal places."""
    return f"{float(value):.18e}"

def convert_file(src_path):
    """Convert a single file to standard format."""
    with open(src_path, 'r') as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        line = line.strip()
        if line:
            # Split by tab or comma
            parts = line.replace('\t', ',').split(',')
            if len(parts) >= 4:
                # Convert each value to scientific notation
                new_parts = [convert_to_scientific_notation(p) for p in parts[:4]]
                new_lines.append(','.join(new_parts))

    with open(src_path, 'w') as f:
        f.write('\n'.join(new_lines))

def convert_trackingnet_results():
    if not os.path.exists(RESULTS_DIR):
        print(f"Error: Results directory not found: {RESULTS_DIR}")
        return

    # Get all txt files (exclude time files)
    files = [f for f in os.listdir(RESULTS_DIR) if f.endswith('.txt') and not f.endswith('_time.txt')]
    print(f"Found {len(files)} bbox files")

    converted_count = 0
    for filename in files:
        filepath = os.path.join(RESULTS_DIR, filename)
        convert_file(filepath)
        converted_count += 1

    print(f"Converted {converted_count} files to standard format")

    # Show example
    print("\n=== Example conversion ===")
    example_file = os.path.join(RESULTS_DIR, "__WaG8fRMto_0.txt")
    if os.path.exists(example_file):
        print(f"Content of {filename}:")
        with open(example_file, 'r') as f:
            for i, line in enumerate(f):
                if i < 5:
                    print(f"  {line.strip()}")
                else:
                    break

if __name__ == "__main__":
    convert_trackingnet_results()