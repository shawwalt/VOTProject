"""
Convert GOT-10k tracking results to standard format.

Standard format:
- File name: <sequence_name>_001.txt
- Values: x,y,w,h with 4 decimal places (e.g., 395.0000,340.0000,532.0000,407.0000)

Usage:
    python convert_got10k_to_standard.py
"""

import os
import shutil

# Path to reorganized GOT-10k results
RESULTS_DIR = "/mnt/disk2/Shawalt/SOT/OSTrack/output/test/tracking_results/ostrack/vitb_256_mae_ce_32x4_ep300/got-10k"

def convert_to_standard_format():
    if not os.path.exists(RESULTS_DIR):
        print(f"Error: Results directory not found: {RESULTS_DIR}")
        return

    # Get all sequence folders
    folders = [f for f in os.listdir(RESULTS_DIR) if os.path.isdir(os.path.join(RESULTS_DIR, f))]
    print(f"Found {len(folders)} sequence folders")

    converted_count = 0
    for folder in folders:
        folder_path = os.path.join(RESULTS_DIR, folder)

        # Process bbox txt file: <seq_name>.txt -> <seq_name>_001.txt
        src_bbox = os.path.join(folder_path, f"{folder}.txt")
        dst_bbox = os.path.join(folder_path, f"{folder}_001.txt")

        if os.path.exists(src_bbox):
            # Read content
            with open(src_bbox, 'r') as f:
                lines = f.readlines()

            # Convert to 4 decimal places
            new_lines = []
            for line in lines:
                line = line.strip()
                if line:
                    parts = line.split(',')
                    if len(parts) == 4:
                        # Convert each value to float with 4 decimal places
                        new_parts = [f"{float(p):.4f}" for p in parts]
                        new_lines.append(','.join(new_parts))
                    else:
                        new_lines.append(line)

            # Write to new file with _001 suffix
            with open(dst_bbox, 'w') as f:
                f.write('\n'.join(new_lines))

            # Remove old file
            os.remove(src_bbox)
            converted_count += 1

        # Process time txt file (keep same name, just ensure comma delimiter)
        time_file = os.path.join(folder_path, f"{folder}_time.txt")
        if os.path.exists(time_file):
            with open(time_file, 'r') as f:
                content = f.read()
            # Ensure comma delimiter
            new_content = content.replace('\t', ',')
            with open(time_file, 'w') as f:
                f.write(new_content)

    print(f"Converted {converted_count} files to standard format")

    # Show example
    print("\n=== Example conversion ===")
    example_folder = os.path.join(RESULTS_DIR, "GOT-10k_Test_000001")
    if os.path.exists(example_folder):
        print(f"Files in {example_folder}:")
        for f in os.listdir(example_folder):
            print(f"  {f}")
        print("\nContent of _001.txt:")
        with open(os.path.join(example_folder, "GOT-10k_Test_000001_001.txt"), 'r') as f:
            for i, line in enumerate(f):
                if i < 5:
                    print(f"  {line.strip()}")
                else:
                    break

if __name__ == "__main__":
    convert_to_standard_format()