"""
Convert GOT-10k tracking results delimiter from tab to comma.

Usage:
    python convert_got10k_delimiter.py
"""

import os

# Path to reorganized GOT-10k results
RESULTS_DIR = "/mnt/disk2/Shawalt/SOT/OSTrack/output/test/tracking_results/ostrack/vitb_256_mae_ce_32x4_ep300/got-10k"

def convert_delimiter():
    if not os.path.exists(RESULTS_DIR):
        print(f"Error: Results directory not found: {RESULTS_DIR}")
        return

    # Get all sequence folders
    folders = [f for f in os.listdir(RESULTS_DIR) if os.path.isdir(os.path.join(RESULTS_DIR, f))]
    print(f"Found {len(folders)} sequence folders")

    converted_count = 0
    for folder in folders:
        folder_path = os.path.join(RESULTS_DIR, folder)

        # Process bbox txt file
        bbox_file = os.path.join(folder_path, f"{folder}.txt")
        if os.path.exists(bbox_file):
            with open(bbox_file, 'r') as f:
                content = f.read()

            # Replace tab with comma
            new_content = content.replace('\t', ',')

            with open(bbox_file, 'w') as f:
                f.write(new_content)
            converted_count += 1

        # Process time txt file
        time_file = os.path.join(folder_path, f"{folder}_time.txt")
        if os.path.exists(time_file):
            with open(time_file, 'r') as f:
                content = f.read()

            # Replace tab with comma
            new_content = content.replace('\t', ',')

            with open(time_file, 'w') as f:
                f.write(new_content)
            converted_count += 1

    print(f"Converted {converted_count} files to comma delimiter")

if __name__ == "__main__":
    convert_delimiter()