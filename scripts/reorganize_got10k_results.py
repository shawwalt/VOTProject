"""
Reorganize GOT-10k tracking results into sequence-named folders.

Each sequence folder contains:
- <sequence_name>.txt (tracking bounding boxes)
- <sequence_name>_time.txt (processing time per frame)

Usage:
    python reorganize_got10k_results.py
"""

import os
import shutil

# Path to GOT-10k results
RESULTS_DIR = "/mnt/disk2/Shawalt/SOT/OSTrack/output/test/tracking_results/ostrack/vitb_256_mae_ce_32x4_ep300/got-10k"

def reorganize_got10k_results():
    if not os.path.exists(RESULTS_DIR):
        print(f"Error: Results directory not found: {RESULTS_DIR}")
        return

    # Get all txt files
    files = [f for f in os.listdir(RESULTS_DIR) if f.endswith('.txt')]

    # Separate bbox and time files
    bbox_files = [f for f in files if not f.endswith('_time.txt')]
    time_files = [f for f in files if f.endswith('_time.txt')]

    print(f"Found {len(bbox_files)} bbox files and {len(time_files)} time files")

    moved_count = 0
    for bbox_file in bbox_files:
        seq_name = bbox_file.replace('.txt', '')
        time_file = f"{seq_name}_time.txt"

        # Create sequence folder
        seq_folder = os.path.join(RESULTS_DIR, seq_name)
        os.makedirs(seq_folder, exist_ok=True)

        # Move bbox file
        src_bbox = os.path.join(RESULTS_DIR, bbox_file)
        dst_bbox = os.path.join(seq_folder, bbox_file)
        if os.path.exists(src_bbox) and not os.path.exists(dst_bbox):
            shutil.move(src_bbox, dst_bbox)
            moved_count += 1

        # Move time file if exists
        src_time = os.path.join(RESULTS_DIR, time_file)
        dst_time = os.path.join(seq_folder, time_file)
        if os.path.exists(src_time) and not os.path.exists(dst_time):
            shutil.move(src_time, dst_time)
            moved_count += 1

    print(f"Reorganized {moved_count} files into sequence folders")

    # Verify structure
    folders = [f for f in os.listdir(RESULTS_DIR) if os.path.isdir(os.path.join(RESULTS_DIR, f))]
    print(f"Created {len(folders)} sequence folders")

if __name__ == "__main__":
    reorganize_got10k_results()