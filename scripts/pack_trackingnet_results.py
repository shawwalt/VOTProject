"""
Pack TrackingNet tracking results into a zip file (without time files).

Usage:
    python pack_trackingnet_results.py
"""

import os
import zipfile

# Path to TrackingNet results
RESULTS_DIR = "/mnt/disk2/Shawalt/SOT/OSTrack/output/test/tracking_results/ostrack/vitb_256_mae_ce_32x4_ep300/trackingnet"
OUTPUT_ZIP = "/mnt/disk2/Shawalt/SOT/OSTrack/output/test/tracking_results/ostrack/vitb_256_mae_ce_32x4_ep300/trackingnet_results.zip"

def pack_trackingnet_results():
    if not os.path.exists(RESULTS_DIR):
        print(f"Error: Results directory not found: {RESULTS_DIR}")
        return

    # Get all txt files (exclude time files)
    files = [f for f in os.listdir(RESULTS_DIR) if f.endswith('.txt') and not f.endswith('_time.txt')]
    print(f"Found {len(files)} bbox files")

    # Create zip file
    with zipfile.ZipFile(OUTPUT_ZIP, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for filename in files:
            filepath = os.path.join(RESULTS_DIR, filename)
            zipf.write(filepath, filename)
            print(f"  Added: {filename}")

    print(f"\nCreated zip file: {OUTPUT_ZIP}")
    print(f"Total files: {len(files)}")

if __name__ == "__main__":
    pack_trackingnet_results()