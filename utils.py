import os

# ---------- Create output folders ----------
def create_folder(matched_dir, unmatched_dir):
    os.makedirs(matched_dir, exist_ok=True)
    os.makedirs(unmatched_dir, exist_ok=True)