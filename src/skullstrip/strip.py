import subprocess
from pathlib import Path

# ================= CONFIG =================
BASE_DIR = Path(__file__).resolve().parents[3]

input_dir = BASE_DIR / "res" / "dataset" / "IXI-T2"
output_dir = BASE_DIR / "res" / "skullstripped" / "IXI-T2"
model_path = BASE_DIR / "src" / "prototyping" / "skullstrip" / "synthstrip.1.pt"

output_dir.mkdir(parents=True, exist_ok=True)

files = list(input_dir.rglob("*.nii.gz"))
print(f"Found {len(files)} files to process.")

# ================= PROCESS FILES =================
for file in files:
    output_file = output_dir / f"{file.parent.name}_{file.stem}.gz"

    if output_file.exists():
        print(f"Skipping {file.name}, already processed.")
        continue

    cmd = [
        "nipreps-synthstrip",
        "-i", str(file),
        "-o", str(output_file),
        "--model", str(model_path),
        "-g",  # GPU acceleration
        "-n", "16"
    ]

    print(f"Processing {file.name} ...")
    print("Running:", " ".join(cmd))

    subprocess.run(cmd, check=True)

print("All files processed.")