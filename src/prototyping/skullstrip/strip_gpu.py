import subprocess
from pathlib import Path

# ===== CONFIG =====
input_dir = Path("normal")
output_dir = Path("output")
model_path = Path("synthstrip.1.pt")

output_dir.mkdir(exist_ok=True)

# ===== FIND FILES =====
files = list(input_dir.rglob("*.nii.gz"))
print(f"Found {len(files)} files to process.")

# ===== PROCESS FILES =====
for file in files:
    output_file = output_dir / f"{file.parent.name}_{file.name}"

    cmd = [
        "nipreps-synthstrip",
        "-i", str(file),
        "-o", str(output_file),
        "--model", str(model_path),
        "-g"  # use GPU
    ]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

print("Done")