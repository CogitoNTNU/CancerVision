import subprocess
from pathlib import Path
from multiprocessing import Pool

def run(file_tuple):
    file, output_dir, model_path = file_tuple
    output_file = output_dir / f"{file.parent.name}_{file.name}"

    cmd = [
        "nipreps-synthstrip",
        "-i", str(file),
        "-o", str(output_file),
        "--model", str(model_path),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

if __name__ == "__main__":  #
    input_dir = Path("res/data/Brats2024/training_data1_v2/BraTS-GLI-00005-100")
    output_dir = Path("res/data/output")
    model_path = Path("src/prototyping/skullstrip/synthstrip.1.pt")

    output_dir.mkdir(exist_ok=True)

    files = list(input_dir.rglob("*.nii.gz"))
    file_args = [(f, output_dir, model_path) for f in files]

    with Pool(2) as p:
        p.map(run, file_args)

    print("Done")