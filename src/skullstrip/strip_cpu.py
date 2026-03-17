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
        "-n", "4"
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

if __name__ == "__main__":  #
    BASE_DIR = Path(__file__).resolve().parents[3]
    input_dir = BASE_DIR / "res" / "dataset" / "IXI-T1"
    output_dir = BASE_DIR / "res" / "skullstripped" / "IXI-T1"
    model_path = BASE_DIR / "src" / "prototyping" / "skullstrip" / "synthstrip.1.pt"

    output_dir.mkdir(parents=True, exist_ok=True)

    files = list(input_dir.rglob("*.nii.gz"))
    file_args = [(f, output_dir, model_path) for f in files]

    with Pool(2) as p:
        p.map(run, file_args)

    print("Done")