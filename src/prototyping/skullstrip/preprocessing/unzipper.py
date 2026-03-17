import os
import zipfile
import shutil
from pathlib import Path

# =========================
# KONFIGURASJON
# =========================

# Mappen der du har mange zip-filer / undermapper
SOURCE_ROOT = Path("path/to/your/dataset_root")

# Midlertidig mappe der alt pakkes ut
EXTRACT_ROOT = Path("path/to/extracted_all")

# Endelig sortert output
OUTPUT_ROOT = Path("path/to/sorted_dataset")
HEALTHY_DIR = OUTPUT_ROOT / "healthy"
UNHEALTHY_DIR = OUTPUT_ROOT / "unhealthy"

# Nøkkelord for å kjenne igjen unhealthy / healthy
UNHEALTHY_KEYWORDS = [
    "tumor", "tumour", "glioma", "cancer", "lesion", "mass",
    "abnormal", "metastasis", "neoplasm", "svulst"
]

HEALTHY_KEYWORDS = [
    "healthy", "normal", "control", "non-tumor", "nontumor",
    "non_tumor", "no_tumor", "notumor", "benign_free"
]


# =========================
# HJELPEFUNKSJONER
# =========================

def ensure_dirs():
    HEALTHY_DIR.mkdir(parents=True, exist_ok=True)
    UNHEALTHY_DIR.mkdir(parents=True, exist_ok=True)
    EXTRACT_ROOT.mkdir(parents=True, exist_ok=True)


def is_zip_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() == ".zip"


def find_all_zip_files(root: Path):
    return [p for p in root.rglob("*") if is_zip_file(p)]


def safe_extract_zip(zip_path: Path, extract_to: Path):
    """
    Pakker ut én zip-fil til en egen undermappe.
    """
    target_dir = extract_to / zip_path.stem
    target_dir.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(target_dir)
        print(f"Unzipped: {zip_path} -> {target_dir}")
    except zipfile.BadZipFile:
        print(f"ADVARSEL: Klarte ikke åpne zip-fil: {zip_path}")


def extract_all_zips_recursive(source_root: Path, extract_root: Path):
    """
    Finner alle zip-filer i source_root og pakker dem ut.
    Kjøres i flere runder slik at også zip-filer som ligger inni andre zip-filer blir tatt med.
    """
    seen = set()

    while True:
        zip_files = [p for p in list(source_root.rglob("*.zip")) + list(extract_root.rglob("*.zip")) if p not in seen]
        if not zip_files:
            break

        for zip_path in zip_files:
            seen.add(zip_path)
            safe_extract_zip(zip_path, extract_root)


def find_all_nii_gz(root: Path):
    return [p for p in root.rglob("*.nii.gz") if p.is_file()]


def classify_file(path: Path) -> str | None:
    """
    Returnerer:
    - 'healthy'
    - 'unhealthy'
    - None hvis den ikke klarer å bestemme
    """
    text = str(path).lower()

    if any(keyword in text for keyword in UNHEALTHY_KEYWORDS):
        return "unhealthy"

    if any(keyword in text for keyword in HEALTHY_KEYWORDS):
        return "healthy"

    return None


def unique_destination(dest_dir: Path, original_name: str) -> Path:
    """
    Lager unikt filnavn hvis det allerede finnes en fil med samme navn.
    """
    candidate = dest_dir / original_name
    if not candidate.exists():
        return candidate

    if original_name.endswith(".nii.gz"):
        base = original_name[:-7]
        ext = ".nii.gz"
    else:
        base = Path(original_name).stem
        ext = Path(original_name).suffix

    i = 1
    while True:
        new_name = f"{base}_{i}{ext}"
        candidate = dest_dir / new_name
        if not candidate.exists():
            return candidate
        i += 1


def sort_nii_files(extract_root: Path):
    nii_files = find_all_nii_gz(extract_root)

    healthy_count = 0
    unhealthy_count = 0
    skipped_count = 0

    for nii_file in nii_files:
        label = classify_file(nii_file)

        if label == "healthy":
            dst = unique_destination(HEALTHY_DIR, nii_file.name)
            shutil.copy2(nii_file, dst)
            healthy_count += 1
            print(f"[HEALTHY]   {nii_file} -> {dst}")

        elif label == "unhealthy":
            dst = unique_destination(UNHEALTHY_DIR, nii_file.name)
            shutil.copy2(nii_file, dst)
            unhealthy_count += 1
            print(f"[UNHEALTHY] {nii_file} -> {dst}")

        else:
            skipped_count += 1
            print(f"[SKIPPED]   Fant ikke klasse for: {nii_file}")

    print("\nFerdig ✅")
    print(f"Healthy:   {healthy_count}")
    print(f"Unhealthy: {unhealthy_count}")
    print(f"Skipped:   {skipped_count}")


# =========================
# MAIN
# =========================

def main():
    ensure_dirs()

    print("Starter unzip av alle zip-filer...")
    extract_all_zips_recursive(SOURCE_ROOT, EXTRACT_ROOT)

    print("\nStarter sortering av .nii.gz-filer...")
    sort_nii_files(EXTRACT_ROOT)


if __name__ == "__main__":
    main()