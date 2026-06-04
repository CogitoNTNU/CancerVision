Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

function Get-EnvOrDefault {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Default
    )

    $value = [Environment]::GetEnvironmentVariable($Name)
    if ([string]::IsNullOrWhiteSpace($value)) {
        return $Default
    }
    return $value
}

function Test-Command {
    param([Parameter(Mandatory = $true)][string]$Name)
    return $null -ne (Get-Command $Name -ErrorAction SilentlyContinue)
}

if (Test-Command "uv") {
    $script:PythonRunner = "uv"
}
elseif (Test-Command "python") {
    $script:PythonRunner = "python"
}
elseif (Test-Command "py") {
    $script:PythonRunner = "py"
}
else {
    throw "Missing uv/python/py in PATH."
}

function Invoke-Python {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Args)

    if ($script:PythonRunner -eq "uv") {
        & uv run python @Args
    }
    elseif ($script:PythonRunner -eq "python") {
        & python @Args
    }
    else {
        & py -3 @Args
    }

    if ($LASTEXITCODE -ne 0) {
        throw ("Command failed with exit code {0}: python {1}" -f $LASTEXITCODE, ($Args -join ' '))
    }
}

$standardizedRoot = Get-EnvOrDefault "STANDARDIZED_ROOT" "Z:\dataset\cancervision-standardized"
$manifestDir = Get-EnvOrDefault "MANIFEST_DIR" (Join-Path $standardizedRoot "manifests")
$taskDir = Get-EnvOrDefault "TASK_DIR" (Join-Path $standardizedRoot "task_manifests")
$segNativeDir = Get-EnvOrDefault "SEG_NATIVE_DIR" (Join-Path $standardizedRoot "segmentation_native")

$mergedSourceManifest = Get-EnvOrDefault "MERGED_SOURCE_MANIFEST" (Join-Path $manifestDir "all_sources_merged.csv")
$segTaskManifest = Get-EnvOrDefault "SEG_TASK_MANIFEST" (Join-Path $taskDir "segmentation_binary_curated.csv")
$segMaterializedManifest = Get-EnvOrDefault "SEG_MATERIALIZED_MANIFEST" (Join-Path $segNativeDir "segmentation_materialized_manifest.csv")

$brats2020Root = Get-EnvOrDefault "BRATS2020_ROOT" "Z:\dataset\brats2020"
$brats2023Root = Get-EnvOrDefault "BRATS2023_ROOT" "Z:\dataset\brats2023\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
$brats2024Root = Get-EnvOrDefault "BRATS2024_ROOT" "Z:\dataset\brats2024\BraTS2024_small_dataset"
$cfbGbmRoot = Get-EnvOrDefault "CFB_GBM_ROOT" "Z:\dataset\PKG - CFB-GBM version 1\CFB-GBM"
$upennGbmRoot = Get-EnvOrDefault "UPENN_GBM_ROOT" "Z:\dataset\PKG - UPENN-GBM-NIfTI"
$ucsfPdgmRoot = Get-EnvOrDefault "UCSF_PDGM_ROOT" "Z:\dataset\UCSF-PDGM-v5"
$ucsdPtgbmRoot = Get-EnvOrDefault "UCSD_PTGBM_ROOT" "Z:\dataset\UCSD-PTGBM"
$utswGliomaRoot = Get-EnvOrDefault "UTSW_GLIOMA_ROOT" "Z:\dataset\UTSW-Glioma"
$remindRoot = Get-EnvOrDefault "REMIND_ROOT" "Z:\dataset\remind"
$remindMaskRoot = Get-EnvOrDefault "REMIND_MASK_ROOT" "Z:\dataset\PKG - ReMIND_NRRD_Seg_Sep_2023\ReMIND_NRRD_Seg_Sep_2023"
$yaleRoot = Get-EnvOrDefault "YALE_ROOT" "Z:\dataset\PKG - Yale-Brain-Mets-Longitudinal\Yale-Brain-Mets-Longitudinal"
$vestibularRoot = Get-EnvOrDefault "VESTIBULAR_ROOT" "Z:\dataset\Vestibular-Schwannoma-MC-RC2_Oct2025"

$defaultSynthstripCmd = if ($script:PythonRunner -eq "uv") {
    "uv run python scripts/synthstrip_docker.py"
}
elseif ($script:PythonRunner -eq "python") {
    "python scripts/synthstrip_docker.py"
}
else {
    "py -3 scripts/synthstrip_docker.py"
}

$synthstripCmd = Get-EnvOrDefault "SYNTHSTRIP_CMD" $defaultSynthstripCmd
$forceRebuild = Get-EnvOrDefault "FORCE_REBUILD" "0"

New-Item -ItemType Directory -Force -Path $manifestDir, $taskDir, $segNativeDir | Out-Null

if ($forceRebuild -eq "1") {
    Write-Host "FORCE_REBUILD=1 -> removing old merged/task/materialized outputs"
    foreach ($path in @($mergedSourceManifest, $segTaskManifest, $segMaterializedManifest, $segNativeDir)) {
        if (Test-Path -LiteralPath $path) {
            Remove-Item -LiteralPath $path -Recurse -Force
        }
    }
    New-Item -ItemType Directory -Force -Path $segNativeDir | Out-Null
}

$builtManifests = [System.Collections.Generic.List[string]]::new()

function Run-Build {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $false)][string]$RootPath,
        [Parameter(Mandatory = $true)][string]$OutputCsv,
        [Parameter(Mandatory = $true)][string[]]$CommandArgs
    )

    if ([string]::IsNullOrWhiteSpace($RootPath)) {
        Write-Host "skip $Name`: root empty"
        return
    }
    if (-not (Test-Path -LiteralPath $RootPath)) {
        Write-Host "skip $Name`: root missing -> $RootPath"
        return
    }
    if (($forceRebuild -ne "1") -and (Test-Path -LiteralPath $OutputCsv)) {
        Write-Host "reuse $Name`: $OutputCsv"
        $builtManifests.Add($OutputCsv) | Out-Null
        return
    }

    Write-Host "build $Name`: $RootPath"
    Invoke-Python main.py standardize @CommandArgs --output-csv $OutputCsv --include-excluded
    $builtManifests.Add($OutputCsv) | Out-Null
}

Run-Build "brats2020" $brats2020Root (Join-Path $manifestDir "brats2020_source_manifest.csv") @(
    "build-brats2020-manifest",
    "--brats2020-root", $brats2020Root
)

Run-Build "brats2023" $brats2023Root (Join-Path $manifestDir "brats2023_source_manifest.csv") @(
    "build-brats2023-manifest",
    "--brats2023-root", $brats2023Root
)

Run-Build "brats2024" $brats2024Root (Join-Path $manifestDir "brats2024_source_manifest.csv") @(
    "build-brats2024-manifest",
    "--brats2024-root", $brats2024Root
)

Run-Build "cfb_gbm" $cfbGbmRoot (Join-Path $manifestDir "cfb_gbm_source_manifest.csv") @(
    "build-cfb-gbm-manifest",
    "--cfb-gbm-root", $cfbGbmRoot
)

Run-Build "upenn_gbm" $upennGbmRoot (Join-Path $manifestDir "upenn_gbm_source_manifest.csv") @(
    "build-upenn-gbm-manifest",
    "--upenn-gbm-root", $upennGbmRoot
)

Run-Build "ucsf_pdgm" $ucsfPdgmRoot (Join-Path $manifestDir "ucsf_pdgm_source_manifest.csv") @(
    "build-ucsf-pdgm-manifest",
    "--ucsf-pdgm-root", $ucsfPdgmRoot
)

Run-Build "ucsd_ptgbm" $ucsdPtgbmRoot (Join-Path $manifestDir "ucsd_ptgbm_source_manifest.csv") @(
    "build-ucsd-ptgbm-manifest",
    "--ucsd-ptgbm-root", $ucsdPtgbmRoot
)

Run-Build "utsw_glioma" $utswGliomaRoot (Join-Path $manifestDir "utsw_glioma_source_manifest.csv") @(
    "build-utsw-glioma-manifest",
    "--utsw-glioma-root", $utswGliomaRoot
)

Run-Build "remind" $remindRoot (Join-Path $manifestDir "remind_source_manifest.csv") @(
    "build-remind-manifest",
    "--remind-root", $remindRoot,
    "--remind-mask-root", $remindMaskRoot
)

Run-Build "yale_brain_mets_longitudinal" $yaleRoot (Join-Path $manifestDir "yale_brain_mets_longitudinal_source_manifest.csv") @(
    "build-yale-brain-mets-longitudinal-manifest",
    "--yale-root", $yaleRoot
)

Run-Build "vestibular_schwannoma_mc_rc2" $vestibularRoot (Join-Path $manifestDir "vestibular_schwannoma_mc_rc2_source_manifest.csv") @(
    "build-vestibular-schwannoma-mc-rc2-manifest",
    "--vestibular-root", $vestibularRoot
)

if ($builtManifests.Count -eq 0) {
    throw "No source manifests built. Set dataset root env vars first."
}

Write-Host "merge source manifests -> $mergedSourceManifest"
Invoke-Python scripts/merge_standardized_manifests.py --output $mergedSourceManifest @builtManifests

Write-Host "build seg task manifest -> $taskDir"
Invoke-Python main.py standardize build-task-manifests --input-manifest $mergedSourceManifest --output-dir $taskDir

Write-Host "materialize seg images -> $segNativeDir"
Invoke-Python main.py standardize materialize-segmentation-native --input-manifest $segTaskManifest --output-dir $segNativeDir --output-manifest $segMaterializedManifest --synthstrip-cmd $synthstripCmd

Write-Host ""
Write-Host "done"
Write-Host "merged source manifest: $mergedSourceManifest"
Write-Host "seg task manifest: $segTaskManifest"
Write-Host "seg materialized manifest: $segMaterializedManifest"
