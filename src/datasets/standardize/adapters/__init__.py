"""Dataset adapters for standardization."""

from .brats import BraTSAdapter, BraTSDatasetSpec
from .cfb_gbm import CfbGbmAdapter, parse_cfb_series_path
from .brain_structure import BrainStructureAdapter, map_brain_structure_diagnosis
from .remind import (
    RemindAdapter,
    is_remind_real_t1_description,
    parse_remind_mask_filename,
)
from .ucsd_ptgbm import (
    UcsdPtgbmAdapter,
    parse_ucsd_case_dir_name,
    parse_ucsd_series_name,
)
from .ucsf_pdgm import UcsfPdgmAdapter, parse_ucsf_case_dir_name, parse_ucsf_series_name
from .upenn_gbm import UpennGbmAdapter, parse_upenn_series_path
from .utsw_glioma import UtswGliomaAdapter
from .vestibular_schwannoma_mc_rc2 import (
    VestibularSchwannomaMcRc2Adapter,
    parse_vestibular_series_path,
)
from .yale_brain_mets_longitudinal import (
    YaleBrainMetsLongitudinalAdapter,
    parse_yale_series_path,
)

__all__ = [
    "BraTSAdapter",
    "BraTSDatasetSpec",
    "CfbGbmAdapter",
    "BrainStructureAdapter",
    "RemindAdapter",
    "UcsdPtgbmAdapter",
    "UcsfPdgmAdapter",
    "UpennGbmAdapter",
    "UtswGliomaAdapter",
    "VestibularSchwannomaMcRc2Adapter",
    "YaleBrainMetsLongitudinalAdapter",
    "is_remind_real_t1_description",
    "map_brain_structure_diagnosis",
    "parse_cfb_series_path",
    "parse_remind_mask_filename",
    "parse_ucsd_case_dir_name",
    "parse_ucsd_series_name",
    "parse_ucsf_case_dir_name",
    "parse_ucsf_series_name",
    "parse_upenn_series_path",
    "parse_vestibular_series_path",
    "parse_yale_series_path",
]
