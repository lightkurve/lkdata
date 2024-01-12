"""Define TESS quality flags"""
from ..quality import QualityFlags


class TESSQualityFlags(QualityFlags):
    """
    This class encodes the meaning of the various TESS QUALITY bitmask flags,
    as documented in the TESS Data Products Description Document (Ref. [1], Table 28).

    References
    ----------
    .. [1] TESS Science Data Products Description Document (EXP-TESS-ARC-ICD-0014)
        https://archive.stsci.edu/missions/tess/doc/EXP-TESS-ARC-ICD-TM-0014.pdf
    """

    AttitudeTweak = 1
    SafeMode = 2
    CoarsePoint = 4
    EarthPoint = 8
    Argabrightening = 16
    Desat = 32
    ApertureCosmic = 64
    ManualExclude = 128
    Discontinuity = 256
    ImpulsiveOutlier = 512
    CollateralCosmic = 1024
    #: The first stray light flag is set manually by MIT based on visual inspection.
    Straylight = 2048
    #: The second stray light flag is set automatically by Ames/SPOC based on background level thresholds.
    Straylight2 = 4096
    # See TESS Science Data Products Description Document
    PlanetSearchExclude = 8192
    BadCalibrationExclude = 16384
    # Set in the sector 20 data release notes
    InsufficientTargets = 32768

    #: DEFAULT bitmask identifies all cadences which are definitely useless.
    DEFAULT_BITMASK = (
        AttitudeTweak | SafeMode | CoarsePoint | EarthPoint | Desat | ManualExclude
    )
    #: HARD bitmask is conservative and may identify cadences which are useful.
    HARD_BITMASK = (
        DEFAULT_BITMASK | ApertureCosmic | CollateralCosmic | Straylight | Straylight2
    )
    #: HARDEST bitmask identifies cadences with any flag set. Its use is not recommended.
    HARDEST_BITMASK = 65535

    #: Dictionary which provides friendly names for the various bitmasks.
    OPTIONS = {
        "none": 0,
        "default": DEFAULT_BITMASK,
        "hard": HARD_BITMASK,
        "hardest": HARDEST_BITMASK,
    }

    #: Pretty string descriptions for each flag
    STRINGS = {
        1: "Attitude tweak",
        2: "Safe mode",
        4: "Coarse point",
        8: "Earth point",
        16: "Argabrightening",
        32: "Desaturation event",
        64: "Cosmic ray in optimal aperture",
        128: "Manual exclude",
        256: "Discontinuity corrected",
        512: "Impulsive outlier",
        1024: "Cosmic ray in collateral data",
        2048: "Straylight",
        4096: "Straylight2",
        8192: "Planet Search Exclude",
        16384: "Bad Calibration Exclude",
        32768: "Insufficient Targets for Error Correction Exclude",
    }
