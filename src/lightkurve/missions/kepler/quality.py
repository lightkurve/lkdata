"""Defines Kepler quality flags"""
from ..quality import QualityFlags


class KeplerQualityFlags(QualityFlags):
    """
    This class encodes the meaning of the various Kepler QUALITY bitmask flags,
    as documented in the Kepler Archive Manual (Ref. [1], Table 2.3).

    References
    ----------
    .. [1] Kepler: A Search for Terrestrial Planets. Kepler Archive Manual.
        http://archive.stsci.edu/kepler/manuals/archive_manual.pdf
    """

    AttitudeTweak = 1
    SafeMode = 2
    CoarsePoint = 4
    EarthPoint = 8
    ZeroCrossing = 16
    Desat = 32
    Argabrightening = 64
    ApertureCosmic = 128
    ManualExclude = 256
    # Bit 2**10 = 512 is unused by Kepler
    SensitivityDropout = 1024
    ImpulsiveOutlier = 2048
    ArgabrighteningOnCCD = 4096
    CollateralCosmic = 8192
    DetectorAnomaly = 16384
    NoFinePoint = 32768
    NoData = 65536
    RollingBandInAperture = 131072
    RollingBandInMask = 262144
    PossibleThrusterFiring = 524288
    ThrusterFiring = 1048576

    #: DEFAULT bitmask identifies all cadences which are definitely useless.
    DEFAULT_BITMASK = (
        AttitudeTweak
        | SafeMode
        | CoarsePoint
        | EarthPoint
        | Desat
        | ManualExclude
        | DetectorAnomaly
        | NoData
        | ThrusterFiring
    )
    #: HARD bitmask is conservative and may identify cadences which are useful.
    HARD_BITMASK = (
        DEFAULT_BITMASK
        | SensitivityDropout
        | ApertureCosmic
        | CollateralCosmic
        | PossibleThrusterFiring
    )
    #: HARDEST bitmask identifies cadences with any flag set. Its use is not recommended.
    HARDEST_BITMASK = 2096639

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
        16: "Zero crossing",
        32: "Desaturation event",
        64: "Argabrightening",
        128: "Cosmic ray in optimal aperture",
        256: "Manual exclude",
        1024: "Sudden sensitivity dropout",
        2048: "Impulsive outlier",
        4096: "Argabrightening on CCD",
        8192: "Cosmic ray in collateral data",
        16384: "Detector anomaly",
        32768: "No fine point",
        65536: "No data",
        131072: "Rolling band in optimal aperture",
        262144: "Rolling band in full mask",
        524288: "Possible thruster firing",
        1048576: "Thruster firing",
    }
