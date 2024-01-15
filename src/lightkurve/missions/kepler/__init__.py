"""Methods that are specific to Kepler"""

from .time import *  # noqa: F403
from .quality import *  # noqa: F403
from .utils import *  # noqa: F403

import astropy.units as u

# Define some constants here? e.g.?
pixel_scale = 4 * u.arcsecond

# Maybe we need to define some term which is "segment"?
# Different missions have different nomenclature for a segment, so it's worth writing that down?

SEGMENT_NAME = "quarter"
CATALOG_NAME = "KIC"
INSTRUMENT_NAME = "Kepler"
MISSION_NAME = "Kepler"
NESTING_NAMES = ["quarter", "channel"]
