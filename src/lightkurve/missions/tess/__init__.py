"""Methods that are specific to TESS"""

from .time import *  # noqa: F403
from .quality import *  # noqa: F403

import astropy.units as u

# Define some constants here? e.g.?
pixel_scale = 21 * u.arcsecond

# Maybe we need to define some term which is "segment"?
# Different missions have different nomenclature for a segment, so it's worth writing that down?

SEGMENT_NAME = "Sector"
CATALOG_NAME = "TIC"
INSTRUMENT_NAME = "TESS"
MISSION_NAME = "TESS"
