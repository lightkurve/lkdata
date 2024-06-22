#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import os

PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))
MPLSTYLE = "{}/data/lightkurve.mplstyle".format(PACKAGEDIR)
ROOTNAME = "lightkurve"
TESTDATA = f"{PACKAGEDIR}/data/tess-s0001-4-2_84.291190_-80.469170_6x6_astrocut.fits"

# Bibtex entry detailing how to cite the package
__citation__ = """@MISC{2018ascl.soft12013L,
    author = {{Lightkurve Collaboration} and {Cardoso}, J.~V.~d.~M. and
                {Hedges}, C. and {Gully-Santiago}, M. and {Saunders}, N. and
                {Cody}, A.~M. and {Barclay}, T. and {Hall}, O. and
                {Sagear}, S. and {Turtelboom}, E. and {Zhang}, J. and
                {Tzanidakis}, A. and {Mighell}, K. and {Coughlin}, J. and
                {Bell}, K. and {Berta-Thompson}, Z. and {Williams}, P. and
                {Dotson}, J. and {Barentsen}, G.},
    title = "{Lightkurve: Kepler and TESS time series analysis in Python}",
    keywords = {Software, NASA},
howpublished = {Astrophysics Source Code Library},
        year = 2018,
    month = dec,
archivePrefix = "ascl",
    eprint = {1812.013},
    adsurl = {http://adsabs.harvard.edu/abs/2018ascl.soft12013L},
}"""


import logging  # noqa: E402
from rich.logging import RichHandler  # noqa: E402


def get_logger():
    """Configure and return a logger with RichHandler."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.WARN)

    # Add RichHandler
    rich_handler = RichHandler()
    rich_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )

    logger.addHandler(rich_handler)
    return logger


from .version import __version__  # noqa: E402, F401
from .datacube import *  # noqa: F403, E402
from .dataframe import *  # noqa: F403, E402
from .dataseries import *  # noqa: F403, E402
from .periodogram import *  # noqa: F403, E402
from .time import *  # noqa: F403, E402
from .meta import *  # noqa: F403, E402
