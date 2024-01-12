#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import os

PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))
MPLSTYLE = "{}/data/lightkurve.mplstyle".format(PACKAGEDIR)

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

log = logging.getLogger(__name__)
log.addHandler(logging.StreamHandler())

from . import config as _config  # noqa: E402


class Conf(_config.ConfigNamespace):
    """
    Configuration parameters for `lightkurve`.

    Refer to `astropy.config.ConfigNamespace` for API details.

    Refer to `Astropy documentation <https://docs.astropy.org/en/stable/config/index.html#accessing-values>`_
    for usage.

    The attributes listed below are the available configuration parameters.

    Attributes
    ----------
    cache_dir
        Default cache directory for data files downloaded, etc. Defaults to ``~/.lightkurve/cache`` if not specified.
    """

    cache_dir = _config.ConfigItem(
        None,
        "Default cache directory for data files downloaded, etc.",
        cfgtype="string",
        module="lightkurve.config",
    )


conf = Conf()


from .lightcurve import *  # noqa: E402, F403
from .datacube import *  # noqa: F403, E402
from .periodogram import *  # noqa: F403, E402
