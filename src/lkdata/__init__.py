#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from .version import __version__  # noqa: E402, F401
from .datacube import DataCube, ErrorCube, BoolCube, BitwiseCube  # noqa: F403, E402
from .dataframe import DataFrame, ErrorFrame, BoolFrame, BitwiseFrame  # noqa: F403, E402
from .dataseries import DataSeries, ErrorSeries, BoolSeries, BitwiseSeries  # noqa: F403, E402
from .dataset import DataSet  # noqa: E402
import os

__all__ = [
    "DataCube",
    "ErrorCube",
    "DataFrame",
    "ErrorFrame",
    "DataSeries",
    "ErrorSeries",
    "DataSet",
    "BoolCube",
    "BoolFrame",
    "BoolSeries",
    "BitwiseCube",
    "BitwiseFrame",
    "BitwiseSeries",
]

PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))
MPLSTYLE = f"{PACKAGEDIR}/data/lightkurve.mplstyle"
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
