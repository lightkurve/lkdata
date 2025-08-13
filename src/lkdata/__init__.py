#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from .version import __version__  # noqa: E402, F401
from .datacube import DataCube, BoolCube, BitwiseCube  # noqa: F403, E402
from .seriescollection import (
    DataSeriesCollection,
    BoolSeriesCollection,
    BitwiseSeriesCollection,
)  # noqa: F403, E402
from .dataseries import DataSeries, BoolSeries, BitwiseSeries  # noqa: F403, E402
from .dataset import DataSet  # noqa: E402
import os
from typing import Union

__all__ = [
    "DataCube",
    "DataSeriesCollection",
    "DataSeries",
    "DataSet",
    "BoolCube",
    "BoolSeriesCollection",
    "BoolSeries",
    "BitwiseCube",
    "BitwiseSeriesCollection",
    "BitwiseSeries",
]

LkDataTypes = Union[DataCube, DataSeriesCollection, DataSeries]
LkBoolTypes = Union[BoolCube, BoolSeriesCollection, BoolSeries]
LkBitwiseTypes = Union[BitwiseCube, BitwiseSeriesCollection, BitwiseSeries]
LkTypes = Union[LkDataTypes, LkBoolTypes, LkBitwiseTypes]

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
