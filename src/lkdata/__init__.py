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

PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))
MPLSTYLE = f"{PACKAGEDIR}/data/lightkurve.mplstyle"
ROOTNAME = "lkdata"
TESTDATA = f"{PACKAGEDIR}/data/tess-s0001-4-2_84.291190_-80.469170_6x6_astrocut.fits"
