# lkdata

[![Installation](https://github.com/tessgi/secret/actions/workflows/install.yml/badge.svg?event=push)](https://github.com/tessgi/secret/actions/workflows/install.yml)
[![ruff](https://github.com/tessgi/secret/actions/workflows/check.yml/badge.svg?event=push)](https://github.com/tessgi/secret/actions/workflows/check.yml)
[![Tests](https://github.com/tessgi/secret/actions/workflows/test.yml/badge.svg?branch=main&event=push)](https://github.com/tessgi/secret/actions/workflows/test.yml)
[![Coverage](https://github.com/tessgi/secret/actions/workflows/coverage.yml/badge.svg?event=push)](https://github.com/tessgi/secret/actions/workflows/coverage.yml)
[![Coverage badge](https://github.com/tessgi/secret/raw/python-coverage-comment-action-data/badge.svg)](https://github.com/tessgi/secret/tree/python-coverage-comment-action-data)

## How to install
The easiest way:
```
git clone ...
pip install poetry --upgrade
make install
```

## What's all this, then?

The `lkdata` package has been developed to handle time-series data using the built-in power and standard functionality of [pandas](https://pandas.pydata.org/).

Users of `lightkurve` will not typically interact directly with lkdata. Rather, the lkdata package serves at the backend for `lightkurve` v3. This represents a paradigm shift towards modularity and smaller independently maintained packages composing the `lightkurve` ecosystem.

As `lightkurve` is updated to v3, `lkdata` will be used for the affiliated packages to process [Lightcurve](https://archive.stsci.edu/missions/tess/doc/EXP-TESS-ARC-ICD-TM-0014.pdf#page=32) (LC), [Target Pixel File](https://archive.stsci.edu/missions/tess/doc/EXP-TESS-ARC-ICD-TM-0014.pdf#page=24) (TPF), and [Full Frame Image](https://archive.stsci.edu/missions/tess/doc/EXP-TESS-ARC-ICD-TM-0014.pdf#page=18) (FFI) [data](https://outerspace.stsci.edu/display/TESS/2.0+-+Data+Product+Overview) for the [TESS](https://tess.mit.edu) and [Kepler/K2](https://science.nasa.gov/mission/kepler/) missions.

## What does `lkdata` do, specifically?
### Time Series Data Format
`lkdata` supports three data formats commonly dealt with in time-domain astronomy: time series, collections of  multiple time series, and data cubes.
- Time series (such as light curves) data is handled by `Series` objects, containing a time index and data.
- Collections of time series-like arrays with shared times can be handled by `SeriesCollection` objects.
- Data cube products (such as TPF or FFI time series) are handled by `Cube` objects. The difference between a collection and a `Cube` is that `Cube` objects must be contiguous along the spatial axes.

### Data types, numerical and otherwise
Data that are associated with time values can be represented by one of three different classes: DataSeries, BoolSeries, or BitwiseSeries.

**Data[Series/SeriesCollection/Cube]** represent numerical data that varies with time, such as flux, energy, or other lightcurve-like data.

**Bool[Series/SeriesCollection/Cube]** handle any Boolean data that varies by time, such as a Boolean mask for time steps that are 'good' or 'bad'.

**Bitwise[Series/SeriesCollection/Cube]** handle data that represents bitwise information. This was motivated by the [TESS](https://outerspace.stsci.edu/display/TESS/2.0+-+Data+Product+Overview#id-2.0-DataProductOverview-Table:CadenceQualityFlags), [Kepler, and K2](https://archive.stsci.edu/files/live/sites/mast/files/home/missions-and-data/k2/_documents/MAST_Kepler_Archive_Manual_2020.pdf#page=18) quality masks, which use bitwise AND operations to represent different combinations of data quality flags.

### Familiar, intuitive, efficient

`lkdata` uses `pandas` for its familiar interface and its efficient data handling, and extends it for time domain specific applications.
All lkdata products operate on data associated with temporal indices and spatial columns. For numerical data, support for associated errors has been included, as well as classes to handle boolean and bitwise data type aggregation.

A mixture of any of these types of data can be bundled in a single lkdata DataSet class, provided they all have the same times associated with them. The advantage of the DataSet is that operations applied to the DataSet, such as binning or slicing, are applied to all associated Series, Frame, and Cube objects as appropriate. For example, you could bundle a group of DataSeries that each represent a single source of flux (simple aperture flux, corrected flux, background flux, etc) and simulataneously cutout a time range of interest. We will see more applications for DataCubes in the examples throughout this notebook.

## Lightkurve; now more easy to develop

Developing for lightkurve now includes some new features:

- pre-commit hooks. This is going to stop anyone from commiting anything to their branch that breaks our standards. That means linted, formatted code, well written markdown files, checking with mypy etc. This means it's going to be harder (but not impossible) for you to open a PR against lightkurve that doesn't have these things fixed. This is much more strict than V2.
<!-- - docs that compile better. We're making documentation that is easier to compile and upload by using a sphinx gallery.-->