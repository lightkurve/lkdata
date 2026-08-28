# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `units` property on `DataSeries`, `DataSeriesCollection`, and `DataCube`; units can be passed via a `units` kwarg or inherited automatically from astropy `Quantity` objects. Multiplying a series or cube by an astropy unit now returns a copy with units set. (#44)

### Changed

- `DataCube.__getitem__` / `__setitem__` overhauled: replaced `singledispatchmethod` dispatch with a unified method and a new `_get_iloc_key` helper; added string key support and made get/set behaviour consistent. (#45)
- `DataSet.__getitem__` now accepts any `Iterable` for time-index selection rather than requiring `list` or `np.ndarray` specifically. (#41)
- `Cube` dimension attributes (`nrow`, `ncol`, `row_names`, `col_names`) refactored to private backing fields (`_nrow`, `_ncol`, etc.); added `_stats_post_process` method; removed internal `_set_dim` helper. (#43)
- Type hints updated across `DataSeries` and `BitwiseSeries` to resolve Pylance errors; `DataSeries.array` property now returns the internal `_array` directly; `_user_kwargs` initialisation guarded against `None`. (#42)
- Substantially expanded test coverage for `DataCube`, `DataSet`, `DataSeries`, `BitSet`, mixins, and uncertainty. (#46)
- CI: bumped `peaceiris/actions-gh-pages` from v3 to v4. (#37)

### Fixed

- `median` method was consuming the `axis` keyword argument before passing it to `np.median`, producing incorrect results. (#46)
- `DataSet.describe_set` raised `ValueError` when the dataset had no user-defined attributes (`max_name_len` was computed before the early-return guard).
- Index creation for batch functions on `DataSet`: switched from `==` to `.equals()` for index comparison, added NaN-safe column matching, and moved `_include_convenience_index()` to `__setitem__` so it fires at the right time.
- `SeriesCollection` treated `nrow`/`ncol` values of `None` as falsy, causing incorrect zero defaults.
- `DataSeriesCollection.__repr__` was displaying `SeriesCollection` instead of `DataSeriesCollection`.
- Replaced deprecated `np.in1d` with `np.isin` in `Series` index matching. (#40)


## [1.0.1] - 2026-04-16

Initial public release.
