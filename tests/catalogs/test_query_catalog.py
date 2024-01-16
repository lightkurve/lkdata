import pytest

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.table import Table

from lightkurve.catalogs.query import query_skycatalog


@pytest.mark.remote_data
def test_query_tic():
    # Tests the region around TIC 228760807 which should return a catalog containing 4 objects.
    c = SkyCoord(194.10141041659, -27.3905828803397, unit="deg")
    epoch = Time(1569.4424277786259, scale="tdb", format="btjd")

    catalog = query_skycatalog(c, epoch, "tic", 80, 18)
    assert len(catalog["ID"]) == 4

    # Checks that an astropy Table is returned
    assert isinstance(catalog, Table)

    # Test that the proper motion works
    correct_ra = 194.10075230969787
    correct_dec = -27.390340343480744

    assert np.isclose(catalog["RA"][0], correct_ra, atol=1e-6)
    assert np.isclose(catalog["DEC"][0], correct_dec, atol=1e-6)

    # Test different epochs
    catalog_new = query_skycatalog(
        c, Time(2461041.500, scale="tt", format="jd"), "tic", 80, 18
    )

    correct_ra_new = 194.10052070792756
    correct_dec_new = -27.390254988629433

    assert np.isclose(catalog_new["RA"][0], correct_ra_new, atol=1e-6)
    assert np.isclose(catalog_new["DEC"][0], correct_dec_new, atol=1e-6)


@pytest.mark.remote_data
def test_bad_catalog():
    # test the catalog type i.e., simbad is not included in our catalog list.
    # Look at other tests to see if this is correct syntax
    c = SkyCoord(194.10141041659, -27.3905828803397, unit="deg")
    epoch = Time(1569.4424277786259, scale="tdb", format="btjd")

    with pytest.raises(ValueError, match="Can not parse catalog name 'badcat'"):
        query_skycatalog(c, epoch, "badcat", 80, 18)


@pytest.mark.remote_data
def test_query_gaia():
    # Test each other catalog
    # Gaia
    catalog_gaia = query_skycatalog(
        SkyCoord(194.10141041659, -27.3905828803397, unit="deg"),
        Time(1569.4424277786259, scale="tdb", format="btjd"),
        "gaiadr3",
        80,
        18,
    )

    assert len(catalog_gaia["ID"]) == 2


@pytest.mark.remote_data
def test_query_kic():
    # Kepler
    catalog_kepler = query_skycatalog(
        SkyCoord(285.679391, 50.2413, unit="deg"),
        Time(120.5391465105713, scale="tdb", format="bkjd"),
        "kic",
        20,
        18,
    )

    assert len(catalog_kepler["ID"]) == 5


@pytest.mark.remote_data
def test_query_epic():
    # K2
    catalog_k2 = query_skycatalog(
        SkyCoord(172.560465, 7.588391, unit="deg"),
        Time(1975.1781333280233, scale="tdb", format="bkjd"),
        "epic",
        20,
        18,
    )

    assert len(catalog_k2["ID"]) == 1
