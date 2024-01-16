"""Tools to query catalogs"""
import warnings
import numpy as np
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, Angle, Distance
from astroquery.vizier import Vizier
from astropy.table import Table
from typing import Union

from .utils import CATALOG_DICTIONARY

from .. import get_logger

logger = get_logger()

__all__ = ["query_KIC", "query_TIC", "query_EPIC", "query_Gaia", "query_skycatalog"]


def query_KIC(
    coord: SkyCoord,
    epoch: Time,
    radius: Union[float, u.Quantity] = u.Quantity(40, "arcsecond"),
    magnitude_limit: float = 18.0,
    equinox: Time = Time(2000, format="jyear", scale="tt"),
) -> Table:
    """Query the Kepler Input Catalog

    Parameters:
    -----------
    coord : astropy.coordinates.SkyCoord
        Coordinates around which to do a radius query
    epoch: astropy.time.Time
        The time of observation in JD and TT. Note that Kepler data is in bkjd & tdb - so a user would have to specify in the Time object
        For example you could put in `Time(np.mean(lc.time.value), scale='tdb', format='bkjd')`
    radius : float
        Radius in arcseconds to query
    magnitude_limit : float
        A value to limit the results in based on the Kepler mag. Default, 18.
    equinox: astropy.time.Time
        The R.A and Dec. values taken from the catalogs is in J2000.
        The J2000. 0 epoch is precisely the Julian year 2000 in terrestrial time (tt).

    Returns:
    -------
    result : astropy.table.Table
        Returns an astropy.table of the sources within radius query corrected for proper motion
    """
    logger.debug(f"Querying KIC at {coord}")
    return query_skycatalog(
        epoch=epoch,
        coord=coord,
        catalog="kic",
        radius=radius,
        magnitude_limit=magnitude_limit,
        equinox=equinox,
    )


def query_TIC(
    coord: SkyCoord,
    epoch: Time,
    radius: Union[float, u.Quantity] = u.Quantity(21 * 5, "arcsecond"),
    magnitude_limit: float = 16.0,
    equinox: Time = Time(2000, format="jyear", scale="tt"),
) -> Table:
    """Query the TESS Input Catalog

    Parameters:
    -----------
    coord : astropy.coordinates.SkyCoord
        Coordinates around which to do a radius query
    epoch: astropy.time.Time
        The time of observation in JD and TT. Note that TESS data is in btjd & tdb - so a user would have to specify in the Time object
        For example you could put in `Time(np.mean(lc.time.value), scale='tdb', format='btjd')`
    radius : float
        Radius in arcseconds to query
    magnitude_limit : float
        A value to limit the results in based on the Tmag mag. Default, 16.
    equinox: astropy.time.Time
        The R.A and Dec. values taken from the catalogs is in J2000.
        The J2000. 0 epoch is precisely the Julian year 2000 in terrestrial time (tt).

    Returns:
    -------
    result : astropy.table.Table
        Returns an astropy.table of the sources within radius query corrected for proper motion
    """
    logger.debug(f"Querying TIC at {coord}")
    return query_skycatalog(
        epoch=epoch,
        coord=coord,
        catalog="tic",
        radius=radius,
        magnitude_limit=magnitude_limit,
        equinox=equinox,
    )


def query_EPIC(
    coord: SkyCoord,
    epoch: Time,
    radius: Union[float, u.Quantity] = u.Quantity(40, "arcsecond"),
    magnitude_limit: float = 18.0,
    equinox: Time = Time(2000, format="jyear", scale="tt"),
) -> Table:
    """Query the Ecliptic Plane Input Catalog

    Parameters:
    -----------
    coord : astropy.coordinates.SkyCoord
        Coordinates around which to do a radius query
    epoch: astropy.time.Time
        The time of observation in JD and TT. Note that K2 data is in bkjd & tdb - so a user would have to specify in the Time object
        For example you could put in `Time(np.mean(lc.time.value), scale='tdb', format='bkjd')`
    radius : float
        Radius in arcseconds to query
    magnitude_limit : float
        A value to limit the results in based on the K2 mag. Default, 18.
    equinox: astropy.time.Time
        The R.A and Dec. values taken from the catalogs is in J2000.
        The J2000. 0 epoch is precisely the Julian year 2000 in terrestrial time (tt).

    Returns:
    -------
    result : astropy.table.Table
        Returns an astropy.table of the sources within radius query corrected for proper motion
    """
    logger.debug(f"Querying EPIC at {coord}")
    return query_skycatalog(
        epoch=epoch,
        coord=coord,
        catalog="epic",
        radius=radius,
        magnitude_limit=magnitude_limit,
        equinox=equinox,
    )


def query_Gaia(
    coord: SkyCoord,
    epoch: Time,
    radius: Union[float, u.Quantity] = u.Quantity(10, "arcsecond"),
    magnitude_limit: float = 21.0,
    equinox: Time = Time(2000, format="jyear", scale="tt"),
) -> Table:
    """Query the Ecliptic Plane Input Catalog

    Parameters:
    -----------
    coord : astropy.coordinates.SkyCoord
        Coordinates around which to do a radius query
    epoch: astropy.time.Time
        The time of observation in JD and TT. A user needs to specify the relativistic coordinate time scale in the Time object
        For example you could put in `Time(np.mean(lc.time.value), scale='tdb', format='jd')`
    radius : float
        Radius in arcseconds to query
    magnitude_limit : float
        A value to limit the results in based on the Gaia G mag. Default, 21.
    equinox: astropy.time.Time
        The R.A and Dec. values taken from the catalogs is in J2000.
        The J2000. 0 epoch is precisely the Julian year 2000 in terrestrial time (tt).

    Returns:
    -------
    result : astropy.table.Table
        Returns an astropy.table of the sources within radius query corrected for proper motion
    """
    logger.debug(f"Querying GaiaDR3 at {coord}")
    return query_skycatalog(
        epoch=epoch,
        coord=coord,
        catalog="gaiadr3",
        radius=radius,
        magnitude_limit=magnitude_limit,
        equinox=equinox,
    )


def query_skycatalog(
    coord: SkyCoord,
    epoch: Time,
    catalog: str,
    radius: Union[float, u.Quantity] = u.Quantity(100, "arcsecond"),
    magnitude_limit: float = 18.0,
    equinox: Time = Time(2000, format="jyear", scale="tt"),
) -> Table:
    """Function that returns an astropy table of sources in the region of interest

    Parameters:
    -----------
    coord : astropy.coordinates.SkyCoord
        Coordinates around which to do a radius query
    epoch: astropy.time.Time
        The time of observation in JD and TT.  A user needs to specify the relativistic coordinate time scale in the Time object
        For example for TESS  you could put in `Time(np.mean(lc.time.value), scale='tdb', format='btjd')`
    catalog: str
        The catalog to query, either 'kepler', 'k2', or 'tess', 'gaia'
    radius : float
        Radius in arcseconds to query
    magnitude_limit : float
        A value to limit the results in based on the Tmag/Kepler mag/K2 mag or Gaia G mag. Default, 18.
    equinox: astropy.time.Time
        The R.A and Dec. values taken from the catalogs is in J2000.
        The J2000. 0 epoch is precisely the Julian year 2000 in terrestrial time (tt).

    Returns:
    -------
    result : astropy.table.Table
        Returns an astropy.table of the sources within radius query corrected for proper motion
    """

    # Check to make sure that user input is in the correct format
    if not isinstance(coord, SkyCoord):
        raise TypeError("Must pass an `astropy.coordinates.SkyCoord` object.")
    if not isinstance(epoch, Time):
        raise TypeError("Must pass an `astropy.time.Time object`.")
    if not isinstance(equinox, Time):
        raise TypeError("Must pass an `astropy.time.Time object`.")

    # Here we check to make sure that the radius entered is in arcseconds
    # This also means we do not need to specify arcseconds in our catalog query
    try:
        radius = u.Quantity(radius, "arcsecond")
    except u.UnitConversionError:
        raise

    # Check to make sure that the catalog provided by the user is valid for this function
    if catalog.lower() not in CATALOG_DICTIONARY.keys():
        raise ValueError(f"Can not parse catalog name '{catalog}'")

    # Get the Vizier catalog name
    catalog_name = CATALOG_DICTIONARY[catalog.lower()]["catalog"]

    # Get the appropriate column names and filters to be applied
    filters = Vizier(
        columns=CATALOG_DICTIONARY[catalog.lower()]["columns"],
        column_filters={
            CATALOG_DICTIONARY[catalog.lower()]["column_filters"]: f"<{magnitude_limit}"
        },
    )

    # The catalog can cut off at 50 - we dont want this to happen
    filters.ROW_LIMIT = -1
    # Now query the catalog
    result = filters.query_region(coord, catalog=catalog_name, radius=Angle(radius))[
        catalog_name
    ]

    # Rename the columns so that the output is uniform
    result.rename_columns(
        CATALOG_DICTIONARY[catalog.lower()]["rename_in"],
        CATALOG_DICTIONARY[catalog.lower()]["rename_out"],
    )

    # Based on the input coordinates pick the object with the mininmum separation as the reference star.

    c = SkyCoord(result["RAJ2000"], result["DEJ2000"], unit="deg")

    sep = coord.separation(c)

    # Find the object with the minimum separation - this is our target
    ref_index = np.argmin(sep)

    # apply_propermotion
    result = _apply_propermotion(result, equinox=equinox, epoch=epoch)

    # Now we want to repete but using the values corrected for proper motion
    # First get the correct values for target
    coord_pm_correct = SkyCoord(
        result["RA"][ref_index], result["DEC"][ref_index], unit="deg"
    )

    c_motion = SkyCoord(result["RA"], result["DEC"], unit="deg")

    # Then calculate the separation based on pm corrected values
    sep_pm_correct = coord_pm_correct.separation(c_motion)

    # Provide the separation in the output table
    result["Separation"] = sep_pm_correct

    # Calculate the relative flux
    result["Relative_Flux"] = np.exp(
        (
            [value for key, value in result.items() if "_Mag" in key][0]
            - [value for key, value in result.items() if "_Mag" in key][0][ref_index]
        )
        / -2.5
    )

    # Now sort the table based on separation
    result.sort(["Separation"])

    return result


def _apply_propermotion(table: Table, equinox: Time, epoch: Time):
    """
    Hidden function that returns an astropy table of sources with the proper motion correction applied

    Parameters:
    -----------
    table : astropy.table.Table
        Table which contains the coordinates of targets and proper motion values
    equinox: astropy.time.Time
        The R.A and Dec. values taken from the catalogs is in J2000.
        The J2000. 0 epoch is precisely the Julian year 2000 in terrestrial time (tt).
    epoch : astropy.time.Time
        Time of the observation - This is taken from the table R.A and Dec. values and re-formatted as an astropy.time.Time object

    Returns:
    ------
    table : astropy.table.Table
        Returns an astropy table with ID, corrected RA, corrected Dec, and Mag(?Some ppl might find this benifical for contamination reasons?)
    """

    # We need to remove any nan values from our proper  motion list
    # Doing this will allow objects which do not have proper motion to still be displayed
    table["pmRA"] = np.ma.filled(table["pmRA"].astype(float), 0.0)
    table["pmDEC"] = np.ma.filled(table["pmDEC"].astype(float), 0.0)
    # If an object does not have a parallax then we treat it as if the object is an "infinite distance"
    # and set the parallax to 1e-7 arcseconds or 10Mpc.
    table["Plx"] = np.ma.filled(table["Plx"].astype(float), 1e-4)

    # Suppress warning caused by Astropy as noted in issue 111747 (https://github.com/astropy/astropy/issues/11747)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="negative parallaxes")

        # Get the input data from the table
        c = SkyCoord(
            ra=table["RAJ2000"],
            dec=table["DEJ2000"],
            distance=Distance(parallax=table["Plx"].quantity, allow_negative=True),
            pm_ra_cosdec=table["pmRA"],
            pm_dec=table["pmDEC"],
            frame="icrs",
            obstime=equinox,
        )

    # Suppress warning caused by Astropy as noted in issue 111747 (https://github.com/astropy/astropy/issues/11747)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="ERFA function")
        warnings.filterwarnings("ignore", message="invalid value")

        # Calculate the new values
        c_motion = c.apply_space_motion(new_obstime=epoch)

    # Add new data corrected RA and Dec
    table["RA"] = c_motion.ra.to(u.deg)
    table["DEC"] = c_motion.dec.to(u.deg)

    # Get the index of the targets with zero proper motions
    pmzero_index = np.where((table["pmRA"] == 0.0) & (table["pmDEC"] == 0.0))

    # In those instances replace with J2000 values
    table["RA"][pmzero_index] = table["RAJ2000"][pmzero_index]
    table["DEC"][pmzero_index] = table["DEJ2000"][pmzero_index]

    return table
