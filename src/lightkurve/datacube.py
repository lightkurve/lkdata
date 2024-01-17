"""Classes and tools for working with 3 dimensional data."""

import numpy as np
import astropy.units as u
from astropy.table import QTable
import astropy.io.fits as fits

"""class SadHumanCube(depth, row, column, unit):
   def __init__(self) -> None:
      data = np.ndarray(shape=(depth, row, column), dtype=float)
      # this should be forced to be user specified not unit = u.dimensionless_unscaled
      unit = u.dimensionless_unscaled
      # unit type checks, format checks
    def get_data():
      return data * unit
# Should we not use the above class?  Seems unescessary?"""


class BorgCube(object):
    """def __init__(self, path) -> None:

    # list of cubes is coming from io *somewhere*
    if path:
        # What API do we want?
        # Hypercube(path)
        # ypercube.from_fits(path)
        # loop through each extension
        # - create data add_cube-with _get_datacube
        # - save meta dectionary
        # Make sure this fits, or else throw an error

        def _from_fits(path):
            return " "

        # whats the minimal required information
        # A WCS -
        # one cube with an assosciated unit
        # one timeseries time index
        # cube_keys = list_of_whats_in_each_cube(e.g. 'flux', 'flux_err'))

        self.cube_data, self.cube_keys, self.timeseries = _from_fits(path)
    else:
        self.cube_data = ""
        self.cube_keys = ""
        self.timeseries = ""
    # parse some inputs if not from file
    # If No Datacube class, use this for units?
    self.cubes = {self.cube_keys: self.cube_data}
    self.cube_units = {self.cube_keys: u.unit}"""

    def __init__(
        self,
        flux,
        flux_err,
        time,
        flux_unit,
        quality_bitmask="default",
        targetid=None,
        **kwargs,
    ):
        # Can pass in a path OR provide at minimum a cube of flux, flux_err, and time array
        # Time should have units, if not will assume BJD?

        self.cube_data = {"flux": flux, "flux_err": flux_err}
        self.cube_units = {"flux": flux_unit, "flux_err": flux_unit}
        self.flux_unit = flux_unit
        self.cube_keys = self.cube_data.keys()
        self.time = time
        self.quality_bitmask = quality_bitmask
        self.targetid = targetid
        # A TPF file h

        # For consistency with `LightCurve`, provide a `meta` dictionary
        # May not implement LightCurve the same way anymore.
        # self.meta = HduToMetaMapping(self.hdu[0])

    @staticmethod
    def from_file(self, path: str):
        ### Path is either a path to a fits file or an hdu object.
        # The hdu object is assumed to be spoc-like with flux/flux_err/time in the first extension
        self.path = path
        if isinstance(path, fits.HDUList):
            hdu = path
        else:
            hdu = fits.open(self.path)  # Add kwargs?
            # hdu = deepcopy(hdulist)
        if hdu[1].header["TUNIT5"] == "e-/s":
            unit = "electron/s"
        else:
            unit = "unitless"
        return BorgCube(
            flux=hdu[1].data["FLUX"],
            flux_err=hdu[1].data["flux_err"],
            time=hdu[1].data["TIME"],
            flux_unit=unit,
        )

        # Propose if you want more than the bare minimum, we write an add_from_fits method to add data
        # Also a function that goes through the image extensions and provides users with all options

        # def get_header(self, ext=0):
        """Returns the metadata embedded in the file.

        Target Pixel Files contain embedded metadata headers spread across three
        different FITS extensions:

        1. The "PRIMARY" extension (``ext=0``) provides a metadata header
           providing details on the target and its CCD position.
        2. The "PIXELS" extension (``ext=1``) provides details on the
           data column and their coordinate system (WCS).
        3. The "APERTURE" extension (``ext=2``) provides details on the
           aperture pixel mask and the expected coordinate system (WCS).

        However - we want to make this more generic.

        Parameters
        ----------
        ext : int or str
            FITS extension name or number.

        Returns
        -------
        header : `~astropy.io.fits.header.Header`
            Header object containing metadata keywords.
        """
        # return self.hdu[ext].header

    # Defining a bunch of properties for our data cubes
    '''
    We've rearranged some of this now, these will come from the data_cube
    @property
    def flux(self, flux=None, unit=u.unitless) -> u.Quantity:
        """Returns the flux"""
        if flux != None:
            if self.get_header(1).get("TUNIT5") == "e-/s":
                unit = "electron/s"
            else:
                unit = unit
        return u.Quantity(flux, unit=unit)

    @property
    def flux_err(self, flux_err=None, unit=u.unitless) -> u.Quantity:
        """Returns the flux_err"""
        if flux_err != None:
            if self.get_header(1).get("TUNIT6") == "e-/s":
                unit = "electron/s"
            else:
                unit = unit
        return u.Quantity(flux_err, unit=unit)

    @property
    def flux_bkg(self, flux_bkg=None, unit=u.unitless) -> u.Quantity:
        """Returns the background flux"""
        if flux_bkg != None:
            self.hdu[1].data["FLUX_BKG"] = "e-/s"
            unit = "electron/s"
        else:
            unit = unit
        return u.Quantity(flux_bkg, unit=unit)

    @property
    def flux_bkg_err(self, flux_bkg_err=None, unit=u.unitless) -> u.Quantity:
        """Returns the background flux"""
        if flux_bkg_err != None:
            self.hdu[1].data["FLUX_BKG_ERR"] = "e-/s"
            unit = "electron/s"
        else:
            unit = unit
        return u.Quantity(flux_bkg_err, unit=unit)

    @property
    def time(self, time=None, unit=u.unitless) -> u.Time:
        """Returns the time"""
        if time != None:
            time = self.hdu[1].data["TIME"]
            # Some data products have missing time values;
            # we need to set these to zero or `Time` cannot be instantiated.
            time[~np.isfinite(time)] = 0

        bjdrefi = self.hdu[1].header.get("BJDREFI")
        if bjdrefi == 2454833:
            time_format = "bkjd"
        elif bjdrefi == 2457000:
            time_format = "btjd"
        else:
            time_format = "jd"

        return Time(
            time_values,
            scale=self.hdu[1].header.get("TIMESYS", "tdb").lower(),
            format=time_format,
        )'''

    # @property
    # def wcs(self) -> WCS:
    #    """Returns an `astropy.wcs.WCS` object with the World Coordinate System
    #    solution from the file.

    #    #Is this ia must?

    #   Returns
    #    -------
    #    w : `astropy.wcs.WCS` object
    #        WCS solution
    #    """
    #    if "MAST" in self.hdu[0].header["ORIGIN"]:  # Is it a TessCut TPF?
    # TPF's generated using the TESSCut service in early 2019 only appear
    # to contain a valid WCS in the second extension (the aperture
    # extension), so we treat such files as a special case.
    #        return WCS(self.hdu[2])
    #    else:
    # For standard (Ames-pipeline-produced) TPF files, we use the WCS
    # keywords provided in the first extension (the data table extension).
    # Specifically, we use the WCS keywords for the 5th data column (FLUX).
    #        wcs_keywords = {
    #            "1CTYP5": "CTYPE1",
    #            "2CTYP5": "CTYPE2",
    #            "1CRPX5": "CRPIX1",
    #            "2CRPX5": "CRPIX2",
    #            "1CRVL5": "CRVAL1",
    #            "2CRVL5": "CRVAL2",
    #            "1CUNI5": "CUNIT1",
    #            "2CUNI5": "CUNIT2",
    #            "1CDLT5": "CDELT1",
    #            "2CDLT5": "CDELT2",
    #            "11PC5": "PC1_1",
    #            "12PC5": "PC1_2",
    #            "21PC5": "PC2_1",
    #             "22PC5": "PC2_2",
    #             "NAXIS1": "NAXIS1",
    #            "NAXIS2": "NAXIS2",
    #        }
    #        mywcs = {}
    #        for oldkey, newkey in wcs_keywords.items():
    #            if self.hdu[1].header.get(oldkey, None) is not None:
    #                mywcs[newkey] = self.hdu[1].header[oldkey]
    #        return WCS(mywcs)

    # @property
    # def data_cubes(self) -> Dictionary:
    #    "Gets cube-like data (depth, row, column)"
    # flux, flux_err, flux_bkg, flu  """Returns the flux for all good-quality cadences."""

    def __repr__(self) -> str:
        pass
        # build in default operators here for math function on arrays?

    def _get_DataCube(self, key):
        # pandas like functionality?  eg. BorgCube.flux represents the cubes['flux'] numpy NDARRAY
        return self.cube_data[key] * self.cube_units[key]

    def cube_key(self):
        return QTable(self.cube_keys, self.cube_units, names=("keys", "units"))

    def index_cube(self, mask):
        # 1D boolean array to mask over time of length de[th]
        # Apply this to each cube and each 1D array
        for key in self.cube_keys:
            self.cube_data[key] = self.cube_data[key][mask, :, :]
            # do some masking
        # then do some astropy timeseries table masking

        # timeseries not implemented yet, commented out below
        # self.timeseries = self.timeseries[mask]
        return

    def rotate_cube(cube_key):
        # rotate or warp pixels, sort of optional
        raise NotImplementedError

    def bin_Borgcube(time):
        # bins all cubes and the timeserires
        raise NotImplementedError

    def bin_Borgcube_spatially(npixels):
        # bins all cubes
        raise NotImplementedError

    def plot(self, overlay=None):
        # Can give a key or a cube (ie, background cube + flux)
        # overlay='gaia' plots sources
        # How much of this should be jdaviz?
        raise NotImplementedError

    def get_background_mask(self):
        # add a background cube
        raise NotImplementedError

    def image_operation(Im_2darray):
        # perform an mathematical operation from a 2-D Image on each
        # index of BorgCube
        # return self or modify in place?
        # should 2darray be stored in BorgCube? Collective is feeling no
        raise NotImplementedError

    def interact(self, options=("catalog", "phot", "gaia")):
        """Function for interacting with the Hypercube Data"""
        # Interact has selectable toggles/widgets
        # one stop shop?  look into jdaviz
        # Looks like Jdaviz would be our one stop shop tool that we would want s
        # Take python package and specify how we want to use?
        # Always a picture of the TPF File
        # Toggle additional functionality
        # Toggle on gaia panel to show you a gaia/2mass catalog
        # Toggle on gaia panel to show you a gaia/2mass image alongside
        # Toggle on lightcurve to display interactive photometry
        # store data from interactive plots?
        # display aperture completeness for known sources
        # display aperture blending for known sources?
        # toggleable display "cube" - e.g. background model, etc

        raise NotImplementedError

    def add_cube(self, data, key, unit=None):
        # what do we do with no unit specified?  unitless?  ore raise error?
        # ValidateCube
        # Add Cube to Cube Collection
        # How many are we allowing - will people abuse this?

        # Check if data has a unit, or if not if unit is specified
        if isinstance(data, u.Quantity):
            unit = data.unit
        else:
            if not isinstance(unit, u.Quantity):
                raise TypeError("No Unit Specified")

        if not isinstance(key, str):
            raise TypeError(f"Key {key} is not a string")
        # Check to make sure that data is a 3-dimensional numpy array
        if isinstance(data, np.ndarray):
            if data.ndim == 3:
                if data.shape == self.cube_data["flux"].shape:
                    # every Cube should have a flux
                    self.cube_data[key] = data
                    self.cube_units[key] = unit
                else:
                    raise ValueError("Input Data Shape does not match existing cubes")
            else:
                raise TypeError("Input data is not 3 dimensions")
        else:
            raise TypeError(" Input data is not a numpy array")

    def remove_cubes(self, keys):
        """Removes a data cube and assosciated unit from the dictionary of cubes
        & cube_keys
        """
        for key in keys:
            try:
                self.cube_data.pop(key)
            except KeyError:
                print(f"Cube {key} not fount in data cubes")
            try:
                self.cube_units.pop(key)
            except KeyError:
                print(f"Cube {key} not fount in data cube units")

    def _specify_target_location(rowcol_tuple, aperture, radec_tuple, brightness):
        # DO WE NEED THIS?????????
        # TPF have it, do TESSCut?
        # Any other HLSP - do they have this?
        raise NotImplementedError

    def PRF_lightcurve(self):
        # This will use the PRF module to find the best aperture using the PRF model?
        raise NotImplementedError

    def aper_lightcurve(self, keys, source=[], background=[]):
        # This uses a given aperture, such as the pipeline or a user specified one
        raise NotImplementedError

    def to_timeseries(self, keys=["flux", "fluxerr"], method="aper"):
        """Turn our three-dimensional data into a 1D timeseries using a variety of methods

        Parameters
        ----------
        keys: tuple or str
            What data cubes we want to turn into a time series

        method: str
            How we want to turn data cubes[keys] into a timeseries object

        Returns
        -------
            timeseries : ~'lightkurve.timeseries'
                function call to an appropriate method, that returns a lightkurve.timeseries object

        """
        # are we passing a PRF class for method=PRF? what about Aper?  Do we need photometry classes?

        # According to mypy we can't use case statements, re-write as ifs I guess
        # match method:
        #    case "aper":
        #        raise NotImplementedError
        #    case "PRF":
        #        raise NotImplementedError
        #    case _:
        #        raise ValueError(f"to_timeseries method {method} not found")
        raise NotImplementedError
