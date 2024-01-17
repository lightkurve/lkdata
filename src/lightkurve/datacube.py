"""Classes and tools for working with 3 dimensional data."""

# import numpy as np
import astropy.units as u

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
    def __init__(self, path) -> None:
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
        self.cube_units = {self.cube_keys: u.unit}

    def __repr__(self) -> str:
        pass
        # build in default operators here for math function on arrays?

    def _get_DataCube(self):
        # pandas like functionality?  eg. BorgCube.flux represents the cubes['flux'] numpy NDARRAY
        return self.cubes[self.cube_keys] * self.cube_units[self.cube_keys]

    def cube_key(self):
        # returns a table of keys and units
        raise NotImplementedError

    def index_cube(self, mask):
        # 1D boolean array to mask over time of length de[th]
        # Apply this to each cube and each 1D array
        for key in self.cube_keys:
            self.cubes[key] = self.cubes[key][mask, :, :]
            # do some masking
        # then do some astropy timeseries table masking
        self.timeseries = self.timeseries[mask]
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

    def add_cube(self, SadHumanCube):
        # what do we do with no unit specified?  unitless?  ore raise error?
        # ValidateCube
        # Add Cube to Cube Collection
        # How many are we allowing - will people abuse this?
        raise NotImplementedError

    def remove_cube(self, key):
        # via cube_key
        raise NotImplementedError

    def _specify_target_location(rowcol_tuple, aperture, radec_tuple, brightness):
        # DO WE NEED THIS?????????
        # TPF have it, do TESSCut?
        # Any other HLSP - do they have this?
        raise NotImplementedError

    def to_timeseries(self, keys=["flux", "fluxerr"], method=("aper", "PRF")):
        # are we passing a PRF class for method=PRF? what about Aper?  Do we need photometry classes?
        raise NotImplementedError

    def PRF_lightcurve(self):
        # This will use the PRF module to find the best aperture using the PRF model?
        raise NotImplementedError

    def aper_lightcurve(self):
        # This uses a given aperture, such as the pipeline or a user specified one
        raise NotImplementedError
