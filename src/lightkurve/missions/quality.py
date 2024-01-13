"""Classes for working with quality bitmasks, and removing bad data"""
import logging
from typing import Dict
import numpy as np

from astropy.units.quantity import Quantity
import astropy.units as u


log = logging.getLogger(__name__)


__all__ = [
    "QualityFlags",
]


class QualityFlags(object):
    """Abstract class for handling quality flags"""

    STRINGS: Dict[int, str] = {}
    OPTIONS: Dict[str, int] = {}

    @classmethod
    def decode(cls, quality):
        """Converts a QUALITY value into a list of human-readable strings.

        This function takes the QUALITY bitstring that can be found for each
        cadence in Kepler/K2/TESS' pixel and light curve files and converts into
        a list of human-readable strings explaining the flags raised (if any).

        Parameters
        ----------
        quality : int
            Value from the 'QUALITY' column of a Kepler/K2/TESS pixel or lightcurve file.

        Returns
        -------
        flags : list of str
            List of human-readable strings giving a short description of the
            quality flags raised.  Returns an empty list if no flags raised.
        """
        # If passed an astropy quantity object, get the value
        if isinstance(quality, Quantity):
            quality = quality.value
        result = []
        for flag in cls.STRINGS.keys():
            if quality & flag > 0:
                result.append(cls.STRINGS[flag])
        return result

    @classmethod
    def create_quality_mask(cls, quality_array, bitmask=None):
        """Returns a boolean array which flags good cadences given a bitmask.

        This method is used by the readers of :class:`KeplerTargetPixelFile`
        and :class:`KeplerLightCurve` to initialize their `quality_mask`
        class attribute which is used to ignore bad-quality data.

        Parameters
        ----------
        quality_array : array of int
            'QUALITY' column of a Kepler target pixel or lightcurve file.
        bitmask : int or str
            Bitmask (int) or one of 'none', 'default', 'hard', or 'hardest'.

        Returns
        -------
        boolean_mask : array of bool
            Boolean array in which `True` means the data is of good quality.
        """
        # Return an array filled with `True` by default (i.e. ignore nothing)
        if bitmask is None:
            return np.ones(len(quality_array), dtype=bool)
        if isinstance(quality_array, u.Quantity):
            quality_array = quality_array.value
        # A few pre-defined bitmasks can be specified as strings
        if isinstance(bitmask, str):
            try:
                bitmask = cls.OPTIONS[bitmask]
            except KeyError:
                valid_options = tuple(cls.OPTIONS.keys())
                raise ValueError(
                    "quality_bitmask='{}' is not supported, "
                    "expected one of {}"
                    "".format(bitmask, valid_options)
                )
        # The bitmask is applied using the bitwise AND operator
        quality_mask = (quality_array & bitmask) == 0
        # Log the quality masking as info or warning
        n_cadences = len(quality_array)
        n_cadences_masked = (~quality_mask).sum()
        percent_masked = 100.0 * n_cadences_masked / n_cadences
        logmsg = (
            "{:.0f}% ({}/{}) of the cadences will be ignored due to the "
            "quality mask (quality_bitmask={})."
            "".format(percent_masked, n_cadences_masked, n_cadences, bitmask)
        )
        if percent_masked > 20:
            log.warning("Warning: " + logmsg)
        else:
            log.info(logmsg)
        return quality_mask
