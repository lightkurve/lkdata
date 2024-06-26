import numpy as np


class hdulist_parser:
    def __init__(self, hdulist):
        self.flux_array = hdulist[1].data["FLUX"].astype(float)
        self.flux_units = hdulist[1].columns["FLUX"].unit
        self.flux_err_array = hdulist[1].data["FLUX_ERR"].astype(float)
        self.flux_err_units = hdulist[1].columns["FLUX_ERR"].unit
        self.time = hdulist[1].data["TIME"].astype(float)
        self.time_units = hdulist[1].columns["TIME"].unit
        self.time_corr = hdulist[1].data["TIMECORR"].astype(float)
        self.time_corr_units = hdulist[1].columns["TIMECORR"].unit
        c0, r0 = (
            hdulist[1].header["1CRV?P"][0],
            hdulist[1].header["2CRV?P"][0],
        )  # TODO: may need to expand key words to look for
        self.row = np.arange(self.flux_array.shape[1]) + r0
        self.col = np.arange(self.flux_array.shape[2]) + c0
