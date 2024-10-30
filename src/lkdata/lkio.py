from astropy.io import fits
from .dataset import DataSet
from .datacube import DataCube, ErrorCube


def parse_filetype(hdu: fits.HDUList):
    def make_DataSet(hdu, cube_keys, err_cube_keys, time_dict):
        ds = DataSet()

        time_dict = {}
        for k in time_keys:
            time_dict[k] = hdu[1].data[k].astype(float)

        for ii in range(len(cube_keys)):
            if ii == 0:
                ds = DataSet(
                    {
                        cube_keys[ii]: DataCube(
                            hdu[1].data[cube_keys[ii]].astype(float),
                            time_indices=time_dict,
                            index=cube_keys[ii],
                        )
                    }
                )
            else:
                ds.data[cube_keys[ii]] = DataCube(
                    hdu[1].data[cube_keys[ii]].astype(float), time_indices=time_dict
                )
        for k in err_cube_keys:
            ds.data[k] = ErrorCube(hdu[1].data[k].astype(float), time_indices=time_dict)

        return ds

    # Add a function for each HLSP?
    if hdu[0].header.get("ORIGIN") == "NASA/Ames":
        cube_keys = ["RAW_CNTS", "FLUX", "FLUX_BKG"]
        err_cube_keys = ["FLUX_ERR", "FLUX_BKG_ERR"]
        time_keys = ["TIME", "TIMECORR", "CADENCENO", "POS_CORR1", "POS_CORR2"]
        # TODO: add WCS/CR/other boolean cubes?

        return make_DataSet(hdu, cube_keys, err_cube_keys, time_keys)
