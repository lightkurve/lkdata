"""
Tests for the functionality of the generic PRF module
"""
from lightkurve import PRF
# Create a grid that is super-sampled relative to true PRF pixels - to do this, creae a 2D Gaussian/exponential funcition

def make2DGaussian(size, fwhm = 3, center=None):
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)

generic_prf = PRF() #initialize model here
supersample_factor=10
supersampled_prf = make2DGaussian(prf_model.shape[0]*supersample_factor, fwhm=prf_model.shape[0]*supersample_factor/3)

# Need to flesh out _prepare_prf()
prf_model = generic_prf.evaluate()


class PRF(object):

    def __init__(self, prf_image:np.NDArray, row, column): 



class KeplerPRF(PRF):
    """Grabs the PRF files from the right place online
    Intializes them from files correctly"""
    def __init__(self, ..., quarter, channel):
        # open file
        # initialize super class



# Add check for completeness and contamination
# check the aperture matches expectation

