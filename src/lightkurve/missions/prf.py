"""Generic tools to work with PRFs"""
from abc import ABC, abstractmethod
from typing import Union, List, Tuple
import numpy.typing as npt
import numpy as np
import math
import warnings
from ..utils import LightkurveWarning


class PRF(ABC):
    """Abstract PRF class for working with Point Spread Functions"""

    @abstractmethod
    def __init__(
        self,
        target_locations: Union[List[Tuple], Tuple] = (5.5, 5.5),
        origin: Tuple = (0, 0),
        shape: Tuple = (11, 11),
    ):
        """
        A generic base class object for PRFs. Not to be used directly.

        Will enable users to create an image with shape `shape`, where the
        origin (lower left row, lower left column) is at `(row, column)`.

        Parameters:
        -----------
        target_locations : List[Tuple] or Tuple
                row, column pixel locations for the Pixel Response Functions
        origin: Tuple column value
        row : int
        shape   Tuple
        """
        # load PSF file
        # Initialize object
        self.origin_row, self.origin_column = origin
        self.shape = shape
        self.target_row, self.target_column = np.atleast_2d(target_locations)

    def __repr__(self):
        return "PRF Base Class"

    def get_target_aperture(
        self,
        prf_model: npt.ArrayLike,
        target_idx: int = 0,
        min_completeness: float = 0.9,
    ) -> npt.ArrayLike:
        """
        Based on completeness requirement, create an aperture.
        This basic aperture does NOT account for contamination by other sources.

        Parameters:
        -----------
        prf_model : npt.ArrayLike
                3D cube of the PRFs of all sources with shape (nsources x nrows x ncolumns)
                Only the target source is used for this function.
        target_idx : int
                The index of the source to be considered as the target (default 0)
        min_completeness : float
                Minimum fraction of flux contained within the aperture. Value between 0 and 1


        Returns:
        --------
        aperture : npt.ArrayLike
                2D boolean array of the same (nrows x ncolumns) as `prf_model`.
                True where source is inside aperture
        """

        if (min_completeness < 0) | (min_completeness > 1):
            raise ValueError("Completeness must be between 0 and 1")

        prf = prf_model[target_idx, :, :]

        if np.sum(prf) > 1.0:
            prf = prf / np.sum(prf)

        # If completeness = 1, return any pixel that contains any amount of flux
        if min_completeness == 1.0:
            aperture = prf.astype(bool)
            aperture[prf != 0.0] = True

        else:
            sort = np.argsort(prf.flatten())
            cusu = np.cumsum(prf.flatten()[sort])

            # indices that are "inside" aperture
            ap_index = sort[(1 - cusu) < min_completeness]
            aperture = np.zeros(prf.shape, dtype=bool).flatten()
            aperture[ap_index] = True
            # reshape to shape of prf
            aperture = aperture.reshape(np.shape(prf))

        return aperture

    def get_contamination(
        self,
        prf_model: npt.ArrayLike,
        aperture: npt.ArrayLike,
        fluxes: npt.ArrayLike,
        target_idx: int = 0,
    ) -> float:
        """

        Parameters:
        -----------
        prf_model : npt.ArrayLike
            3D cube of the PRFs of all sources with shape (nsources x nrows x ncolumns)
        aperture: npt.ArrayLike
            2D boolean array same size as `prf`, True where source is inside aperture
        fluxes : npt.ArrayLike
            Array of fluxes for each source with shape (nsources)
        target_idx : int
            The index of the source to be considered as the target (default 0)

        Returns:
        --------
        contamination : float
            Fraction of total flux in the aperture that comes from the target source.
            Will be a value between 0 and 1 (1 being all flux comes from the target source)
        """

        if prf_model.shape[0] == 1:
            print(
                "Only a target source is provided. Contamination will be 1 (no contamination)"
            )
            return 1.0

        prfs = self.get_flux_weighted_prf(prf_model, fluxes)

        target_flux = np.sum(prfs[target_idx, aperture])
        all_flux = np.sum(prfs[:, aperture])
        return target_flux / all_flux

    def get_flux_weighted_prf(
        self, prf_model: npt.ArrayLike, fluxes: npt.ArrayLike
    ) -> npt.ArrayLike:
        """

        Parameters:
        -----------
        prf_model : npt.ArrayLike
            3D cube of the PRFs of all sources with shape (nsources x nrows x ncolumns)
        fluxes : npt.ArrayLike
            Array of fluxes for each source with shape (nsources)

        Returns:
        --------
        prf_with_flux : npt.ArrayLike
            prf model multiplied by total expected flux

        """
        if len(fluxes) != prf_model.shape[0]:
            raise ValueError(
                "number of input fluxes do not match number of elements in prf_model."
            )
        return np.array([prf_model[ii, :, :] * fluxes[ii] for ii in range(len(fluxes))])

    def get_completeness(
        self, prf_model: npt.ArrayLike, aperture: npt.ArrayLike, target_idx: int = 0
    ) -> float:
        """
        Returns the fraction of total flux from the target contained in a given aperture

        Parameters:
        -----------
        prf_model : npt.ArrayLike
            3D cube of the PRFs of all sources with shape (nsources x nrows x ncolumns)
        aperture: npt.ArrayLike
            2D boolean array same size as the prf model, True where source is inside aperture
        target_idx : int
            The index of the source to be considered as the target (default 0)

        Returns:
        --------
        completeness : float
            fraction of total flux contained within the aperture
        """
        return np.sum(prf_model[target_idx, aperture]) / np.sum(
            prf_model[target_idx, :, :]
        )

    def evaluate(
        self,
        center_col=None,
        center_row=None,
        scale: float = 1.0,
        rotation_angle: float = 0.0,
    ):
        """
        Interpolates the PRF model onto detector coordinates.

        Parameters
        ----------
        center_col, center_row : float
                                        Column and row coordinates of the center
        scale : float
                                        Pixel scale stretch parameter, can be used to account for focus changes.
                                        Values > 1 stretch the image, Values < 1 make the PRF more compact.
                                        E.g. a scale value of 2 will double the PRF footprint.
        rotation_angle : float
                                        Rotation angle in radians

        Returns
        -------
        prf : 2D array
            Two dimensional array representing the PRF values parametrized
            by flux, centroids, widths, and rotation as applicble.
        """
        if scale <= 0:
            scale = 1
            warnings.warn(
                "Scale can not be <= 0. Resetting scale to 1.",
                LightkurveWarning,
            )

        scale = 1.0 / scale

        if center_col is None:
            center_col = self.origin_column + self.shape[1] / 2
        if center_row is None:
            center_row = self.origin_row + self.shape[0] / 2

        delta_col = self.target_column - center_col
        delta_row = self.target_row - center_row

        if (scale == 1.0) and (rotation_angle == 0.0):
            prf = self.interpolate(delta_row, delta_col)

        else:
            cosa = math.cos(rotation_angle)
            sina = math.sin(rotation_angle)

            delta_col, delta_row = np.meshgrid(delta_col, delta_row)
            rot_row = delta_row * cosa - delta_col * sina
            rot_col = delta_row * sina + delta_col * cosa

            prf = self.interpolate(
                rot_row.flatten() * scale, rot_col.flatten() * scale, grid=False
            ).reshape(self.shape)

            prf = prf / (1 / scale) ** 2

            # Normalize the values when 'scale' is set to decrease the PRF spread
            if (1 / scale) < 1:
                prf = prf / np.sum(prf)

        # Ignore relative flux below a given threshold as the resulting flux change is not detectable
        prf[prf < 1e-16] = 0.0
        return prf

    def gradient(
        self,
        center_col=None,
        center_row=None,
        flux: float = 1.0,
        scale: float = 1.0,
        rotation_angle: float = 0.0,
    ) -> list:
        """
        This function returns the gradient of the PRF model with
        respect to center_col, center_row, flux, scale,
        and rotation_angle.

        Parameters
        ----------
        center_col, center_row : float
            Column and row coordinates of the center
        flux : float
            Total integrated flux of the PRF
        scale : float
            Pixel scale stretch parameter, i.e. the numbers by which the PRF
            model needs to be multiplied in the column and row directions to
            account for focus changes.
            Values > 1 stretch the image, Values < 1 make the PRF more compact.
            E.g. a scale value of 2 will double the PRF footprint.
        rotation_angle : float
            Rotation angle in radians

        Returns
        -------
        grad_prf : list
            Returns a list of arrays where the elements are the partial derivatives
            of the PRF model with respect to center_col, center_row, flux, scale_col, scale_row, and rotation_angle, respectively.
        """

        if scale <= 0:
            scale = 1
            warnings.warn(
                "Scale can not be <= 0. Resetting scale to 1.",
                LightkurveWarning,
            )
        # Implemented to match intuition that larger scale value results in a broader PRF
        scale = 1.0 / scale

        if center_col is None:
            center_col = self.origin_column + self.shape[1] / 2
        if center_row is None:
            center_row = self.origin_row + self.shape[0] / 2

        delta_col = self.target_column - center_col
        delta_row = self.target_row - center_row

        if (scale == 1.0) and (rotation_angle == 0.0):
            deriv_flux = self.interpolate(delta_row, delta_col)
            deriv_center_col = -flux * self.interpolate(delta_row, delta_col, dy=1)
            deriv_center_row = -flux * self.interpolate(delta_row, delta_col, dx=1)

            return [deriv_center_col, deriv_center_row, deriv_flux]

        else:
            cosa = math.cos(rotation_angle)
            sina = math.sin(rotation_angle)

            delta_col, delta_row = np.meshgrid(delta_col, delta_row)
            rot_row = delta_row * cosa - delta_col * sina
            rot_col = delta_row * sina + delta_col * cosa

            # for a proof of the maths that follow, see the pdf attached
            # to pull request #198 in lightkurve GitHub repo.
            deriv_flux = self.interpolate(
                rot_row.flatten() * scale, rot_col.flatten() * scale, grid=False
            ).reshape(self.shape)

            interp_dy = self.interpolate(
                rot_row.flatten() * scale,
                rot_col.flatten() * scale,
                grid=False,
                dy=1,
            ).reshape(self.shape)

            interp_dx = self.interpolate(
                rot_row.flatten() * scale,
                rot_col.flatten() * scale,
                grid=False,
                dx=1,
            ).reshape(self.shape)

            scale_row_times_interp_dx = scale * interp_dx
            scale_col_times_interp_dy = scale * interp_dy

            deriv_center_col = -flux * (
                cosa * scale_col_times_interp_dy - sina * scale_row_times_interp_dx
            )
            deriv_center_row = -flux * (
                sina * scale_col_times_interp_dy + cosa * scale_row_times_interp_dx
            )
            deriv_scale_row = flux * interp_dx * rot_row
            deriv_scale_col = flux * interp_dy * rot_col
            deriv_rotation_angle = flux * (
                interp_dy * scale * (delta_row * cosa - delta_col * sina)
                - interp_dx * scale * (delta_row * sina + delta_col * cosa)
            )

            return [
                deriv_center_col,
                deriv_center_row,
                deriv_flux,
                deriv_scale_col,
                deriv_scale_row,
                deriv_rotation_angle,
            ]

    def _prf_model(
        self,
        center_col: Union[float, list[float], npt.ArrayLike],
        center_row: Union[float, list[float], npt.ArrayLike],
        scale: float = 1.0,
        rotation_angle: float = 0.0,
    ) -> npt.ArrayLike:
        """
        Creates a stack of PRF models.
        if center_col/center_row are lists (e.g., a list of pixel locations for each star
        located in a TPF), a prf will be generated for each.

        Parameters
        ----------
        center_col : float or list of floats
            column location of the target on the CCD.
        center_row : float or list of floats
            row location of the target on the CCD.
        scale : float
            Pixel scale stretch parameter, i.e. the numbers by which the PRF
            model needs to be multiplied in the column and row directions to
            account for focus changes. Default is 1 (no scaling)
        rotation_angle : float
             Rotation angle in radians. default 0.0, ie no rotation

        Returns
        -------
        PRF model : npt.ArrayLike
            3D cube of PRF models of shape (ntargets, nrows, ncolumns) at instrument pixel resolution

        """

        # PRF.evaluate returns a PRF for one onject.
        # Here, evaluate is called for each target provided (e.g., for each source within a tpf)
        if (center_col is None) or (center_row is None):
            warnings.warn(
                "center_col and center_row not providing. Defaulting to center pixel. ",
                LightkurveWarning,
            )
            center_col = self.origin_column + self.shape[-1] / 2
            center_row = self.origin_row + self.shape[-2] / 2
        if isinstance(center_col, (list, np.ndarray)):
            if len(center_col) != len(center_row):
                raise ValueError("Column/row locations must have the same shape.")
            prf_model = np.zeros((len(center_col), self.shape[0], self.shape[1]))
            for ii in range(len(center_col)):
                # Flux for each target is NOT taken into account by default.
                # To account for flux, see get_flux_weighted_prf().
                prf_model[ii, :, :] = self.evaluate(
                    center_col=center_col[ii],
                    center_row=center_row[ii],
                    scale=scale,
                    rotation_angle=rotation_angle,
                )
            return prf_model
        else:
            prf_model = self.evaluate(
                center_col, center_row, scale=scale, rotation_angle=rotation_angle
            )
            return np.expand_dims(prf_model, axis=0)

    def interpolate(self, row_grid, column_grid, dx=None, dy=None, grid=True):
        raise NotImplementedError

    @abstractmethod
    def _prepare_prf(self):
        """Method to open PRF files for given mission"""
        pass

    @abstractmethod
    def _read_prf_calibration_file(self):
        """Method to read PRF fits files for each mission and extract needed information"""
        pass
