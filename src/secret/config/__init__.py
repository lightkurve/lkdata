import os
import warnings

import astropy.config as astropyconfig


ROOTNAME = "lightkurve"


class ConfigNamespace(astropyconfig.ConfigNamespace):
    rootname = ROOTNAME


class ConfigItem(astropyconfig.ConfigItem):
    rootname = ROOTNAME


def get_config_dir():
    """
    Determines the package configuration directory name and creates the
    directory if it doesn't exist.

    Returns
    -------
    configdir : str
        The absolute path to the configuration directory.

    """
    return astropyconfig.get_config_dir(ROOTNAME)


def get_cache_dir():
    """
    Determines the default data cache directory name and creates the
    directory if it doesn't exist.

    The value can be also configured via ``cache_dir`` configuration parameter.

    Users can set their defaults in their configuration file, defaulted at ``$HOME/.lightkurve/config/lightkurve.cfg``.

    Furthermore, they can also change the values at runtime via `lightkurve.conf` object.

    The remaining specifics can be found in `Astropy documentation <https://docs.astropy.org/en/stable/config/index.html>`_.

    Returns
    -------
    cachedir : str
        The absolute path to the cache directory.

    Examples
    --------
    To configure "/my_research/data" as the `cache_dir`, users can set it:

    1. in the user's ``lightkurve.cfg`` file::

        [config]
        cache_dir = /my_research/data

    2. at run time::

        import lightkurve
        lightkurve.conf.cache_dir = '/my_research/data'

    See :ref:`configuration <api.config>` for more information.
    """
    from .. import conf

    cache_dir = conf.cache_dir
    if cache_dir is None or cache_dir == "":
        cache_dir = astropyconfig.get_cache_dir(ROOTNAME)
    cache_dir = _ensure_cache_dir_exists(cache_dir)
    cache_dir = os.path.abspath(cache_dir)

    return cache_dir


def _ensure_cache_dir_exists(cache_dir):
    if os.path.isdir(cache_dir):
        return cache_dir
    else:
        # if it doesn't exist, make a new cache directory
        try:
            os.mkdir(cache_dir)
        # user current dir if OS error occurs
        except OSError:
            warnings.warn(
                "Warning: unable to create {} as cache dir "
                " (for downloading MAST files, etc.). Use the current "
                "working directory instead.".format(cache_dir)
            )
            cache_dir = "."
        return cache_dir
