"""Classes to work with searching, and io."""
# Third-party
from astropy.utils.data import download_file  # noqa: E402
import os
from .. import get_logger

logger = get_logger()


def download_large_file(url):
    p = download_file(
        url,
        cache=True,
        pkgname="lightkurve-large-files",
        show_progress=True,
    )
    if not os.path.isfile(p):
        p = download_file(
            url,
            cache="update",
            pkgname="lightkurve-large-files",
            show_progress=True,
        )
    logger.info(f"Downloaded {url} to large file cache.")


from .fits import *  # noqa: F403, E402
