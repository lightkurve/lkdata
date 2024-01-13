#!/usr/bin/env python
# -*- coding: utf-8 -*-
# mypy: ignore-errors
"""Helper functions to work with notebooks and jupyter environments."""
import os
import urllib


def _get_notebook_environment():
    """Returns 'jupyter', 'colab', or 'terminal'.

    One can detect whether or not a piece of Python is running by executing
    `get_ipython().__class__`, which returns the following result:

        * Jupyter notebook: `ipykernel.zmqshell.ZMQInteractiveShell`
        * Google colab: `google.colab._shell.Shell`
        * IPython terminal: `IPython.terminal.interactiveshell.TerminalInteractiveShell`
        * Python terminal: `NameError: name 'get_ipython' is not defined`
    """
    try:
        ipy = str(type(get_ipython())).lower()
        if "zmqshell" in ipy:
            return "jupyter"
        if "colab" in ipy:
            return "colab"
    except NameError:
        pass  # get_ipython() is not a builtin
    return "terminal"


def is_notebook():
    """Returns `True` if we are running in a notebook."""
    return _get_notebook_environment() in ["jupyter", "colab"]


def remote_jupyter_proxy_url(port):
    """
    Callable to configure Bokeh's show method when a proxy must be
    configured.    If port is None we're asking about the URL
    for the origin header.
    """
    base_url = os.environ["LK_JUPYTERHUB_EXTERNAL_URL"]
    host = urllib.parse.urlparse(base_url).netloc

    # If port is None we're asking for the URL origin
    # so return the public hostname.
    if port is None:
        return host

    service_url_path = os.environ["JUPYTERHUB_SERVICE_PREFIX"]
    proxy_url_path = "proxy/%d" % port

    user_url = urllib.parse.urljoin(base_url, service_url_path)
    full_url = urllib.parse.urljoin(user_url, proxy_url_path)
    return full_url


def finalize_notebook_url(notebook_url):
    """Based on `notebook_url` and the environment, compute a final value for
    notebook_url to be passed on to bokeh enabling transparent operation on JupyterHub.

    See Bokeh instructions here:
    https://docs.bokeh.org/en/latest/docs/user_guide/output/jupyter.html

    This handles two aspects of Bokeh made tricky by JupyterHub, firstly
    accessing the random Bokeh server port while behind a proxy, and second not
    triggering CORS restrictions while accessing a second server.

    A key aspect of the computed URL is the externally visible DNS name of the
    JupyterHub, so for the case of TIKE we might have:

    export LK_JUPYTERHUB_EXTERNAL_URL="https://timeseries.science.stsci.edu"

    If LK_JUPYTERHUB_EXTERNAL_URL is implicitly defined by the hub environment,
    JupyterHub users can nominally ignore the notebook_url parameter and
    Lightkurve should "just work" as if the local default URL localhost:8888
    was sufficient.

    For example remote_jupyter_proxy_url(25346) would return the URL
    "https://test.timeseries.science.stsci.edu/user/homer@stsci.edu/proxy/24356"

    which is essentially HUB + USER_SESSION + BOKEH_PORT_IN_SESSION

    The function result should be identical to past behavior unless the definition
    of LK_JUPYTERHUB_EXTERNAL_URL indicates JupyterHub is in use.  In this case the
    use of remote_jupyter_proxy_url is activated.   This effectively makes it the
    JupyterHub default instead of localhost:8888.
    """
    if notebook_url is not None:
        return notebook_url
    elif os.environ.get("LK_JUPYTERHUB_EXTERNAL_URL"):
        return remote_jupyter_proxy_url
    else:
        return "localhost:8888"
