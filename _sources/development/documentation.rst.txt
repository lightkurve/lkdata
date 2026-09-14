.. _docs:

Building the documentation
==========================

Coding and documentation standards
----------------------------------

Lightkurve adopts AstroPy's coding guidelines and standards,
as documented in `AstroPy's Development Documentation <http://docs.astropy.org/en/stable/index.html#developer-documentation>`_.


Building documentation
----------------------

.. note::

    Building the documentation is not necessary unless you are
    writing new documentation or do not have internet access, because the
    latest version of the documentation is available online at
    `lightkurve.github.io/lkdata/ <https://lightkurve.github.io/lkdata/>`_.

Building the *lkdata* documentation requires `sphinx` and a few extra packages. We recommend using `poetry` to install the development dependencies::

    $  poetry install

To make a clean directory for the docs use::

    $ cd docs
    $ make clean

To build the documentation in HTML format, execute::

    $ cd docs
    $ make html

Serving Documentation Locally

To view the documentation with live updates as you write:

    $ make serve

This will start a local server. You can view the documentation in your browser at `http://localhost:8001`.

Stopping the Local Server

If you need to stop the local server, use:

    $ make stop-serve

.. Finally, if you have write permission to *lkdata*'s GitHub repository,
.. you can upload the documentation to the web server using::

..     $ make upload

.. .. warning::

..     Not yet implemented
