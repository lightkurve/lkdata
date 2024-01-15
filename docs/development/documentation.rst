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
    `docs.lightkurve.org <https://docs.lightkurve.org/>`_ .

Building the *lightkurve* documentation requires `sphinx` and a few extra packages. We recommend using `poetry` to install the development dependencies::

    $  poetry install

To make a clean directory for the docs use::

    $ cd docs
    $ make clean

To build the documentation in HTML format, execute::

    $ cd docs
    $ make html

.. warning::

    Christina needs to add information here about how to use make serve

Finally, if you have write permission to *lightkurve*'s GitHub repository,
you can upload the documentation to the web server using::

    $ make upload

.. warning::

    Christina needs to implement this
