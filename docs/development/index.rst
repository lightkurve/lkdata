.. _development:

Developing for lkdata
=====================

The *lkdata* package is a *lightkurve* module.
Lightkurve is an open-source, community driven set of modules.
We strongly encourage users to contribute and develop new features for Lightkurve.
These pages how to contribute Pull Requests and Issues on github for improvements.
These pages also discuss how to compile our documentation (including this page!).
Use the menu bar on the left to scroll through the docs or click below.

Developer documentation
-----------------------

.. toctree::
    :maxdepth: 1

    contributing.rst
    documentation.rst

Setting up a Development Environment
====================================

To set up a development environment for lkdata you can follow the steps below.

Fork lkdata's GitHub repository
-------------------------------

The first step is to create a copy of lkdata's GitHub repository by logging into GitHub, browsing to
`https://github.com/lightkurve/lkdata <https://github.com/lightkurve/lkdata>`_,
and clicking the ``Fork`` button in the top right corner.

Clone the fork to your computer
-------------------------------

Head to a local directory of your choice and download your fork:

.. code-block:: bash

    $ git clone https://github.com/YOUR-GITHUB-USERNAME/lkdata.git


.. _install-dev-env:

Install the development environment
-----------------------------------

Lightkurve uses the `poetry <https://python-poetry.org/>`_ package to create an isolated development
environment which you can use to modify and test changes to the source code without interfering with
your existing Python environment.

You can set up the environment as follows:

.. code-block:: bash

    $ cd lkdata
    $ python -m pip install poetry
    $ make install

A key advantage of the development environment is that any changes you make to the Lightkurve source
code will be reflected right away, i.e., there is no need to re-install Lightkurve or the environment
after every change.

To run code in the development environment, you will need to prefix every Python command with
`poetry run`. For example:

.. code-block:: bash

    $ poetry run python YOUR-SCRIPT.py
    $ poetry run jupyter notebook
    $ poetry run pytest  # runs all the unit tests

You can find more details on the `poetry website <https://python-poetry.org/>`_,
and you can find additional examples of tasks developers commonly execute in the development
environment in Lightkurve's `Makefile <https://github.com/lightkurve/lkdata/blob/main/Makefile>`_.

.. note::

    The use of `poetry` is not required to prepare and propose modifications to Lightkurve.
    If you wish to do so, you can install the development version in your current
    Python environment as follows:

    .. code-block:: bash

        $ cd lkdata
        $ python -m pip install .

    In this scenario, you will have to re-run `pip install .` every time you make changes
    to the source code.

    To avoid this extra step, you have the option of creating a symbolic
    link from your environment's `site-packages` directory to the lightkurve source code tree
    as follows:

    .. code-block:: bash

        $ python -m pip uninstall lkdata
        $ python -m pip install --editable .  # creates the symbolic link

    This "editable install" method requires `pip` version `21.3` or higher.
    While this method of installing Lightkurve is not usually recommended, it can be useful
    when you wish to modify and test multiple different packages in a single environment.


Add a link to the main repository to your git environment
---------------------------------------------------------

To be able to pull in any recent changes, we need to tell your copy of lightkurve
where the upstream repository is located:

.. code-block:: bash

    $ git remote add upstream https://github.com/lightkurve/lkdata.git

To verify that everything is setup correctly, execute:

.. code-block:: bash

    $ git remote -v

You should see something like this:

.. code-block:: bash

    origin	https://github.com/YOUR-GITHUB-USERNAME/lkdata.git (fetch)
    origin	https://github.com/YOUR-GITHUB-USERNAME/lkdata.git (push)
    upstream	https://github.com/lightkurve/lkdata.git (fetch)
    upstream	https://github.com/lightkurve/lkdata.git (push)
