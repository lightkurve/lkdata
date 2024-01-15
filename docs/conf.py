# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
# Try to import the global configuration from sphinx-astropy

try:
    from sphinx_astropy.conf.v2 import *  # noqa: F403
except ImportError:
    print("sphinx_astropy not found, default configuration will be used.")


project = "lightkurve"
copyright = "2024, TESS Science Support Center"
author = "TESS Science Support Center"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx_autodoc_typehints",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx_automodapi.automodapi",
    "sphinx_gallery.gen_gallery",
    "nbsphinx",
    "numpydoc",
    "sphinx.ext.intersphinx",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_static_path = ["_static"]

html_theme_options = {
    "github_url": "https://github.com/lightkurve/lightkurve",
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://docs.scipy.org/doc/numpy/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
    "matplotlib": ("https://matplotlib.org", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "astropy": ("https://docs.astropy.org/en/latest/", None),
}

sphinx_gallery_conf = {
    "examples_dirs": "tutorials",  # path to your example scripts
    "gallery_dirs": "auto_examples",  # path to where to save gallery generated output
}

# rst_epilog = """
# .. |version| replace:: 0.1.0
# """
