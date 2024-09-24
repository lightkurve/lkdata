# Documentation Guide for `Lightkurve`'s `lkdata`

Welcome to the documentation guide for `Lightkurve`'s `lkdata`. This README will provide you with the guidelines and tools necessary to contribute to the project's documentation.

## Writing Style

- **Docstring Format**: We use the Numpy docstring format. Refer to the [Numpy Documentation Guide](https://numpydoc.readthedocs.io/en/latest/format.html) for details.
- **Narrative Documentation**: Follow the [Astropy Documentation Style Guide](https://docs.astropy.org/en/stable/development/style-guide.html#style-guide). This guide helps maintain consistency and clarity in our narrative documentation.

## Setting Up Your Environment

Before working on the documentation, ensure you have the necessary tools and dependencies installed. We use [Poetry](https://python-poetry.org/docs/) for dependency management.

1. Install dependencies:
   ```bash
   poetry install
   ```

2. Use the `Makefile` to run commands such as `make html` and `make serve` (see below).

3. If you are using your own commands, make sure to use `poetry run` before each command to use the poetry environment. For example, if you wanted to run `Python`, you'd use `poetry run python`.

## Compiling the Documentation

To compile the documentation into HTML format:

```bash
make html
```

This command generates HTML files in `_build/html`.

## Serving Documentation Locally

To view the documentation with live updates as you write:

```bash
make serve
```

This will start a local server. You can view the documentation in your browser at `http://localhost:8001`.

## Stopping the Local Server

If you need to stop the local server, use:

```bash
make stop-serve
```

## Adding Tutorials

1. Place your tutorial files (either `.rst` or `.ipynb`) in the `examples` directory.
2. Ensure they follow the same writing and formatting guidelines.
3. The Sphinx build process will automatically include them in the documentation.

## Making Your Pages Appear

To add a new page:

1. Create your `.rst` file in the appropriate directory.
2. Update the relevant `index.rst` file to include your new file in the toctree.


## Writing Style and Conventions

### Numpy Docstring Format Example

When documenting Python functions, classes, and methods, we follow the Numpy docstring format. Here's an example:

```python
def example_function(param1, param2):
    """
    A brief description of what the function does.

    Parameters
    ----------
    param1 : int
        Description of param1.
    param2 : str
        Description of param2.

    Returns
    -------
    bool
        Description of the return value.
    """
    return True
```

## Using Intersphinx

### Adding a New Target Project

To link to other projects' documentation, like Numpy, add the following to your `conf.py` under `intersphinx_mapping`:

```python
intersphinx_mapping = {
    'numpy': ('https://numpy.org/doc/stable/', None),
    'astropy': ('https://docs.astropy.org/en/stable/', None),
    # other mappings...
}
```

### Examples of Intersphinx References

- Referencing the Numpy documentation:

  ```rst
  See :ref:`numpy:reference/arrays.ndarray` for details on NumPy arrays.
  ```

- Linking to a specific part of the Astropy documentation:

  ```rst
  More on units in Astropy: :ref:`astropy:units-index`.
  ```

## Using ReST Substitutions

### Adding a Substitution in `conf.py`

Define your substitutions in the `rst_epilog` section in `conf.py`:

```python
rst_epilog = """
.. |ProjectName| replace:: Your Project Name
"""
```

### Using a Substitution in the Docs

In your `.rst` files, use the substitution like this:

```rst
Welcome to |ProjectName| documentation!
```

## Writing in ReStructuredText

For guidelines on writing in ReStructuredText (`.rst`), refer to the [Sphinx ReST Primer](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html).

