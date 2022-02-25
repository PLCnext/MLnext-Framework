# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
# -- Path setup --------------------------------------------------------------
# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from datetime import date

sys.path.insert(0, os.path.abspath('../..'))


if 'READTHEDOCS' in os.environ:
    import glob

    if glob.glob('../changelog/*.*.rst'):
        print('-- Found changes; running towncrier --', flush=True)
        import subprocess

        subprocess.run(
            ['towncrier', '--yes', '--date', 'not released yet'], cwd='..',
            check=True
        )


import mlnext  # noqa

# -- Project information -----------------------------------------------------


project = mlnext.__title__
author = mlnext.__author__
copyright = f'{date.today().year}, {author}'


# The full version, including alpha/beta/rc tags
version = mlnext.__version__
release = version

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'myst_parser',
    'sphinx_copybutton',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx_lfs_content'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'furo'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_theme_options = {
    'light_logo': 'digital-factory-logo.png',
    'dark_logo': 'digital-factory-dark-logo.png'
}
html_favicon = '_static/favicon.ico'

html_show_sourcelink = False

# -- Extensions --------------------------------------------------------------

autosectionlabel_prefix_document = True
autosectionlabel_maxdepth = 2
autosummary_generate = True
autodoc_inherit_docstrings = False
set_type_checking_flag = True
