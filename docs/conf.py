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
import pytorch_sphinx_theme
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))


# -- Project information -----------------------------------------------------

project = 'KoshurRecognition'
copyright = '2020, murtaza, wajid'
author = 'murtaza, wajid'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc', 'sphinx_rtd_theme', 'sphinx.ext.mathjax',
              #   'sphinx.ext.autosummary',
              #   'sphinx.ext.doctest',
              #   'sphinx.ext.intersphinx',
              #   'sphinx.ext.todo',
              #   'sphinx.ext.coverage',
              #   'sphinx.ext.napoleon',
              #   'sphinx.ext.viewcode',
              #   'sphinxcontrib.katex',
              #   'sphinx.ext.autosectionlabel',
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
# html_theme = 'alabaster'
# html_theme = 'yeen'
# html_theme = "sphinx_rtd_theme"
html_theme = 'pytorch_sphinx_theme'
html_theme_path = [pytorch_sphinx_theme.get_html_theme_path()]


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
# html_style = 'css/theme.css'
# html_css_files = [
#     'css/theme.css',
#     'css/gallery.css'
# ]
html_logo = 'logo.png'
html_theme_options = {
    'collapse_navigation': True,
}

# pygments_style = 'sphinx'

# todo_include_todos = True

autodoc_inherit_docstrings = False

# def setup(app):
#     app.add_css_file('css/custom.css')
