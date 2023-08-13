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

# Insert the parent directory (where your code is) into the system path
sys.path.insert(0, os.path.abspath('..'))



# -- Project information -----------------------------------------------------

project = 'nifti2dicom'
copyright = '2023, Lalith Kumar Shiyam Sundar | Aaron Selfridge | Siqi Li'
author = 'Lalith Kumar Shiyam Sundar | Aaron Selfridge | Siqi Li'

# The full version, including alpha/beta/rc tags
release = '1.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

# Extensions
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
    'sphinx.ext.linkcode',
    'sphinx_rtd_dark_mode', # Add this line for direct linking to GitHub source
]

# Intersphinx mapping for external libraries
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Theme and static files
html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']

# GitHub linking for "Edit on Github" feature
html_context = {
    "display_github": True,
    "github_user": "LalithShiyam",
    "github_repo": "nifti2dicom",
    "github_version": "main",
    "conf_py_path": "/docs/",
}

# Function to resolve direct linking to GitHub source
def linkcode_resolve(domain, info):
    if domain != 'py':
        return None
    if not info['module']:
        return None
    filename = info['module'].replace('.', '/')
    return f"https://github.com/LalithShiyam/nifti2dicom/blob/main/{filename}.py"


html_theme_options = {
    "style_nav_header_background": "#343131",  # Optional: Change the navbar header color
    "dark_mode_theme": "darkly",  # Optional: Set the dark mode theme to "darkly"
}

html_css_files = [
    'custom.css',
]

html_logo = '_static/Nifti2dicom-logo.png'