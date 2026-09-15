"""Documentation configuration; public API docs use the installed package."""
from nifti2dicom import __version__

project = "nifti2dicom"
author = "Lalith Kumar Shiyam Sundar, Aaron Selfridge, Siqi Li"
copyright = "2026, " + author
release = __version__

extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx.ext.viewcode"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_theme = "sphinx_rtd_theme"
