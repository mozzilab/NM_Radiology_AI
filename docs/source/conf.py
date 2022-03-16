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
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))
import sphinx_rtd_theme

# -- Project information -----------------------------------------------------

project = 'NM Results Management'
copyright = '2022, NM HIT'
author = 'NM HIT'

# The full version, including alpha/beta/rc tags
release = '1.10'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.imgmath',
    'sphinx.ext.ifconfig',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx_copybutton',
    'sphinx.ext.autosectionlabel',
    "sphinx_panels",
    'sphinx.ext.viewcode',
    'nbsphinx',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# This must be the name of an image file (path relative to the configuration
# directory) that is the favicon of the docs. Modern browsers use this as
# the icon for tabs, windows and bookmarks. It should be a Windows-style
# icon file (.ico).
# html_favicon = "favicon.ico"

# Integrate github
html_context = {
    "display_github": True,                     # Integrate GitHub
    "github_user": "mozzilab",                  # Username
    "github_repo": "NM_Radiology_AI",           # Repo name
    "github_version": "main",                   # Version
    "conf_py_path": "/docs/source/",            # Path in the checkout to the docs root
}

html_show_sourcelink = True

# -- Extension interface -------------------------------------------------------

intersphinx_mapping = {
    'torch': ('https://pytorch.org/docs/stable/', None),
}

def fix_sig(app, what, name, obj, options, signature, return_annotation):
    """Remove -> None from annotation if that's what it is"""
    if return_annotation == "None":
        return (signature, "")
    else:
        return (signature, return_annotation) 
 
def setup(app):
    app.connect("autodoc-process-signature", fix_sig)
    app.add_css_file("css/custom.css")