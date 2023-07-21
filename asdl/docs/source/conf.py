# Configuration file for the Sphinx documentation builder.

import sys
import os
sys.path.insert(0, os.path.abspath('../../'))

import asdl

# -- Project information

project = 'ASDL'
copyright = '2022, Kazuki Osawa'
author = 'Kazuki Osawa'

release = asdl.__version__
version = asdl.__version__

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

add_module_names = False

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'
