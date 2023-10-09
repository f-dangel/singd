"""Configuration file for ``mkdocs-gallery`` export of examples."""

import os
import re

from mkdocs_gallery.gen_gallery import DefaultResetArgv
from mkdocs_gallery.sorting import FileNameSortKey

# See
# https://sphinx-gallery.github.io/stable/_modules/sphinx_gallery/gen_gallery.html
# for options
conf = {
    "reset_argv": DefaultResetArgv(),
    "filename_pattern": f"{re.escape(os.sep)}examples",
    "abort_on_example_error": True,
    # order examples according to file name
    "within_subsection_order": FileNameSortKey,
}
