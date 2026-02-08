#!/usr/bin/env python3
"""Configuration file for the Sphinx documentation builder."""

import datetime
import os
import sys

# -- Path setup --------------------------------------------------------------

ON_READTHEDOCS = os.environ.get("READTHEDOCS") == "True"
if not ON_READTHEDOCS:
    sys.path.insert(0, os.path.abspath("../../src"))

import gradient_free_optimizers  # noqa: E402

# -- Project information -----------------------------------------------------
current_year = datetime.datetime.now().year
project = "Gradient-Free-Optimizers"
project_copyright = f"2019 - {current_year} (MIT License)"
author = "Simon Blanke"

CURRENT_VERSION = f"v{gradient_free_optimizers.__version__}"

if ON_READTHEDOCS:
    READTHEDOCS_VERSION = os.environ.get("READTHEDOCS_VERSION")
    if READTHEDOCS_VERSION == "latest":
        CURRENT_VERSION = "main"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.autosectionlabel",
    "numpydoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_issues",
    "sphinx.ext.doctest",
]

myst_enable_extensions = ["colon_fence"]

templates_path = ["_templates"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

master_doc = "index"
language = "en"

exclude_patterns = [
    "_build",
    ".ipynb_checkpoints",
    "Thumbs.db",
    ".DS_Store",
]

add_module_names = False
toc_object_entries_show_parents = "hide"
pygments_style = "sphinx"

numpydoc_show_class_members = True
numpydoc_class_members_toctree = False
numpydoc_validation_checks = set()

autosummary_generate = True

autodoc_default_options = {
    "members": True,
    "inherited-members": True,
    "member-order": "bysource",
}

add_function_parentheses = False

suppress_warnings = [
    "myst.mathjax",
    "docutils",
    "toc.not_included",
    "autodoc.import_object",
    "autosectionlabel",
    "ref",
]

show_warning_types = True
issues_github_path = "SimonBlanke/Gradient-Free-Optimizers"


def linkcode_resolve(domain, info):
    """Return URL to source code."""

    def find_source():
        obj = sys.modules[info["module"]]
        for part in info["fullname"].split("."):
            obj = getattr(obj, part)
        import inspect

        fn = inspect.getsourcefile(obj)
        fn = os.path.relpath(
            fn, start=os.path.dirname(gradient_free_optimizers.__file__)
        )
        source, lineno = inspect.getsourcelines(obj)
        return fn, lineno, lineno + len(source) - 1

    if domain != "py" or not info["module"]:
        return None
    try:
        filename = "src/gradient_free_optimizers/%s#L%d-L%d" % find_source()
    except Exception:
        filename = info["module"].replace(".", "/") + ".py"
    return f"https://github.com/SimonBlanke/Gradient-Free-Optimizers/blob/{CURRENT_VERSION}/{filename}"


# -- Options for HTML output -------------------------------------------------

html_theme = "pydata_sphinx_theme"

html_theme_options = {
    "logo": {
        "image_light": "_static/images/gradient_logo_ink.png",
        "image_dark": "_static/images/gradient_logo_ink.png",
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/SimonBlanke/Gradient-Free-Optimizers",
            "icon": "fab fa-github",
        },
        {
            "name": "Star on GitHub",
            "url": "https://github.com/SimonBlanke/Gradient-Free-Optimizers/stargazers",
            "icon": "fa-regular fa-star",
        },
    ],
    "show_prev_next": False,
    "use_edit_page_button": False,
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "show_toc_level": 2,
    "secondary_sidebar_items": ["page-toc", "sourcelink"],
}

html_title = "Gradient-Free-Optimizers"
html_context = {
    "github_user": "SimonBlanke",
    "github_repo": "Gradient-Free-Optimizers",
    "github_version": "master",
    "doc_path": "docs/source/",
}

html_sidebars = {
    "**": ["sidebar-nav-bs.html"],
    "index": [],
    "get_started/index": [],
    "installation": [],
    "search": [],
}

html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
html_show_sourcelink = False

htmlhelp_basename = "gfodoc"

# -- Options for LaTeX output ------------------------------------------------

latex_elements = {}

latex_documents = [
    (
        master_doc,
        "gradient_free_optimizers.tex",
        "Gradient-Free-Optimizers Documentation",
        "Simon Blanke",
        "manual",
    ),
]

man_pages = [(master_doc, "gradient_free_optimizers", "GFO Documentation", [author], 1)]

texinfo_documents = [
    (
        master_doc,
        "gradient_free_optimizers",
        "Gradient-Free-Optimizers Documentation",
        author,
        "gradient_free_optimizers",
        "A collection of gradient-free optimization algorithms.",
        "Miscellaneous",
    ),
]


def setup(app):
    """Set up sphinx builder."""

    def adds(pth):
        print("Adding stylesheet: %s" % pth)  # noqa: T201
        app.add_css_file(pth)

    adds("fields.css")


# -- Extension configuration -------------------------------------------------

intersphinx_mapping = {
    "python": (f"https://docs.python.org/{sys.version_info.major}", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "scikit-learn": ("https://scikit-learn.org/stable/", None),
}

todo_include_todos = False

copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
copybutton_line_continuation_character = "\\"

# -- Algorithm counts --------------------------------------------------------

# Count algorithms from __all__
_n_algorithms = len(gradient_free_optimizers.__all__)

# Categories
_n_local = 5  # Hill Climbing variants, Simulated Annealing, Downhill Simplex
_n_global = 8  # Random, Grid, Pattern, Powell, Lipschitz, DIRECT, etc.
_n_population = 6  # PSO, GA, ES, DE, Spiral, Parallel Tempering
_n_smbo = 4  # Bayesian, TPE, Forest, Ensemble

rst_epilog = f"""
.. |version| replace:: {gradient_free_optimizers.__version__}
.. |current_year| replace:: {current_year}
.. |n_algorithms| replace:: {_n_algorithms}
.. |n_local| replace:: {_n_local}
.. |n_global| replace:: {_n_global}
.. |n_population| replace:: {_n_population}
.. |n_smbo| replace:: {_n_smbo}
"""
