"""Sphinx configuration for the mmmJAX documentation."""

import re

project = "mmmJAX"
copyright = "2026, Jordan DeKlerk"
author = "Jordan DeKlerk"

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_design",
    "IPython.sphinxext.ipython_directive",
    "IPython.sphinxext.ipython_console_highlighting",
    "sphinx_immaterial",
]

exclude_patterns = []
templates_path = ["_templates"]
mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-mml-chtml.js"

html_theme = "sphinx_immaterial"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_js_files = [("copybutton-shim.js", {"priority": 200}), "header-title-link.js"]
html_title = "mmmJAX"
html_logo = "_static/mmmjax-logo.svg"
html_favicon = "_static/favicon.ico"

html_theme_options = {
    "font": {"text": "PT Sans", "code": "Fira Mono"},
    "repo_url": "https://github.com/jordandeklerk/mmmJAX",
    "repo_name": "mmmJAX",
    "icon": {"repo": "fontawesome/brands/git-alt"},
    "features": [
        "header.autohide",
        "navigation.instant",
        "navigation.tabs",
        "navigation.tabs.sticky",
        "navigation.path",
        "navigation.top",
        "navigation.footer",
        "navigation.tracking",
        "announce.dismiss",
        "search.highlight",
        "search.share",
        "toc.follow",
    ],
    "toc_title": "On this page",
    "palette": [
        {
            "media": "(prefers-color-scheme)",
            "toggle": {"icon": "material/brightness-auto", "name": "Switch to light mode"},
        },
        {
            "media": "(prefers-color-scheme: light)",
            "scheme": "default",
            "primary": "white",
            "accent": "teal",
            "toggle": {"icon": "material/weather-sunny", "name": "Switch to dark mode"},
        },
        {
            "media": "(prefers-color-scheme: dark)",
            "scheme": "slate",
            "primary": "black",
            "accent": "teal",
            "toggle": {"icon": "material/weather-night", "name": "Switch to system preference"},
        },
    ],
}

copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "none"
napoleon_numpy_docstring = True
napoleon_google_docstring = False
# Let autodoc index each field once while keeping its description in the class docstring
napoleon_use_ivar = True
# Examples sections become admonitions so the theme can style them as example boxes
napoleon_use_admonition_for_examples = True

# Recolor the theme's example admonition to the brand forest green
sphinx_immaterial_custom_admonitions = [
    {"name": "example", "override": True, "icon": "material/code-braces", "color": (7, 66, 48)},
]

# Keep example output compact without rounding the values used in calculations
ipython_execlines = [
    "import numpy as np",
    "np.set_printoptions(precision=1, floatmode='fixed', suppress=True)",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "jax": ("https://docs.jax.dev/en/latest", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

myst_enable_extensions = ["linkify", "colon_fence", "dollarmath"]
myst_heading_anchors = 3
# Guide pages run only when they change. They load stored fits instead of sampling.
nb_execution_mode = "cache"
nb_execution_raise_on_error = True
nb_execution_timeout = 600
nb_output_stderr = "remove"


def _format_signature_defaults(app, what, name, obj, options, signature, return_annotation):
    """Render type and function defaults as valid Python in API signatures."""
    if signature is not None:
        signature = signature.replace("<class 'float'>", "float")
        signature = re.sub(r"<function ([\w.]+)(?: at 0x[0-9a-fA-F]+)?>", r"\1", signature)
    return signature, return_annotation


def _open_examples_boxes(app, doctree):
    """Render napoleon's Examples admonitions as open, collapsible example boxes.

    Boxes that set their own collapsible state keep it.
    """
    from docutils import nodes

    for node in doctree.findall(nodes.admonition):
        if "example" in node["classes"] and node.get("collapsible") is None:
            node["collapsible"] = "open"


def setup(app):
    """Register API signature formatting and example box styling."""
    app.connect("autodoc-process-signature", _format_signature_defaults)
    app.connect("doctree-read", _open_examples_boxes)
