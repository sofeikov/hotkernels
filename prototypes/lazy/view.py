"""The page that draws recorded graphs: one HTML file, opened in a browser.

The page is `templates/graph.html.j2` with three things put into it: the
graphs as JSON, the stylesheet `templates/graph.css` and the script
`templates/graph.js`. The script does all the drawing: it folds groups, asks
a layout library where everything goes, and draws SVG.

This prototype's page loads the layout library, ELK, from a CDN, so it needs
a network connection when it is opened. The page of the package is to carry
the library inside it; see docs/lazy-graph-page-design.md.
"""
from __future__ import annotations

import html
import json
import os
import tempfile
import webbrowser
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

# Where this package keeps its templates.
TEMPLATES = Path(__file__).resolve().parent / "templates"
# The page, and the stylesheet and script that are put into it whole.
PAGE_TEMPLATE, STYLE_FILE, SCRIPT_FILE = "graph.html.j2", "graph.css", "graph.js"
# The layout library the page loads: ELK's layered layout, compiled to
# JavaScript. An exact version, so the page does not change under a reader.
LAYOUT_LIBRARY_URL = "https://cdn.jsdelivr.net/npm/elkjs@0.9.3/lib/elk.bundled.js"
# Height of the frame a notebook shows the page in, in pixels.
NOTEBOOK_FRAME_HEIGHT = 640


def page(data: dict[str, Any], title: str) -> str:
    """Build the page for some graphs.

    Args:
        data: The graphs as built-in types, as `record._page_data` gives them.
        title: The page's title.

    Returns:
        The HTML of the page. It holds the data, the stylesheet and the
        script, and refers to nothing but the layout library.
    """
    environment = Environment(loader=FileSystemLoader(TEMPLATES), undefined=StrictUndefined, autoescape=False)
    # "</" would end the script element the JSON sits in, wherever it stands,
    # including inside a string of the user's code.
    embedded = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    return environment.get_template(PAGE_TEMPLATE).render(
        title=html.escape(title), data=embedded, library=LAYOUT_LIBRARY_URL,
        style=(TEMPLATES / STYLE_FILE).read_text(), script=(TEMPLATES / SCRIPT_FILE).read_text())


def save(text: str, path: str | os.PathLike) -> Path:
    """Write a page to a file.

    Args:
        text: The page's HTML.
        path: Where to write it.

    Returns:
        The path written.
    """
    target = Path(path)
    target.write_text(text, encoding="utf-8")
    return target


def show(text: str) -> Path:
    """Write a page to a temporary file and open it in the browser.

    Args:
        text: The page's HTML.

    Returns:
        The path of the file. It is left in place: the browser reads it after this returns.
    """
    with tempfile.NamedTemporaryFile("w", suffix=".html", prefix="lazy-graphs-", delete=False, encoding="utf-8") as file:
        file.write(text)
    webbrowser.open(Path(file.name).as_uri())
    return Path(file.name)


def framed(text: str) -> str:
    """Wrap a page so that a notebook can show it under a cell.

    Args:
        text: The page's HTML.

    Returns:
        An `iframe` that carries the whole page in its `srcdoc`, so the
        page's styles and script stay apart from the notebook's.
    """
    return (f'<iframe srcdoc="{html.escape(text, quote=True)}" '
            f'style="width: 100%; height: {NOTEBOOK_FRAME_HEIGHT}px; border: 0"></iframe>')
