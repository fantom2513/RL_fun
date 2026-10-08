"""Static checks of the lab web assets: local references only, resolvable ES modules."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from rl_fun.lab.server import WEB_DIR

SVG_NAMESPACE = "http://www.w3.org/2000/svg"
WEB_FILES = sorted(p for p in WEB_DIR.iterdir() if p.suffix in {".html", ".css", ".js"})
EXPECTED = (
    "index.html",
    "style.css",
    "app.js",
    "api.js",
    "track_view.js",
    "params_form.js",
    "charts.js",
    "network_view.js",
)

_ATTRIBUTE = re.compile(r"""\b(?:src|href)\s*=\s*["']([^"']+)["']""", re.IGNORECASE)
_CSS_REFERENCE = re.compile(r"""(?:@import\s+(?:url\()?|url\()\s*["']?([^"')\s]+)""")
_JS_IMPORT = re.compile(
    r"""(?:\bimport\s*(?:[^'"()]*?\bfrom\s*)?|\bexport\s[^'"()]*?\bfrom\s*|\bimport\s*\(\s*)"""
    r"""["']([^"']+)["']"""
)
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_LINE_COMMENT = re.compile(r"(?m)(?:^|(?<=[\s;{}),]))//.*$")


def _strip_comments(path: Path, text: str) -> str:
    if path.suffix == ".html":
        return _HTML_COMMENT.sub("", _BLOCK_COMMENT.sub("", text))
    text = _BLOCK_COMMENT.sub("", text)
    return _LINE_COMMENT.sub("", text) if path.suffix == ".js" else text


def _resolve(page_url: str, reference: str) -> Path:
    """Map a reference found in a file served at `page_url` to a path below web/."""
    assert not re.match(r"^[a-z][a-z0-9+.-]*:", reference, re.IGNORECASE), reference
    assert not reference.startswith("//"), reference
    base = page_url.rsplit("/", 1)[0] + "/"
    absolute = reference if reference.startswith("/") else base + reference
    segments: list[str] = []
    for part in absolute.split("#")[0].split("?")[0].split("/"):
        if part == "..":
            assert segments, f"{reference} escapes the root"
            segments.pop()
        elif part not in ("", "."):
            segments.append(part)
    assert segments[0] == "static", f"{reference} is not served from /static/"
    return WEB_DIR.joinpath(*segments[1:])


def _url_of(path: Path) -> str:
    return "/" if path.name == "index.html" else f"/static/{path.relative_to(WEB_DIR).as_posix()}"


def test_expected_web_files_exist() -> None:
    missing = [name for name in EXPECTED if not (WEB_DIR / name).is_file()]

    assert missing == []


def test_index_html_references_only_existing_local_files() -> None:
    html = _strip_comments(WEB_DIR / "index.html", (WEB_DIR / "index.html").read_text("utf-8"))

    references = _ATTRIBUTE.findall(html)
    targets = [_resolve("/", reference) for reference in references]

    assert references, "index.html should load its stylesheet and script"
    assert [t for t in targets if not t.is_file()] == []
    assert {t.name for t in targets} >= {"style.css", "app.js"}


@pytest.mark.parametrize("path", [p for p in WEB_FILES if p.suffix == ".js"], ids=lambda p: p.name)
def test_js_module_imports_resolve_to_existing_files(path: Path) -> None:
    source = _strip_comments(path, path.read_text("utf-8"))

    specifiers = _JS_IMPORT.findall(source)
    targets = [_resolve(_url_of(path), specifier) for specifier in specifiers]

    assert all(s.startswith(("./", "../", "/static/")) for s in specifiers)
    assert [t for t in targets if not t.is_file()] == []


def test_app_module_imports_form_charts_and_network_modules() -> None:
    path = WEB_DIR / "app.js"
    source = _strip_comments(path, path.read_text("utf-8"))

    imported = {_resolve(_url_of(path), s).name for s in _JS_IMPORT.findall(source)}

    assert {"api.js", "track_view.js", "params_form.js", "charts.js", "network_view.js"} <= imported


def test_network_view_module_is_preloaded_by_index_html() -> None:
    html = (WEB_DIR / "index.html").read_text("utf-8")

    assert "/static/network_view.js" in html


@pytest.mark.parametrize("path", [p for p in WEB_FILES if p.suffix == ".css"], ids=lambda p: p.name)
def test_css_references_resolve_to_existing_files(path: Path) -> None:
    css = _strip_comments(path, path.read_text("utf-8"))

    targets = [
        _resolve(_url_of(path), ref)
        for ref in _CSS_REFERENCE.findall(css)
        if not ref.startswith("data:")
    ]

    assert [t for t in targets if not t.is_file()] == []


@pytest.mark.parametrize("path", WEB_FILES, ids=lambda p: p.name)
def test_web_files_contain_no_external_urls(path: Path) -> None:
    text = _strip_comments(path, path.read_text("utf-8")).replace(SVG_NAMESPACE, "")

    assert re.findall(r"https?://", text) == []
    assert re.findall(r"""["'(]//[a-z0-9.-]+\.[a-z]""", text) == []
