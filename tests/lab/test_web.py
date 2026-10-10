"""Static checks of the lab web assets: local references only, resolvable ES modules."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from rl_fun.lab.server import WEB_DIR

SVG_NAMESPACE = "http://www.w3.org/2000/svg"
WEB_FILES = sorted(
    p for p in WEB_DIR.rglob("*") if p.is_file() and p.suffix in {".html", ".css", ".js"}
)


def _rel(path: Path) -> str:
    return path.relative_to(WEB_DIR).as_posix()


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


def _page_references(html: str) -> list[str]:
    """src/href values that point at files; same-page fragments such as `#/lab` are routes."""
    return [r for r in _ATTRIBUTE.findall(html) if not r.startswith("#")]


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

    references = _page_references(html)
    targets = [_resolve("/", reference) for reference in references]

    assert references, "index.html should load its stylesheet and script"
    assert [t for t in targets if not t.is_file()] == []
    assert {t.name for t in targets} >= {"style.css", "app.js"}


@pytest.mark.parametrize("path", [p for p in WEB_FILES if p.suffix == ".html"], ids=_rel)
def test_html_pages_reference_only_existing_files_below_static(path: Path) -> None:
    html = _strip_comments(path, path.read_text("utf-8"))

    targets = [_resolve(_url_of(path), reference) for reference in _page_references(html)]

    assert [t for t in targets if not t.is_file()] == []


def test_web_tree_scan_includes_subdirectories() -> None:
    nested = [p for p in WEB_DIR.rglob("*") if p.is_file() and p.parent != WEB_DIR]

    assert {p for p in nested if p.suffix in {".html", ".css", ".js"}} <= set(WEB_FILES)


@pytest.mark.parametrize("path", [p for p in WEB_FILES if p.suffix == ".js"], ids=_rel)
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


@pytest.mark.parametrize("path", [p for p in WEB_FILES if p.suffix == ".css"], ids=_rel)
def test_css_references_resolve_to_existing_files(path: Path) -> None:
    css = _strip_comments(path, path.read_text("utf-8"))

    targets = [
        _resolve(_url_of(path), ref)
        for ref in _CSS_REFERENCE.findall(css)
        if not ref.startswith("data:")
    ]

    assert [t for t in targets if not t.is_file()] == []


@pytest.mark.parametrize("path", WEB_FILES, ids=_rel)
def test_web_files_contain_no_external_urls(path: Path) -> None:
    text = _strip_comments(path, path.read_text("utf-8")).replace(SVG_NAMESPACE, "")

    assert re.findall(r"https?://", text) == []
    assert re.findall(r"""["'(]//[a-z0-9.-]+\.[a-z]""", text) == []


# ---- design system: tokens, fonts, icons (DESIGN.md sections 5, 11.2, 11.3) -----------------

REPO = Path(__file__).resolve().parents[2]
DESIGN_DIR = REPO / "docs" / "design" / "lab-ui"
PROTOTYPE_FONTS = DESIGN_DIR / "prototype" / "assets" / "fonts"
CSS_DIR = WEB_DIR / "css"
LAYER_ORDER = ("tokens", "base", "layout", "components", "screens", "utilities")
FILE_LAYERS = {
    "tokens.css": {"tokens"},
    "base.css": {"base", "utilities"},
    "layout.css": {"layout"},
    "components.css": {"components"},
    "screens.css": {"screens"},
}
LEGACY_STYLESHEET = WEB_DIR / "style.css"
CSS_FILES = sorted(CSS_DIR.glob("*.css")) or [CSS_DIR / "tokens.css"]
_COLOUR_LITERAL = re.compile(r"#[0-9a-fA-F]{3,8}\b|\b(?:rgba?|hsla?)\(")
_VAR_USE = re.compile(r"var\(\s*(--[\w-]+)")
_PROPERTY_DEFINITION = re.compile(r"(?:^|[;{\s\"'])(--[\w-]+)\s*:")
_SET_PROPERTY = re.compile(r"""setProperty\(\s*['"](--[\w-]+)['"]""")
_ICON_REFERENCES = (
    re.compile(r"icons\.svg#i-([a-z0-9-]+)"),
    re.compile(r"""\bicon\(\s*['"]([a-z0-9-]+)['"]"""),
    re.compile(r"""\bdata-icon\s*=\s*["']([a-z0-9-]+)["']"""),
    re.compile(r"""\bicon\s*=\s*["']([a-z0-9-]+)["']"""),
)


def _css(path: Path) -> str:
    return _BLOCK_COMMENT.sub("", path.read_text("utf-8"))


def _blocks(text: str) -> list[tuple[str, str | None]]:
    """Split CSS into top-level (prelude, body) items; statements (`@layer a, b;`) have None."""
    items: list[tuple[str, str | None]] = []
    i = 0
    while True:
        match = re.compile(r"[{;]").search(text, i)
        if match is None:
            break
        j = match.start()
        if text[j] == ";":
            items.append((text[i:j].strip(), None))
            i = j + 1
            continue
        depth, k = 1, j + 1
        while depth:
            depth += {"{": 1, "}": -1}.get(text[k], 0)
            k += 1
        items.append((text[i:j].strip(), text[j + 1 : k - 1]))
        i = k
    return items


def _rules(text: str, context: tuple[str, ...] = ()) -> list[tuple[tuple[str, ...], str, str]]:
    """Flatten CSS into (enclosing at-rules, selector, declarations) for every style rule."""
    rules: list[tuple[tuple[str, ...], str, str]] = []
    for prelude, body in _blocks(text):
        if body is None:
            continue
        head = re.sub(r"\s+", " ", prelude)
        if head.startswith(("@media", "@layer", "@supports")):
            rules.extend(_rules(body, (*context, head)))
        elif not head.startswith("@keyframes"):
            rules.append((context, head, body))
    return rules


def _declarations(body: str) -> dict[str, str]:
    pairs = (d.split(":", 1) for d in body.split(";") if ":" in d)
    return {name.strip(): value.strip() for name, value in pairs}


def _normal(value: str) -> str:
    return re.sub(r"(?<![\d.])0\.", ".", re.sub(r"\s+", "", value.lower()))


def _token_blocks() -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """Return the dark default, the explicit light and the system-light token blocks."""
    dark: dict[str, str] = {}
    light: dict[str, str] = {}
    system: dict[str, str] = {}
    for context, selector, body in _rules(_css(CSS_DIR / "tokens.css")):
        media = [c for c in context if c.startswith("@media")]
        if selector == ":root" and not media:
            dark.update(_declarations(body))
        elif selector == ':root[data-theme="light"]' and not media:
            light.update(_declarations(body))
        elif selector == ':root:not([data-theme="dark"])' and media == [
            "@media (prefers-color-scheme: light)"
        ]:
            system.update(_declarations(body))
    return dark, light, system


def _design_token_table() -> list[tuple[str, str, str]]:
    text = (DESIGN_DIR / "DESIGN.md").read_text("utf-8")
    row = re.compile(r"^\| `(--[\w-]+)` \| `([^`]+)` \| `([^`]+)` \|", re.MULTILINE)
    return row.findall(text)


def _design_run_colours() -> list[tuple[str, str, str]]:
    text = (DESIGN_DIR / "DESIGN.md").read_text("utf-8")
    row = re.compile(r"^\| (\d) \| `(#[0-9a-f]{6})` \| `(#[0-9a-f]{6})` \|", re.MULTILINE)
    return [(f"--run-{n}", dark, light) for n, dark, light in row.findall(text)]


def _new_css_files() -> list[Path]:
    return sorted([*CSS_DIR.glob("*.css"), *(WEB_DIR / "dev").glob("*.css")])


def _sprite_ids() -> set[str]:
    import xml.etree.ElementTree as ET

    root = ET.parse(WEB_DIR / "icons.svg").getroot()
    return {s.get("id", "") for s in root.iter(f"{{{SVG_NAMESPACE}}}symbol")}


def test_design_system_files_exist() -> None:
    names = [
        "css/tokens.css",
        "css/base.css",
        "icons.svg",
        "fonts/LICENSES.md",
        *(f"fonts/{p.name}" for p in PROTOTYPE_FONTS.glob("*.woff2")),
    ]

    assert [n for n in names if not (WEB_DIR / n).is_file()] == []


@pytest.mark.parametrize("source", sorted(PROTOTYPE_FONTS.glob("*.woff2")), ids=lambda p: p.name)
def test_bundled_fonts_are_the_vetted_unmodified_woff2_files(source: Path) -> None:
    data = (WEB_DIR / "fonts" / source.name).read_bytes()

    assert data[:4] == b"wOF2" and data == source.read_bytes()


def test_every_bundled_font_is_declared_by_a_font_face_rule() -> None:
    css = _css(CSS_DIR / "base.css")
    faces = [body for _, selector, body in _rules(css) if selector == "@font-face"]
    referenced = {
        _resolve("/static/css/base.css", ref).name
        for body in faces
        for ref in _CSS_REFERENCE.findall(body)
    }

    assert referenced == {p.name for p in (WEB_DIR / "fonts").glob("*.woff2")}
    assert all("font-display: swap" in body and "unicode-range" in body for body in faces)


def test_font_families_of_the_tokens_are_the_bundled_faces() -> None:
    dark, _, _ = _token_blocks()
    faces = {
        _declarations(body)["font-family"].strip("\"'")
        for _, selector, body in _rules(_css(CSS_DIR / "base.css"))
        if selector == "@font-face"
    }

    names = ("--font-display", "--font-ui", "--font-mono")
    firsts = {dark[n].split(",")[0].strip().strip("\"'") for n in names}

    assert firsts == faces == {"Unbounded Variable", "Onest Variable", "JetBrains Mono Variable"}


LICENCES = sorted((DESIGN_DIR / "licenses").glob("*.txt"))


@pytest.mark.parametrize("licence", LICENCES, ids=lambda p: p.name)
def test_font_and_icon_licences_are_shipped_verbatim(licence: Path) -> None:
    shipped = " ".join((WEB_DIR / "fonts" / "LICENSES.md").read_text("utf-8").split())

    assert " ".join(licence.read_text("utf-8").split()) in shipped


def test_tokens_file_declares_the_layer_order_first() -> None:
    items = _blocks(_css(CSS_DIR / "tokens.css"))

    assert items[0] == ("@layer " + ", ".join(LAYER_ORDER), None)


@pytest.mark.parametrize("path", CSS_FILES, ids=_rel)
def test_every_stylesheet_puts_all_rules_into_its_own_layer(path: Path) -> None:
    items = [item for item in _blocks(_css(path)) if not item[0].startswith("@layer tokens,")]
    layers = {re.sub(r"^@layer\s+", "", prelude) for prelude, _ in items}

    assert all(prelude.startswith("@layer ") and body is not None for prelude, body in items)
    assert layers <= FILE_LAYERS[path.name]


@pytest.mark.parametrize(("name", "dark", "light"), _design_token_table(), ids=lambda v: v)
def test_tokens_match_the_design_palette(name: str, dark: str, light: str) -> None:
    dark_block, light_block, system_block = _token_blocks()

    assert _normal(dark_block[name]) == _normal(dark)
    assert _normal(light_block.get(name, dark_block[name])) == _normal(light)
    assert _normal(system_block.get(name, dark_block[name])) == _normal(light)


@pytest.mark.parametrize(("name", "dark", "light"), _design_run_colours(), ids=lambda v: v)
def test_run_colours_match_the_design(name: str, dark: str, light: str) -> None:
    dark_block, light_block, system_block = _token_blocks()

    assert (dark_block[name], light_block[name], system_block[name]) == (dark, light, light)


def test_design_tables_were_found() -> None:
    assert len(_design_token_table()) >= 23 and len(_design_run_colours()) == 6


def test_world_and_motion_tokens_match_the_design() -> None:
    dark, _, _ = _token_blocks()
    expected = {
        "--scene-grass": "#7a8e9c",
        "--scene-road": "#4f616e",
        "--scene-line": "#fffffc",
        "--scene-curb": "#d62e3e",
        "--scene-curb-white": "#fafaf8",
        "--net-node-ring": "#3c4a52",
        "--net-node-neutral": "#eceef0",
        "--net-node-pos": "#5aff50",
        "--net-node-neg": "#fa3e52",
        "--net-edge-pos": "#28c83c",
        "--net-edge-neg": "#d63246",
        "--t-fast": "120ms",
        "--t-base": "180ms",
        "--t-panel": "240ms",
        "--t-screen": "320ms",
        "--z-hud": "10",
        "--z-sticky": "20",
        "--z-popover": "40",
        "--z-dialog": "60",
        "--z-toast": "80",
    }

    assert {name: dark.get(name) for name in expected} == expected


def test_light_scheme_is_identical_for_the_attribute_and_the_system_preference() -> None:
    dark, light, system = _token_blocks()

    assert light == system
    assert set(light) - {"color-scheme"} <= set(dark)


@pytest.mark.parametrize("path", _new_css_files() or [CSS_DIR / "base.css"], ids=_rel)
def test_colour_literals_live_only_in_the_tokens_file(path: Path) -> None:
    if path.name == "tokens.css":
        pytest.skip("the tokens file is where colours are defined")

    assert _COLOUR_LITERAL.findall(_css(path)) == []


@pytest.mark.parametrize("path", _new_css_files() or [CSS_DIR / "base.css"], ids=_rel)
def test_important_is_used_only_for_reduced_motion(path: Path) -> None:
    offending = [
        selector
        for context, selector, body in _rules(_css(path))
        if "!important" in body and not any("prefers-reduced-motion" in c for c in context)
    ]

    assert offending == []


def _defined_properties() -> set[str]:
    defined: set[str] = set()
    for path in _new_css_files():
        defined.update(_PROPERTY_DEFINITION.findall(_css(path)))
    for path in WEB_FILES:
        if path.suffix in {".html", ".js"}:
            text = path.read_text("utf-8")
            defined.update(_SET_PROPERTY.findall(text))
            for style in re.findall(r"""style\s*=\s*["']([^"']*)["']""", text):
                defined.update(_PROPERTY_DEFINITION.findall(" " + style))
    return defined


def _design_system_files() -> list[Path]:
    folders = ("css", "dev", "core", "ui", "views", "screens")
    return [p for p in WEB_FILES if p.relative_to(WEB_DIR).parts[0] in folders]


@pytest.mark.parametrize("path", _design_system_files() or [CSS_DIR / "base.css"], ids=_rel)
def test_every_custom_property_used_is_defined(path: Path) -> None:
    used = set(_VAR_USE.findall(_strip_comments(path, path.read_text("utf-8"))))

    assert sorted(used - _defined_properties()) == []


def test_legacy_stylesheet_does_not_override_design_tokens() -> None:
    dark, light, _ = _token_blocks()
    legacy = set(_PROPERTY_DEFINITION.findall(_css(LEGACY_STYLESHEET)))

    assert sorted(legacy & (set(dark) | set(light))) == []


def test_legacy_stylesheet_and_scripts_use_only_defined_properties() -> None:
    dark, _, _ = _token_blocks()
    legacy_css = _css(LEGACY_STYLESHEET)
    defined = set(dark) | set(_PROPERTY_DEFINITION.findall(legacy_css)) | {"--run"}
    used = set(_VAR_USE.findall(legacy_css))
    for path in WEB_DIR.glob("*.js"):
        used.update(re.findall(r"""get\(\s*['"](--[\w-]+)['"]""", path.read_text("utf-8")))
    used.update(_VAR_USE.findall((WEB_DIR / "index.html").read_text("utf-8")))

    assert sorted(used - defined) == []


def test_index_loads_the_layers_before_the_legacy_stylesheet() -> None:
    html = _strip_comments(WEB_DIR / "index.html", (WEB_DIR / "index.html").read_text("utf-8"))
    sheets = re.findall(r"""<link[^>]*rel=["']stylesheet["'][^>]*href=["']([^"']+)["']""", html)

    assert sheets[:2] == ["/static/css/tokens.css", "/static/css/base.css"]
    assert sheets[-1] == "/static/style.css"


def test_icon_sprite_is_a_local_symbol_sheet() -> None:
    text = (WEB_DIR / "icons.svg").read_text("utf-8")

    assert "Lucide" in text and "ISC" in text
    assert re.findall(r"https?://", text.replace(SVG_NAMESPACE, "")) == []
    assert "<script" not in text and "href=" not in text


def test_icon_sprite_holds_the_prototype_icon_set() -> None:
    prototype = (DESIGN_DIR / "prototype" / "assets" / "icons.js").read_text("utf-8")
    expected = set(re.findall(r'id=\\"(i-[a-z0-9-]+)\\"', prototype))

    assert len(expected) == 78
    assert _sprite_ids() == expected


def test_icon_sprite_holds_the_section_icons_of_the_rail() -> None:
    sections = {"flask-conical", "route", "trophy", "warehouse", "settings"}

    assert {f"i-{name}" for name in sections} <= _sprite_ids()


def test_every_icon_used_by_the_web_files_is_in_the_sprite() -> None:
    used = {
        f"i-{name}"
        for path in WEB_FILES
        for pattern in _ICON_REFERENCES
        for name in pattern.findall(path.read_text("utf-8"))
    }

    assert sorted(used - _sprite_ids()) == []


# ---- components and the dev showcase (DESIGN.md sections 8, 11.1, 11.6 task 3) ---------------

COMPONENT_FILES = (
    "css/components.css",
    "core/dom.js",
    "core/format.js",
    "ui/param.js",
    "ui/seg.js",
    "ui/stepper.js",
    "ui/toast.js",
    "dev/components.html",
    "dev/components.js",
    "dev/components.css",
)
CUSTOM_ELEMENTS = {
    "lab-param": "ui/param.js",
    "lab-seg": "ui/seg.js",
    "lab-stepper": "ui/stepper.js",
    "lab-toast-host": "ui/toast.js",
}
COMPONENT_VARIANTS = (
    "btn-primary", "btn-ghost", "btn-danger", "btn-icon", "btn-sm", "btn-lg", "is-loading",
    "seg-sm", "seg-mono", "seg-glass", "chip-dashed", "is-locked", "param-compact", "is-changed",
    "range", "range-scale", "tag-live", "tag-ok", "tag-warn", "tag-err", "tag-best", "tag-info",
    "dot-running", "dot-paused", "dot-finished", "dot-error", "toast-error", "popover-board",
    "menu-item", "kbd", "callout", "skeleton", "empty", "stars-lg", "cell-best", "board",
    "glass", "tpl-card", "learner-card", "track-card", "field", "input", "legend",
)
SHOWCASE = WEB_DIR / "dev" / "components.html"


def _design_components() -> list[str]:
    text = (DESIGN_DIR / "DESIGN.md").read_text("utf-8")
    section = text.split("## 8. Компоненты", 1)[1].split("\n## ", 1)[0]
    return re.findall(r"^\| [^|]*?`(?:[a-z]+)?\.([a-z-]+)`", section, re.MULTILINE)


def _showcase_text() -> str:
    return SHOWCASE.read_text("utf-8") + (WEB_DIR / "dev" / "components.js").read_text("utf-8")


def _component_selectors() -> str:
    return " ".join(selector for _, selector, _ in _rules(_css(CSS_DIR / "components.css")))


def test_component_files_exist() -> None:
    assert [n for n in COMPONENT_FILES if not (WEB_DIR / n).is_file()] == []


def test_design_component_table_was_found() -> None:
    assert len(_design_components()) >= 19


@pytest.mark.parametrize("name", _design_components() + list(COMPONENT_VARIANTS))
def test_components_stylesheet_styles_every_component(name: str) -> None:
    assert re.search(rf"\.{re.escape(name)}(?![\w-])", _component_selectors())


@pytest.mark.parametrize("name", _design_components() + list(COMPONENT_VARIANTS))
def test_showcase_renders_every_component(name: str) -> None:
    assert re.search(rf"(?<![\w-]){re.escape(name)}(?![\w-])", _showcase_text())


def test_component_selectors_have_no_ids() -> None:
    assert re.findall(r"#[a-zA-Z][\w-]*", _component_selectors()) == []


@pytest.mark.parametrize(("tag", "module"), CUSTOM_ELEMENTS.items())
def test_custom_elements_are_defined_in_light_dom(tag: str, module: str) -> None:
    source = (WEB_DIR / module).read_text("utf-8")

    assert re.search(rf"""customElements\.define\(\s*['"]{tag}['"]""", source)
    assert "attachShadow" not in source


@pytest.mark.parametrize("tag", CUSTOM_ELEMENTS)
def test_showcase_uses_every_custom_element(tag: str) -> None:
    assert f"<{tag}" in _showcase_text() or f"'{tag}'" in _showcase_text()


def test_showcase_loads_the_layers_and_not_the_legacy_stylesheet() -> None:
    html = SHOWCASE.read_text("utf-8")
    sheets = re.findall(r"""<link[^>]*rel=["']stylesheet["'][^>]*href=["']([^"']+)["']""", html)

    assert sheets == [
        "/static/css/tokens.css",
        "/static/css/base.css",
        "/static/css/components.css",
        "/static/dev/components.css",
    ]
    assert '<script type="module" src="/static/dev/components.js"></script>' in html


def test_showcase_switches_between_both_colour_schemes() -> None:
    script = (WEB_DIR / "dev" / "components.js").read_text("utf-8")

    assert "dataset.theme" in script and "'light'" in script and "'dark'" in script


def test_showcase_is_not_part_of_the_app_navigation() -> None:
    app_sources = [WEB_DIR / "index.html", *WEB_DIR.glob("*.js")]

    assert [p.name for p in app_sources if "dev/" in p.read_text("utf-8")] == []


_NODE = shutil.which("node")


def _node(script: str) -> str:
    assert _NODE is not None
    result = subprocess.run(
        [_NODE, "--input-type=module", "-e", script],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        check=True,
    )
    return result.stdout.strip()


@pytest.mark.skipif(_NODE is None, reason="Node is optional; it only runs the pure JS helpers")
def test_format_helpers_follow_the_russian_number_rules() -> None:
    url = (WEB_DIR / "core" / "format.js").as_uri()
    script = (
        f"import * as f from {json.dumps(url)};"
        "console.log(JSON.stringify([f.number(0.2, 2), f.number(-1.5, 1), f.number(36.9, 2, 'с'),"
        " f.percent(0.613), f.number(12345, 0), f.number(1e-4, 'auto')]));"
    )

    assert json.loads(_node(script)) == [
        "0,20",
        "\u22121,5",
        "36,90\u00a0с",
        "61\u00a0%",
        "12\u00a0345",
        "1e\u22124",
    ]


@pytest.mark.skipif(_NODE is None, reason="Node is optional; it only runs the pure JS helpers")
def test_log_scale_maps_slider_positions_to_values_and_back() -> None:
    url = (WEB_DIR / "core" / "format.js").as_uri()
    script = (
        f"import * as f from {json.dumps(url)};"
        "const s = f.logScale(1e-5, 1e-2);"
        "console.log(JSON.stringify([s.toValue(0), s.toValue(1), s.toPosition(1e-2),"
        " Math.round(s.toPosition(3e-4) * 1000) / 1000, s.toValue(s.toPosition(3e-4))]));"
    )

    lo, hi, top, mid, back = json.loads(_node(script))

    assert (lo, hi, top, mid) == (1e-5, 1e-2, 1, 0.492) and abs(back - 3e-4) < 1e-12


# ---- shell: rail, routes, keys, theme, tooltips (DESIGN.md sections 2, 4.1, 11.6 task 4) ----

SHELL_FILES = (
    "css/layout.css",
    "core/router.js",
    "core/keys.js",
    "core/theme.js",
    "ui/tooltip.js",
)
SECTIONS = {
    "lab": "flask-conical",
    "tracks": "route",
    "challenges": "trophy",
    "garage": "warehouse",
    "settings": "settings",
}
INDEX = WEB_DIR / "index.html"


def _index_html() -> str:
    return _HTML_COMMENT.sub("", INDEX.read_text("utf-8"))


def test_shell_files_exist() -> None:
    assert [n for n in SHELL_FILES if not (WEB_DIR / n).is_file()] == []


def test_index_loads_the_component_and_layout_layers() -> None:
    sheets = re.findall(r"""<link[^>]*rel=["']stylesheet["'][^>]*href=["']([^"']+)["']""", _index_html())

    assert [s for s in sheets if s != "/static/style.css"] == [
        "/static/css/tokens.css",
        "/static/css/base.css",
        "/static/css/components.css",
        "/static/css/layout.css",
        "/static/css/screens.css",
    ]


@pytest.mark.parametrize(("name", "icon"), SECTIONS.items())
def test_rail_has_a_link_with_an_icon_and_an_explanation_per_section(name: str, icon: str) -> None:
    link = re.search(
        rf"""<a\b[^>]*class=["']rail-item["'][^>]*href=["']#/{name}["'][^>]*>.*?</a>""",
        _index_html(),
        re.DOTALL,
    )

    assert link is not None
    assert 'class="rail-item' in link.group(0)
    assert f"#i-{icon}" in link.group(0)
    tip = re.search(r"""data-tip=["']([^"']{20,})["']""", link.group(0))
    assert tip is not None, "every rail item explains what is behind it"


@pytest.mark.parametrize("name", SECTIONS)
def test_every_section_has_a_screen_with_a_focusable_title(name: str) -> None:
    screen = re.search(
        rf"""<section\b[^>]*data-screen=["']{name}["'][^>]*>(.*?)(?=<section\b[^>]*data-screen=|</main>)""",
        _index_html(),
        re.DOTALL,
    )

    assert screen is not None
    assert re.search(r"""<h1\b[^>]*tabindex=["']-1["']""", screen.group(1))


def test_placeholder_sections_say_they_are_coming_soon() -> None:
    html = _index_html()
    for name in ("garage", "settings"):
        screen = re.search(
            rf"""<section\b[^>]*data-screen=["']{name}["'][^>]*>(.*?)(?=<section\b[^>]*data-screen=|</main>)""",
            html,
            re.DOTALL,
        )
        assert screen is not None and "скоро" in screen.group(1).lower()


def test_lab_screen_keeps_every_element_the_app_script_looks_up() -> None:
    html = _index_html()
    wanted = set(re.findall(r"""\$\('([\w-]+)'\)""", (WEB_DIR / "app.js").read_text("utf-8")))

    assert sorted(i for i in wanted if f'id="{i}"' not in html) == []


def test_app_shortcuts_only_act_on_the_lab_screen() -> None:
    assert "dataset.screen" in (WEB_DIR / "app.js").read_text("utf-8")


def test_app_starts_the_router_the_global_keys_and_the_tooltips() -> None:
    source = (WEB_DIR / "app.js").read_text("utf-8")

    for module in ("./core/router.js", "./core/keys.js", "./core/theme.js", "./ui/tooltip.js"):
        assert module in source


def test_tooltip_waits_half_a_second_and_closes_on_escape() -> None:
    source = (WEB_DIR / "ui" / "tooltip.js").read_text("utf-8")

    assert "data-tip" in source and "500" in source and "Escape" in source
    assert "focusin" in source and "role" in source and "tooltip" in source


def test_layout_stylesheet_styles_the_rail() -> None:
    selectors = " ".join(s for _, s, _ in _rules(_css(CSS_DIR / "layout.css")))

    for name in ("shell", "rail", "rail-brand", "rail-item", "screen", "screen-head"):
        assert re.search(rf"\.{name}(?![\w-])", selectors), name
    assert 'aria-current="page"' in selectors


def _node_json(module: str, expression: str) -> object:
    url = (WEB_DIR / module).as_uri()
    script = f"import * as m from {json.dumps(url)};console.log(JSON.stringify({expression}));"
    return json.loads(_node(script))


@pytest.mark.skipif(_NODE is None, reason="Node is optional; it only runs the pure JS helpers")
def test_router_parses_hash_routes() -> None:
    result = _node_json(
        "core/router.js",
        "[m.SECTIONS, m.parseRoute(''), m.parseRoute('#/tracks/new'), m.parseRoute('#/lab/new/network'),"
        " m.parseRoute('#/nowhere'), m.parseRoute('#/garage/42/test'), m.hashFor('settings')]",
    )

    assert result == [
        ["lab", "tracks", "challenges", "garage", "settings"],
        {"name": "lab", "rest": [], "valid": False},
        {"name": "tracks", "rest": ["new"], "valid": True},
        {"name": "lab", "rest": ["new", "network"], "valid": True},
        {"name": "lab", "rest": [], "valid": False},
        {"name": "garage", "rest": ["42", "test"], "valid": True},
        "#/settings",
    ]


@pytest.mark.skipif(_NODE is None, reason="Node is optional; it only runs the pure JS helpers")
def test_global_keys_map_digits_to_sections_in_rail_order() -> None:
    result = _node_json(
        "core/keys.js",
        "[m.sectionForKey({code:'Digit1'}), m.sectionForKey({code:'Digit5'}),"
        " m.sectionForKey({code:'Numpad3'}), m.sectionForKey({code:'Digit6'}),"
        " m.sectionForKey({code:'Digit1', ctrlKey:true}), m.sectionForKey({code:'Digit1', shiftKey:true}),"
        " m.sectionForKey({code:'Digit2', isComposing:true}), m.isThemeKey({code:'KeyT'}),"
        " m.isThemeKey({code:'KeyT', metaKey:true})]",
    )

    assert result == ["lab", "settings", "challenges", None, None, None, None, True, False]


@pytest.mark.skipif(_NODE is None, reason="Node is optional; it only runs the pure JS helpers")
def test_theme_preference_helpers() -> None:
    result = _node_json(
        "core/theme.js",
        "[m.normalizeTheme('light'), m.normalizeTheme('x'), m.normalizeTheme(null),"
        " m.effectiveTheme('system', true), m.effectiveTheme('system', false),"
        " m.effectiveTheme('light', true), m.toggledTheme('system', true), m.toggledTheme('light', true),"
        " m.toggledTheme('dark', false)]",
    )

    assert result == ["light", "system", "system", "dark", "light", "light", "light", "dark", "light"]


# ---- learner choice in the run form ---------------------------------------------------------


def test_params_form_builds_the_learner_block_from_the_catalog_schema() -> None:
    source = (WEB_DIR / "params_form.js").read_text("utf-8")

    for needle in ("catalog.learners", "data-learner-group", "schemaField", "validateSchema", "logScale"):
        assert needle in source


def test_runs_of_other_learners_count_iterations_not_generations() -> None:
    sources = {name: (WEB_DIR / name).read_text("utf-8") for name in ("app.js", "charts.js", "track_view.js")}

    assert "итер." in sources["app.js"]
    assert "итерация" in sources["charts.js"]
    assert "Итер." in sources["track_view.js"]


def test_a_run_can_be_edited_and_restarted_in_place() -> None:
    form = (WEB_DIR / "params_form.js").read_text("utf-8")
    app = (WEB_DIR / "app.js").read_text("utf-8")

    for needle in ("Изменить и перезапустить", "showEdit", "onRestart", "Перезапустить с изменениями"):
        assert needle in form
    assert "restartRun" in app and "run.color = old.color" in app


# ---- game layer: levels, stars, comparison -----------------------------------------------------


@pytest.mark.skipif(_NODE is None, reason="Node is optional; it only runs the pure JS helpers")
def test_levels_are_judged_from_the_run_history_and_its_config() -> None:
    result = _node_json(
        "core/levels.js",
        """(() => {
  const gens = [{finished: 0, best_lap_steps: null}, {finished: 2, best_lap_steps: 600}, {finished: 5, best_lap_steps: 450}];
  const config = {track: 'oval', laps: 2, max_steps: 1100, learner: 'evolution',
    model: {inputs: ['ray:-90', 'ray:0', 'ray:90', 'speed'], hidden: [6, 5]}};
  const level = (id) => m.byId(id);
  return [
    m.metrics(gens),
    m.evaluate(level('oval-two'), {config, gens}),
    m.evaluate(level('quick-learner'), {config: {...config, track: 'wavy', max_steps: 1260}, gens}),
    m.evaluate(level('wavy-two'), {config, gens}),
    m.evaluate(level('oval-two'), {config: {...config, max_steps: 1500}, gens}),
    m.evaluate(level('oval-two'), {config: {...config, laps: 1}, gens}),
    m.evaluate(level('tiny-brain'), {config: {...config, max_steps: 1200}, gens}),
    m.evaluate(level('rl-pilot'), {config: {...config, track: 'circuit', max_steps: 2400}, gens}),
    m.evaluate(level('sharp-eye'), {config: {...config, track: 'circuit', max_steps: 2400}, gens: []}),
    m.isUnlocked(level('wavy-two'), {}), m.isUnlocked(level('wavy-two'), {'oval-two': 1}),
    m.bestStars([{config, gens}], {}),
    m.bestStars([{config: {...config, track: 'circuit'}, gens}], {}),
  ];
})()""",
    )

    assert result[0] == {"finish_iter": 2, "race_seconds": 15.0}
    assert result[1] == {"counts": True, "value": 15.0, "stars": 3}
    assert result[2] == {"counts": True, "value": 2, "stars": 3}
    assert result[3]["counts"] is False  # another track
    assert result[4]["counts"] is False  # the step limit was relaxed past the level's
    assert result[5]["counts"] is False  # one lap is not the two the level asks for
    assert result[6]["counts"] is False  # ten hidden neurons break the four-neuron limit
    assert result[7]["counts"] is False  # evolution run, the level wants PPO
    assert result[8] == {"counts": True, "value": None, "stars": 0}
    assert result[9:11] == [False, True]
    assert result[11]["oval-two"] == 3 and "wavy-two" not in result[11]
    assert result[12] == {}  # nothing counts on a track no level fits


@pytest.mark.skipif(_NODE is None, reason="Node is optional; it only runs the pure JS helpers")
def test_every_level_is_playable_and_start_configs_fit_their_own_limits() -> None:
    result = _node_json(
        "core/levels.js",
        """(() => {
  const defaults = {track: 'circuit', name: 'run', learner: 'evolution', laps: 1, max_steps: 1500,
    model: {inputs: ['ray:-90', 'ray:-30', 'ray:0', 'ray:30', 'ray:90', 'speed'], hidden: [6, 5], outputs: ['steer', 'throttle']}};
  return m.LEVELS.map((level) => [level.id, m.fits(level, m.startConfig(level, defaults)), m.fits(level, defaults),
    level.stars.length, [...level.stars].sort((a, b) => b - a).join() === level.stars.join(),
    level.metric === 'race_seconds' ? level.stars[0] <= level.maxSteps / 30 : true]);
})()""",
    )

    for level_id, fits_start, fits_defaults, count, tightening, in_limit in result:
        assert fits_start and count == 3 and tightening and in_limit, level_id
        assert not fits_defaults, f"{level_id} must not be passable with the stock settings"
    assert len({row[0] for row in result}) == len(result) >= 8


@pytest.mark.skipif(_NODE is None, reason="Node is optional; it only runs the pure JS helpers")
def test_comparison_rows_flag_the_best_values_per_track_and_sort_nulls_last() -> None:
    result = _node_json(
        "screens/compare.js",
        """(() => {
  const run = (id, track, gens) => ({id, name: id, color: '#fff', learner: 'evolution', status: 'finished', gens, config: track && {track, population: 10}});
  const a = run('a', 'oval', [{best: 0.5, mean: 0.1, finished: 0, best_lap_steps: null}, {best: 1, mean: 0.6, finished: 3, best_lap_steps: 600}]);
  const b = run('b', 'oval', [{best: 1, mean: 0.9, finished: 5, best_lap_steps: 450}]);
  const c = run('c', 'circuit', [{best: 0.2, mean: 0.1, finished: 0, best_lap_steps: null}]);
  const d = run('d', null, []);
  const rows = m.compareRows([a, b, c, d]);
  return [rows.map((r) => [r.id, r.iterations, r.bestLap, r.firstLap, r.best]),
    m.sortRows(rows, 'bestLap', 'asc').map((r) => r.id), m.sortRows(rows, 'bestLap', 'desc').map((r) => r.id)];
})()""",
    )

    rows, ascending, descending = result
    by_id = {row[0]: row for row in rows}
    assert by_id["a"][1:4] == [2, 20.0, 2]
    assert by_id["b"][2] == 15.0 and by_id["b"][3] == 1
    assert by_id["b"][4] == {"progress": True, "lap": True, "firstLap": True}
    assert by_id["a"][4]["progress"] is True and by_id["a"][4]["lap"] is False
    assert by_id["c"][4]["progress"] is True and by_id["d"][4] == {
        "progress": False,
        "lap": False,
        "firstLap": False,
    }
    assert ascending[:2] == ["b", "a"] and descending[:2] == ["a", "b"]
    assert set(ascending[2:]) == set(descending[2:]) == {"c", "d"}


def test_challenges_and_comparison_are_wired_into_the_lab() -> None:
    app = (WEB_DIR / "app.js").read_text("utf-8")
    html = _index_html()

    for needle in ("renderChallenges", "renderCompare", "renderTaskPanel", "bestStars", "#/lab/compare", "KeyG"):
        assert needle in app
    for needle in ('id="challenges-list"', 'id="compare-view"', 'id="view-toggle"', 'id="stars-total"'):
        assert needle in html


# ---- tracks screen and editor ------------------------------------------------------------------


@pytest.mark.skipif(_NODE is None, reason="Node is optional; it only runs the pure JS helpers")
def test_track_geometry_helpers() -> None:
    result = _node_json(
        "core/geometry.js",
        """(() => {
  const square = [[0, 0], [10, 0], [10, 10], [0, 10]];
  const bowtie = [[0, 0], [50, 50], [50, 0], [0, 50]];
  return [
    m.selfIntersections(bowtie), m.selfIntersections(square), m.polylineLength(square),
    m.snapPoint([1.4, -0.2]), m.chaikin(square, 1).length, m.chaikin(square, 9, 40).length,
    m.insertPoint(square, [5, -1]), m.closestOnPolyline(square, [5, -3]).index,
    Object.entries(m.TEMPLATES).map(([name, make]) => [name, m.selfIntersections(make()).length, m.signedArea(make()) > 0]),
  ];
})()""",
    )

    assert result[0] == [[0, 2]] and result[1] == []
    assert result[2] == 40 and result[3] == [1, 0]
    assert result[4] == 8 and result[5] == 32
    assert result[6] == [[0, 0], [5, -1], [10, 0], [10, 10], [0, 10]] and result[7] == 0
    assert all(count == 0 and counter_clockwise for _, count, counter_clockwise in result[8])


def test_the_tracks_screen_offers_choosing_drawing_editing_and_deleting() -> None:
    screen = (WEB_DIR / "screens" / "tracks.js").read_text("utf-8")
    app = (WEB_DIR / "app.js").read_text("utf-8")

    for needle in ("Выбрать для запуска", "Изменить", "Удалить", "Сохранить трассу", "overwrite", "TEMPLATES"):
        assert needle in screen
    for needle in ("initTracks", "useTrack", "reloadTracks", "setTrack"):
        assert needle in app
    assert 'id="tracks-root"' in _index_html()


def test_the_editor_canvas_edits_points_with_mouse_and_keyboard() -> None:
    source = (WEB_DIR / "views" / "track_editor_canvas.js").read_text("utf-8")

    for needle in ("pointerdown", "contextmenu", "Delete", "ArrowLeft", "KeyZ", "insertPoint", "selfIntersections"):
        assert needle in source


@pytest.mark.skipif(_NODE is None, reason="Node is optional; it only runs the pure JS helpers")
def test_a_training_budget_counts_only_the_first_iterations() -> None:
    result = _node_json(
        "core/levels.js",
        """(() => {
  const config = {track: 'oval', laps: 3, max_steps: 1500, learner: 'evolution', model: {inputs: ['ray:0'], hidden: [4]}};
  const slow = Array.from({length: 50}, (_, i) => ({finished: i === 44 ? 3 : 0, best_lap_steps: i === 44 ? 1400 : null}));
  const quick = [{finished: 0, best_lap_steps: null}, {finished: 4, best_lap_steps: 1380}];
  const level = m.byId('oval-three');
  return [m.evaluate(level, {config, gens: slow}), m.evaluate(level, {config, gens: quick}), m.startConfig(level, config).generations];
})()""",
    )

    assert result[0] == {"counts": True, "value": None, "stars": 0}  # the finish came after iteration 40
    assert result[1]["value"] == 46.0 and result[1]["stars"] == 2
    assert result[2] == 40


def test_a_challenge_opens_a_track_preview_with_stock_settings_and_a_hint_panel() -> None:
    app = (WEB_DIR / "app.js").read_text("utf-8")
    form = (WEB_DIR / "params_form.js").read_text("utf-8")

    assert "previewTrack" in app and "state.challenge" in app and 'id="task-panel"' in _index_html()
    # the level is not applied to the form: only the track and the name are set
    assert "startConfig(" not in app.split("async function startLevel")[1].split("function renderTask")[0]
    assert "onTrack" in form and "onChange" in form


def test_the_form_explains_what_each_activation_function_is() -> None:
    form = (WEB_DIR / "params_form.js").read_text("utf-8")

    for needle in ("ACTIVATION_TEXT", "tanh:", "relu:", "sigmoid:", "activationCurve", "syncActivation"):
        assert needle in form


@pytest.mark.skipif(_NODE is None, reason="Node is optional; it only runs the pure JS helpers")
def test_level_conditions_tell_how_to_fix_each_missing_setting() -> None:
    result = _node_json(
        "core/levels.js",
        """(() => {
  const stock = {track: 'circuit', laps: 1, max_steps: 1500, learner: 'evolution', model: {inputs: ['ray:0', 'ray:30', 'ray:60', 'ray:-30'], hidden: [6, 5]}};
  const levels = ['oval-two', 'rl-pilot', 'sharp-eye', 'tiny-brain'].map((id) => m.byId(id));
  return levels.map((level) => m.conditions(level, stock).filter((item) => !item.ok).map((item) => [item.key, item.how]));
})()""",
    )

    keys = [[key for key, _ in row] for row in result]
    assert keys == [
        ["track", "laps", "steps"],
        ["laps", "learner"],
        ["laps", "rays"],
        ["track", "laps", "steps", "hidden"],
    ]
    assert all("«" in how for row in result for _, how in row)


def test_runs_of_an_earlier_session_are_marked_in_the_tabs_and_explained() -> None:
    app = (WEB_DIR / "app.js").read_text("utf-8")

    assert "archived" in app and "архив" in app and "прошлого сеанса" in app
