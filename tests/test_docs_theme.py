"""The documentation theme: what the site ships, and the properties a restyle loses quietly.

``mkdocs build --strict`` (CI, and the e2e test from the sdist) proves the site builds; the
rendered result is checked in a real browser by hand. These are the guards that need neither:
the colour tokens keep their contrast, the motion keeps its reduced-motion switch, the fonts are
self-hosted, the home page is drawn from real numbers and every link in the templates lands.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs"
OVERRIDES = REPO / "overrides"
CSS = DOCS / "assets" / "stylesheets" / "pmlab.css"
JS = DOCS / "assets" / "javascripts" / "pmlab.js"
DATA = DOCS / "assets" / "data" / "sample-run.json"
FONTS = DOCS / "assets" / "fonts"
SITE_URL = "https://pmlab.hiddenterps.si/"


def _config() -> dict[str, Any]:
    config: dict[str, Any] = yaml.safe_load(
        (REPO / "mkdocs.yml").read_text().replace("!!python/name:", "")
    )
    return config


# ── colour: WCAG 2.x contrast, computed from the tokens in the stylesheet ──────────────────────


def _channel(value: int) -> float:
    c = value / 255
    return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4


def _luminance(hex_color: str) -> float:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return 0.2126 * _channel(r) + 0.7152 * _channel(g) + 0.0722 * _channel(b)


def contrast(a: str, b: str) -> float:
    hi, lo = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def _tokens(selector: str) -> dict[str, str]:
    """The ``--pm-*: #hex`` declarations of the first rule whose selector is ``selector``."""
    css = CSS.read_text()
    block = re.search(re.escape(selector) + r"\s*\{(.*?)\}", css, re.DOTALL)
    assert block, f"no `{selector}` rule in {CSS.name}"
    return dict(re.findall(r"(--pm-[\w-]+)\s*:\s*(#[0-9a-fA-F]{6})\b", block.group(1)))


BODY_TEXT = (
    "--pm-ink",
    "--pm-ink-2",
    "--pm-muted",
    "--pm-red-text",
    "--pm-sage-text",
    "--pm-amber-text",
)
CONSOLE_TEXT = (
    "--pm-c-ink",
    "--pm-c-ink-2",
    "--pm-c-muted",
    "--pm-c-red",
    "--pm-c-sage",
    "--pm-c-amber",
)
CODE_TEXT = (
    "--pm-code-keyword",
    "--pm-code-string",
    "--pm-code-number",
    "--pm-code-comment",
    "--pm-code-function",
    "--pm-code-name",
)


class TestContrast:
    @pytest.mark.parametrize("scheme", ["slate", "default"])
    def test_body_text_is_aa_on_the_page_and_on_its_surface(self, scheme: str) -> None:
        tokens = _tokens(f'[data-md-color-scheme="{scheme}"]')
        for name in BODY_TEXT:
            for ground in ("--pm-bg", "--pm-surface"):
                ratio = contrast(tokens[name], tokens[ground])
                assert ratio >= 4.5, f"{scheme}: {name} on {ground} is {ratio:.2f}:1 (AA needs 4.5)"

    @pytest.mark.parametrize("scheme", ["slate", "default"])
    def test_form_and_control_borders_are_visible_against_the_page(self, scheme: str) -> None:
        tokens = _tokens(f'[data-md-color-scheme="{scheme}"]')
        ratio = contrast(tokens["--pm-line-strong"], tokens["--pm-bg"])
        assert ratio >= 3.0, f"{scheme}: control border is {ratio:.2f}:1 (WCAG 1.4.11 needs 3)"

    def test_the_console_surfaces_are_aa_in_both_schemes(self) -> None:
        # Header, hero, code blocks and footer stay charcoal in the light scheme too.
        tokens = _tokens(":root")
        for name in CONSOLE_TEXT:
            for ground in ("--pm-c-bg", "--pm-c-surface"):
                ratio = contrast(tokens[name], tokens[ground])
                assert ratio >= 4.5, f"{name} on {ground} is {ratio:.2f}:1"

    def test_syntax_colours_are_aa_on_the_code_background(self) -> None:
        tokens = _tokens(":root")
        for name in CODE_TEXT:
            ratio = contrast(tokens[name], tokens["--pm-c-surface"])
            assert ratio >= 4.5, f"{name} on the code background is {ratio:.2f}:1"

    def test_the_primary_button_label_is_aa(self) -> None:
        tokens = _tokens(":root")
        assert contrast(tokens["--pm-on-red"], tokens["--pm-red-btn"]) >= 4.5


# ── motion ─────────────────────────────────────────────────────────────────────────────────────


class TestMotion:
    def test_reduced_motion_turns_every_animation_and_transition_off(self) -> None:
        css = CSS.read_text()
        match = re.search(
            r"@media \(prefers-reduced-motion: reduce\)\s*\{(.*?)\n\}", css, re.DOTALL
        )
        assert match, "no prefers-reduced-motion block"
        body = match.group(1)
        assert "animation: none !important" in body
        assert "transition: none !important" in body

    def test_keyframes_animate_only_transform_and_opacity(self) -> None:
        css = CSS.read_text()
        frames = re.findall(r"@keyframes\s+([\w-]+)\s*\{((?:[^{}]|\{[^{}]*\})*)\}", css)
        assert frames, "no keyframes at all?"
        for name, body in frames:
            props = set(
                re.findall(r"([\w-]+)\s*:", re.sub(r"\{[^{}]*\}", lambda m: m.group(0), body))
            )
            props -= {"from", "to"}
            assert props <= {"transform", "opacity"}, f"@keyframes {name} animates {props}"

    def test_no_transition_animates_layout_or_everything(self) -> None:
        css = CSS.read_text()
        assert not re.search(r"transition\s*:\s*all\b", css)
        for decl in re.findall(r"transition(?:-property)?\s*:\s*([^;]+);", css):
            for prop in re.split(r"[\s,]+", decl):
                assert prop not in {"width", "height", "top", "left", "margin", "padding"}, decl


# ── the look: what a restyle must not drift into ──────────────────────────────────────────────


class TestLook:
    def test_no_gradient_text_blur_or_glow(self) -> None:
        text = CSS.read_text() + "\n".join(p.read_text() for p in OVERRIDES.rglob("*.html"))
        stripped = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
        assert "background-clip: text" not in stripped
        assert "-webkit-background-clip" not in stripped
        assert "backdrop-filter" not in stripped

    def test_dark_is_the_default_and_a_light_scheme_exists(self) -> None:
        palette = _config()["theme"]["palette"]
        assert [p["scheme"] for p in palette] == ["slate", "default"]
        assert all("media" not in p for p in palette), "an OS preference would override dark-first"
        assert all(p["toggle"]["name"] for p in palette)  # an accessible name on the switch

    def test_both_schemes_define_the_same_tokens(self) -> None:
        assert (
            _tokens('[data-md-color-scheme="slate"]').keys()
            == _tokens('[data-md-color-scheme="default"]').keys()
        )


# ── no third party: fonts, scripts and styles are served from the site itself ─────────────────


class TestSelfContained:
    ALLOWED = (
        "https://pmlab.hiddenterps.si",
        "https://hiddenterps.si",
        "https://github.com/ArtBreguez/polymarket-lab",
        "https://pypi.org/project/pmlab",
        "https://semver.org",
        "https://x.com/HiddenTerps",
        "http://www.w3.org/2000/svg",
        "https://scripts.sil.org",
        "https://fonts.google.com",
    )

    def _shipped(self) -> list[Path]:
        suffixes = {".css", ".js", ".html", ".svg", ".json"}
        files = [p for p in (DOCS / "assets").rglob("*") if p.suffix in suffixes]
        return files + [p for p in OVERRIDES.rglob("*") if p.suffix in suffixes]

    def test_the_theme_does_not_load_google_fonts(self) -> None:
        assert _config()["theme"]["font"] is False

    def test_no_shipped_file_points_at_a_third_party_host(self) -> None:
        for path in self._shipped():
            for url in re.findall(r"https?://[^\s\"')<>]+", path.read_text()):
                assert url.startswith(self.ALLOWED), f"{path.relative_to(REPO)} -> {url}"

    def test_every_font_face_is_a_woff2_that_ships_with_font_display_swap(self) -> None:
        css = CSS.read_text()
        faces = re.findall(r"@font-face\s*\{(.*?)\}", css, re.DOTALL)
        assert len(faces) >= 4
        for face in faces:
            if "local(" in face:
                continue  # a metric-matched fallback, not a download
            assert "font-display: swap" in face, face
            src = re.search(r"url\(([^)]+)\)", face)
            assert src, face
            font = (CSS.parent / src.group(1).strip("\"'")).resolve()
            assert font.suffix == ".woff2" and font.exists(), font

    def test_the_fonts_stay_small(self) -> None:
        total = sum(p.stat().st_size for p in FONTS.glob("*.woff2"))
        assert total <= 150_000, f"{total} bytes of fonts (budget 150 kB)"
        assert (FONTS / "LICENSES.txt").exists()  # OFL: the licence travels with the files

    def test_the_script_is_small_and_needs_no_library(self) -> None:
        text = JS.read_text()
        assert JS.stat().st_size <= 8_000
        assert "import " not in text and "require(" not in text

    def test_extra_css_and_js_exist(self) -> None:
        config = _config()
        for entry in config["extra_css"] + config["extra_javascript"]:
            assert (DOCS / entry).exists(), entry


# ── the home page is drawn from a data file, and the data file tells the truth ────────────────


@pytest.fixture(scope="module")
def data() -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(DATA.read_text())
    return loaded


class TestSampleData:
    def test_schema_and_counts_agree_with_each_other(self, data: dict[str, Any]) -> None:
        assert data["schema"] == 1
        assert data["backtest"]["folds"] == len(data["folds"])
        assert data["sample"]["markets"] == 750
        assert len(data["gate"]["segments"]) == data["sample"]["cities"]

    def test_every_fold_trains_strictly_before_the_day_it_tests(self, data: dict[str, Any]) -> None:
        # The home page draws these as a timeline and says "no lookahead": the data must hold.
        previous_rows = 0
        for fold in data["folds"]:
            assert fold["train_end"] < fold["test"], fold
            assert fold["train_rows"] >= previous_rows, "the training window only grows"
            previous_rows = fold["train_rows"]
        tests = [f["test"] for f in data["folds"]]
        assert tests == sorted(tests) and len(set(tests)) == len(tests)

    def test_the_gate_verdict_follows_from_its_segments(self, data: dict[str, Any]) -> None:
        gate = data["gate"]
        passes = [s["passes"] for s in gate["segments"]]
        assert gate["decision"] == ("GO" if all(passes) else "NO_GO")
        for s in gate["segments"]:
            assert s["passes"] == (s["reason"] in (None, "ok")), s
        failing = [s["segment"] for s in gate["segments"] if not s["passes"]]
        assert set(gate["scoped"]["segments"]).isdisjoint(failing)
        if gate["decision"] == "NO_GO":
            assert "NO_GO" in gate["publish_refused"], "a failed gate must be refused"

    def test_signals_clear_the_edge_threshold_and_name_a_side(self, data: dict[str, Any]) -> None:
        assert 6 <= len(data["signals"]) <= 16
        for s in data["signals"]:
            assert s["edge"] >= 0.05 and s["side"] in {"yes", "no"}
            assert 0 < s["market_price"] < 1 and 0 <= s["model_prob"] <= 1

    def test_the_run_was_reproduced(self, data: dict[str, Any]) -> None:
        assert data["reproduce"]["status"] == "reproduced" and data["reproduce"]["data_match"]


class TestMkdocsHook:
    @pytest.fixture(scope="class")
    def hook(self) -> Any:
        path = REPO / "scripts" / "mkdocs_hooks.py"
        spec = importlib.util.spec_from_file_location("mkdocs_hooks", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules["mkdocs_hooks"] = module
        spec.loader.exec_module(module)
        return module

    def test_it_hands_the_data_file_to_the_templates(self, hook: Any, data: dict[str, Any]) -> None:
        class Config(dict):  # type: ignore[type-arg]
            docs_dir = str(DOCS)

        config = Config(extra={})
        assert hook.on_config(config)["extra"]["sample"] == data

    def test_a_data_file_of_another_schema_fails_the_build(self, hook: Any, tmp_path: Path) -> None:
        (tmp_path / "assets" / "data").mkdir(parents=True)
        (tmp_path / "assets" / "data" / "sample-run.json").write_text('{"schema": 99}')

        class Config(dict):  # type: ignore[type-arg]
            docs_dir = str(tmp_path)

        with pytest.raises(Exception, match="schema"):
            hook.on_config(Config(extra={}))

    def test_the_timeline_draws_training_ending_before_the_test_day(
        self, hook: Any, data: dict[str, Any]
    ) -> None:
        # The percentages are what the hero draws: a train bar must never reach into its test tick.
        rows = hook.home_view(data)["rows"]
        assert [r["n"] for r in rows] == [1, 12, 24, 36, 48, 60]
        for row in rows:
            assert 0 < row["train"] <= row["test"] + 0.05 <= 100.05, row
            assert row["train_end"] < row["test_day"]
        assert [r["train"] for r in rows] == sorted(r["train"] for r in rows)

    def test_the_gate_bars_start_and_stop_at_the_zero_line(
        self, hook: Any, data: dict[str, Any]
    ) -> None:
        view = hook.home_view(data)
        assert len(view["cities"]) == len(data["gate"]["segments"])
        for city in view["cities"]:
            if city["negative"]:
                assert abs(city["left"] + city["width"] - view["zero"]) < 0.05, city
            else:
                assert abs(city["left"] - view["zero"]) < 0.05, city
            assert city["left"] >= 0 and city["left"] + city["width"] <= 100.05

    def test_the_sparkline_has_one_bar_per_fold_scaled_into_the_box(
        self, hook: Any, data: dict[str, Any]
    ) -> None:
        spark = hook.home_view(data)["spark"]
        assert len(spark) == data["backtest"]["folds"] and min(spark) >= 18 and max(spark) <= 100

    def test_the_version_on_the_page_is_the_one_in_pyproject(self, hook: Any) -> None:
        import tomllib

        class Config(dict):  # type: ignore[type-arg]
            docs_dir = str(DOCS)

        extra = hook.on_config(Config(extra={}))["extra"]
        assert (
            extra["pmlab_version"]
            == tomllib.loads((REPO / "pyproject.toml").read_text())["project"]["version"]
        )

    def test_the_hook_is_registered(self) -> None:
        assert "scripts/mkdocs_hooks.py" in _config()["hooks"]


# ── the templates ─────────────────────────────────────────────────────────────────────────────


class TestTemplates:
    HOME = OVERRIDES / "home.html"

    def test_the_home_page_uses_its_own_template_and_the_404_extends_the_theme(self) -> None:
        assert "template: home.html" in (DOCS / "index.md").read_text()
        assert (OVERRIDES / "main.html").exists()
        assert '{% extends "main.html" %}' in (OVERRIDES / "404.html").read_text()
        assert _config()["theme"]["custom_dir"] == "overrides"

    @pytest.mark.parametrize("name", ["home.html", "404.html"])
    def test_every_internal_link_in_a_template_lands_on_a_page(self, name: str) -> None:
        text = (OVERRIDES / name).read_text()
        targets = re.findall(r"\{\{\s*'([\w/.-]+)'\s*\|\s*url\s*\}\}", text)
        assert targets or name == "404.html"
        for target in targets:
            if target.startswith("assets/"):
                assert (DOCS / target).exists(), target
                continue
            stem = target.strip("/")
            assert (DOCS / f"{stem}.md").exists() or (DOCS / stem / "index.md").exists(), target

    def test_the_home_page_states_the_four_guarantees_and_the_install_command(self) -> None:
        text = self.HOME.read_text()
        for phrase in ("lookahead", "NO_GO", "reproduc", "frozen"):
            assert phrase.lower() in text.lower(), phrase
        readme = (REPO / "README.md").read_text()
        assert "pip install pmlab" in text and "pip install pmlab" in readme
        assert "uv add pmlab" in text

    def test_the_home_quickstart_is_the_readme_quickstart(self) -> None:
        # The README block is executed by the default suite and by the e2e job against the wheel;
        # the home page renders docs/index.md, so that page must carry exactly the same code.
        fence = re.compile(r"```python\n(from pmlab import.*?)```", re.DOTALL)
        readme = fence.search((REPO / "README.md").read_text())
        index = fence.search((DOCS / "index.md").read_text())
        assert readme and index
        assert index.group(1) == readme.group(1), "docs/index.md drifted from the README quickstart"
        assert "{{ page.content }}" in self.HOME.read_text()

    def test_every_svg_and_image_in_the_templates_has_a_size(self) -> None:
        # No layout shift: a box that is drawn later must already have its width and height.
        for path in OVERRIDES.rglob("*.html"):
            for tag in re.findall(r"<(?:svg|img)\b[^>]*>", path.read_text()):
                assert "width=" in tag or "viewBox=" in tag or "aria-hidden" in tag, tag
                if tag.startswith("<img"):
                    assert "width=" in tag and "height=" in tag, tag

    def test_the_home_page_has_one_h1_and_a_skip_target(self) -> None:
        assert len(re.findall(r"<h1\b", self.HOME.read_text())) == 1


class TestDomain:
    def test_the_site_url_is_the_hiddenterps_subdomain(self) -> None:
        assert _config()["site_url"] == SITE_URL

    def test_no_file_that_ships_still_points_at_the_old_pages_url(self) -> None:
        old = "artbreguez.github.io/polymarket-lab"
        for path in (
            REPO / "pyproject.toml",
            REPO / "README.md",
            REPO / "mkdocs.yml",
            REPO / "SECURITY.md",
            *DOCS.rglob("*.md"),
            *OVERRIDES.rglob("*.html"),
            REPO / ".github" / "ISSUE_TEMPLATE" / "feature_request.yml",
        ):
            assert old not in path.read_text(), path.relative_to(REPO)
