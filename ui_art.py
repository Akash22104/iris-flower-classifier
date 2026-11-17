"""SVG artwork and CSS for the Iris classifier UI.

Everything is generated in code, so the app has no external image
dependencies and works the same locally and on Streamlit Cloud.
"""

import base64
import random
from typing import TypedDict


class SpeciesStyle(TypedDict):
    petal: str
    petal_dark: str
    tint: str
    lengths: tuple[int, int]  # (standards, falls)
    width: int
    blurb: str


SPECIES_STYLE: dict[str, SpeciesStyle] = {
    "setosa": {
        "petal": "#f59e0b",
        "petal_dark": "#b45309",
        "tint": "#3a2a10",
        "lengths": (42, 46),  # (standards, falls): short, rounded petals
        "width": 30,
        "blurb": "Small and compact, with short, broad petals. "
        "The easiest species to tell apart.",
    },
    "versicolor": {
        "petal": "#14b8a6",
        "petal_dark": "#0f766e",
        "tint": "#0c3330",
        "lengths": (60, 66),
        "width": 24,
        "blurb": "The 'blue flag' iris. Medium-sized petals that "
        "overlap with virginica in some measurements.",
    },
    "virginica": {
        "petal": "#3b82f6",
        "petal_dark": "#1d4ed8",
        "tint": "#0f2350",
        "lengths": (72, 80),
        "width": 22,
        "blurb": "The largest of the three, with long, narrow petals "
        "and wide sepals.",
    },
}


def species_key(name):
    return str(name).strip().lower()


def species_style(name) -> SpeciesStyle:
    return SPECIES_STYLE.get(species_key(name), SPECIES_STYLE["versicolor"])


def species_color(name):
    return species_style(name)["petal"]


def _petal_path(length, width):
    """A petal pointing straight up from the origin."""
    w, l = width, length
    return (
        f"M0,0 C{w},{-l * 0.25} {w * 0.75},{-l} 0,{-l} "
        f"C{-w * 0.75},{-l} {-w},{-l * 0.25} 0,0Z"
    )


def flower_svg(species, size=220):
    """An illustrated iris flower whose proportions follow the species."""
    style = species_style(species)
    std_len, fall_len = style["lengths"]
    width = style["width"]
    cx, cy = 110, 108

    standards = "".join(
        f'<path d="{_petal_path(std_len, width * 0.85)}" '
        f'transform="translate({cx},{cy}) rotate({angle})" '
        f'fill="url(#std)" stroke="{style["petal_dark"]}" stroke-width="1.2"/>'
        for angle in (-32, 32, 0)
    )
    falls = "".join(
        f'<g transform="translate({cx},{cy}) rotate({angle})">'
        f'<path d="{_petal_path(fall_len, width)}" fill="url(#fall)" '
        f'stroke="{style["petal_dark"]}" stroke-width="1.2"/>'
        f'<path d="M0,-6 L0,{-fall_len * 0.55}" stroke="#e2e8f0" '
        f'stroke-width="5" stroke-linecap="round"/>'
        f"</g>"
        for angle in (180, 122, 238)
    )

    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 220 260"
 width="{size}" height="{int(size * 260 / 220)}">
  <defs>
    <linearGradient id="std" x1="0" y1="1" x2="0" y2="0">
      <stop offset="0" stop-color="{style['petal_dark']}"/>
      <stop offset="1" stop-color="{style['petal']}"/>
    </linearGradient>
    <linearGradient id="fall" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="{style['petal']}"/>
      <stop offset="1" stop-color="{style['petal_dark']}"/>
    </linearGradient>
  </defs>
  <path d="M110,140 C108,185 112,215 110,258" stroke="#3d7a6a"
   stroke-width="6" fill="none" stroke-linecap="round"/>
  <path d="M110,250 C80,220 70,190 74,160 C90,190 102,215 110,250Z" fill="#4b8b7a"/>
  <path d="M110,252 C140,226 152,196 150,170 C134,196 120,220 110,252Z" fill="#3d7a6a"/>
  {falls}
  {standards}
  <circle cx="{cx}" cy="{cy}" r="7" fill="#e2e8f0"/>
</svg>"""


def _scattered_petals(rng, count, width, height, colors, opacity):
    shapes = []
    for _ in range(count):
        x = rng.uniform(0, width)
        y = rng.uniform(0, height)
        scale = rng.uniform(0.5, 1.6)
        angle = rng.uniform(0, 360)
        color = rng.choice(colors)
        shapes.append(
            f'<path d="{_petal_path(46, 20)}" fill="{color}" '
            f'opacity="{opacity * rng.uniform(0.5, 1):.2f}" '
            f'transform="translate({x:.0f},{y:.0f}) rotate({angle:.0f}) '
            f'scale({scale:.2f})"/>'
        )
    return "".join(shapes)


def _grid(width, height, step, color, opacity):
    lines = [
        f'<path d="M{x},0 V{height}" stroke="{color}" opacity="{opacity}"/>'
        for x in range(0, width + 1, step)
    ] + [
        f'<path d="M0,{y} H{width}" stroke="{color}" opacity="{opacity}"/>'
        for y in range(0, height + 1, step)
    ]
    return "".join(lines)


def _circuit_traces(rng, count, width, height, colors):
    """Right-angled 'circuit board' traces ending in small nodes."""
    traces = []
    for _ in range(count):
        x = rng.randrange(0, width, 40)
        y = rng.randrange(0, height, 40)
        points = [(x, y)]
        for _ in range(rng.randint(2, 4)):
            if rng.random() < 0.5:
                x += rng.choice([-1, 1]) * rng.randint(2, 6) * 40
            else:
                y += rng.choice([-1, 1]) * rng.randint(2, 5) * 40
            points.append((x, y))
        color = rng.choice(colors)
        path = "M" + " L".join(f"{px},{py}" for px, py in points)
        traces.append(
            f'<path d="{path}" stroke="{color}" stroke-width="2" fill="none" '
            f'opacity="0.28"/>'
            f'<circle cx="{points[-1][0]}" cy="{points[-1][1]}" r="5" '
            f'fill="none" stroke="{color}" stroke-width="2" opacity="0.5"/>'
            f'<circle cx="{points[0][0]}" cy="{points[0][1]}" r="3" '
            f'fill="{color}" opacity="0.5"/>'
        )
    return "".join(traces)


def page_background_svg():
    """Dark blueprint backdrop: grid, circuit traces and soft glows."""
    rng = random.Random(7)
    w, h = 1600, 1000
    traces = _circuit_traces(rng, 26, w, h, ["#38bdf8", "#14b8a6", "#3b82f6"])
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}"
 preserveAspectRatio="xMidYMid slice">
  <defs>
    <linearGradient id="base" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#070d1a"/>
      <stop offset="0.6" stop-color="#0b1426"/>
      <stop offset="1" stop-color="#08161a"/>
    </linearGradient>
    <radialGradient id="glow1" cx="0.12" cy="0.08" r="0.55">
      <stop offset="0" stop-color="#1d4ed8" stop-opacity="0.35"/>
      <stop offset="1" stop-color="#1d4ed8" stop-opacity="0"/>
    </radialGradient>
    <radialGradient id="glow2" cx="0.92" cy="0.85" r="0.55">
      <stop offset="0" stop-color="#0f766e" stop-opacity="0.32"/>
      <stop offset="1" stop-color="#0f766e" stop-opacity="0"/>
    </radialGradient>
  </defs>
  <rect width="{w}" height="{h}" fill="url(#base)"/>
  {_grid(w, h, 40, "#38bdf8", 0.05)}
  {_grid(w, h, 200, "#38bdf8", 0.09)}
  <rect width="{w}" height="{h}" fill="url(#glow1)"/>
  <rect width="{w}" height="{h}" fill="url(#glow2)"/>
  {traces}
</svg>"""


def hero_svg():
    """Banner art: a scanner-style display analysing three irises."""
    rng = random.Random(3)
    w, h = 1200, 260
    flowers = []
    for i, species in enumerate(["setosa", "versicolor", "virginica"]):
        x = 740 + i * 170
        y = 30 + rng.uniform(0, 20)
        inner = flower_svg(species)
        inner = inner[inner.index(">") + 1 : inner.rindex("</svg>")]
        color = SPECIES_STYLE[species]["petal"]
        flowers.append(
            f'<g transform="translate({x},{y}) scale(0.78) translate(-110,0)">'
            f"{inner}</g>"
            # Targeting brackets around each flower.
            f'<g stroke="{color}" stroke-width="3" fill="none" opacity="0.9">'
            f'<path d="M{x - 70},{y + 10} v-14 h14"/>'
            f'<path d="M{x + 70},{y + 10} v-14 h-14"/>'
            f'<path d="M{x - 70},{y + 200} v14 h14"/>'
            f'<path d="M{x + 70},{y + 200} v14 h-14"/></g>'
        )
    traces = _circuit_traces(rng, 10, int(w * 0.6), h, ["#38bdf8", "#14b8a6"])
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}"
 preserveAspectRatio="xMaxYMid slice">
  <defs>
    <linearGradient id="steel" x1="0" y1="0" x2="1" y2="0.4">
      <stop offset="0" stop-color="#0b1220"/>
      <stop offset="0.55" stop-color="#10213f"/>
      <stop offset="1" stop-color="#0d3b4a"/>
    </linearGradient>
  </defs>
  <rect width="{w}" height="{h}" fill="url(#steel)"/>
  {_grid(w, h, 30, "#38bdf8", 0.08)}
  {traces}
  <path d="M0,{h - 3} H{w}" stroke="#38bdf8" stroke-width="3" opacity="0.8"/>
  {''.join(flowers)}
</svg>"""


def svg_data_uri(svg):
    encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


def base_css():
    """Page background, hero banner and dark glass panels.

    Panels are targeted by their Streamlit container keys
    (`.st-key-<key>`), so styling stays scoped to this app's own blocks.
    """
    background = svg_data_uri(page_background_svg())
    hero = svg_data_uri(hero_svg())
    return f"""
<style>
.stApp {{
  background: url("{background}") center / cover fixed no-repeat;
}}
.st-key-hero {{
  background:
    linear-gradient(90deg, rgba(11, 18, 32, 0.92) 0%,
      rgba(11, 18, 32, 0.55) 45%, rgba(11, 18, 32, 0) 70%),
    url("{hero}") right center / cover no-repeat;
  border-radius: 0.6rem;
  border: 1px solid rgba(56, 189, 248, 0.35);
  padding: 2rem 2.25rem;
  box-shadow: 0 0 0 1px rgba(56, 189, 248, 0.08),
    0 20px 45px -22px rgba(14, 165, 233, 0.55);
}}
.st-key-hero h1 {{
  color: #f8fafc !important;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}}
.st-key-hero p {{
  color: #cbd5e1 !important;
}}
[class*="st-key-panel_"] {{
  background: rgba(10, 18, 34, 0.82);
  backdrop-filter: blur(8px);
  -webkit-backdrop-filter: blur(8px);
  border: 1px solid rgba(56, 189, 248, 0.2);
  border-top: 3px solid #3b82f6;
  border-radius: 0.6rem;
  padding: 1.25rem 1.4rem;
  box-shadow: 0 14px 34px -20px rgba(0, 0, 0, 0.8);
}}
</style>
"""


def result_card_css(species):
    style = species_style(species)
    return f"""
<style>
.st-key-panel_result {{
  background: linear-gradient(160deg, {style['tint']} 0%,
    rgba(10, 18, 34, 0.9) 75%);
  border: 1px solid {style['petal']}66;
  border-top: 3px solid {style['petal']};
  box-shadow: 0 0 28px -10px {style['petal']}88;
  transition: background 0.4s ease, border-color 0.4s ease;
}}
</style>
"""
