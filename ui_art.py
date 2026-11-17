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
        "petal": "#e879b9",
        "petal_dark": "#be3d8a",
        "tint": "#fde7f3",
        "lengths": (42, 46),  # (standards, falls): short, rounded petals
        "width": 30,
        "blurb": "Small and compact, with short, broad petals. "
        "The easiest species to tell apart.",
    },
    "versicolor": {
        "petal": "#9b7ede",
        "petal_dark": "#6d4bc4",
        "tint": "#efe9fc",
        "lengths": (60, 66),
        "width": 24,
        "blurb": "The 'blue flag' iris. Medium-sized petals that "
        "overlap with virginica in some measurements.",
    },
    "virginica": {
        "petal": "#4f7fe0",
        "petal_dark": "#2c55b0",
        "tint": "#e6eefc",
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
        f'<path d="M0,-6 L0,{-fall_len * 0.55}" stroke="#f6c945" '
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
  <path d="M110,140 C108,185 112,215 110,258" stroke="#3f8f4f"
   stroke-width="6" fill="none" stroke-linecap="round"/>
  <path d="M110,250 C80,220 70,190 74,160 C90,190 102,215 110,250Z" fill="#58a967"/>
  <path d="M110,252 C140,226 152,196 150,170 C134,196 120,220 110,252Z" fill="#4a9a5a"/>
  {falls}
  {standards}
  <circle cx="{cx}" cy="{cy}" r="7" fill="#f6c945"/>
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


def page_background_svg():
    """Soft garden backdrop for the whole page."""
    rng = random.Random(7)
    w, h = 1600, 1000
    colors = ["#e879b9", "#9b7ede", "#4f7fe0", "#f6c945"]
    petals = _scattered_petals(rng, 38, w, h, colors, 0.16)

    # Gentle grass blades along the bottom edge.
    blades = []
    for i in range(70):
        x = i * (w / 70) + rng.uniform(-8, 8)
        tall = rng.uniform(60, 170)
        lean = rng.uniform(-30, 30)
        blades.append(
            f'<path d="M{x:.0f},{h} Q{x + lean / 2:.0f},{h - tall / 2:.0f} '
            f'{x + lean:.0f},{h - tall:.0f}" stroke="#58a967" '
            f'stroke-width="{rng.uniform(2, 5):.1f}" fill="none" opacity="0.22"/>'
        )

    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}"
 preserveAspectRatio="xMidYMid slice">
  <defs>
    <linearGradient id="sky" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#fbf6ff"/>
      <stop offset="0.55" stop-color="#f3ecff"/>
      <stop offset="1" stop-color="#eaf6ee"/>
    </linearGradient>
    <radialGradient id="glow1" cx="0.15" cy="0.1" r="0.5">
      <stop offset="0" stop-color="#f9c6e3" stop-opacity="0.7"/>
      <stop offset="1" stop-color="#f9c6e3" stop-opacity="0"/>
    </radialGradient>
    <radialGradient id="glow2" cx="0.9" cy="0.3" r="0.55">
      <stop offset="0" stop-color="#c9d8ff" stop-opacity="0.7"/>
      <stop offset="1" stop-color="#c9d8ff" stop-opacity="0"/>
    </radialGradient>
  </defs>
  <rect width="{w}" height="{h}" fill="url(#sky)"/>
  <rect width="{w}" height="{h}" fill="url(#glow1)"/>
  <rect width="{w}" height="{h}" fill="url(#glow2)"/>
  {petals}
  {''.join(blades)}
</svg>"""


def hero_svg():
    """Banner art: a row of irises in all three colors over a dusk gradient."""
    rng = random.Random(3)
    w, h = 1200, 260
    flowers = []
    for i, species in enumerate(["setosa", "versicolor", "virginica"] * 3):
        x = 640 + i * 62 + rng.uniform(-10, 10)
        y = 40 + rng.uniform(0, 70)
        scale = rng.uniform(0.55, 0.85)
        inner = flower_svg(species)
        inner = inner[inner.index(">") + 1 : inner.rindex("</svg>")]
        flowers.append(
            f'<g transform="translate({x:.0f},{y:.0f}) scale({scale:.2f}) '
            f'translate(-110,0)" opacity="0.95">{inner}</g>'
        )
    petals = _scattered_petals(
        rng, 22, w * 0.55, h, ["#ffffff", "#f9c6e3", "#c9d8ff"], 0.18
    )
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}"
 preserveAspectRatio="xMaxYMid slice">
  <defs>
    <linearGradient id="dusk" x1="0" y1="0" x2="1" y2="0.6">
      <stop offset="0" stop-color="#4c1d95"/>
      <stop offset="0.5" stop-color="#7c3aed"/>
      <stop offset="1" stop-color="#db7cc0"/>
    </linearGradient>
  </defs>
  <rect width="{w}" height="{h}" fill="url(#dusk)"/>
  {petals}
  {''.join(flowers)}
</svg>"""


def svg_data_uri(svg):
    encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


def base_css():
    """Page background, hero banner and frosted-glass panels.

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
  background: url("{hero}") right center / cover no-repeat;
  border-radius: 1.25rem;
  padding: 2rem 2.25rem;
  box-shadow: 0 18px 40px -20px rgba(76, 29, 149, 0.6);
}}
.st-key-hero h1, .st-key-hero p {{
  color: #ffffff !important;
  text-shadow: 0 2px 12px rgba(30, 10, 60, 0.35);
}}
[class*="st-key-panel_"] {{
  background: rgba(255, 255, 255, 0.72);
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
  border: 1px solid rgba(124, 58, 237, 0.14);
  border-radius: 1.25rem;
  padding: 1.25rem 1.4rem;
  box-shadow: 0 10px 30px -18px rgba(76, 29, 149, 0.45);
}}
</style>
"""


def result_card_css(species):
    style = species_style(species)
    return f"""
<style>
.st-key-panel_result {{
  background: linear-gradient(160deg, {style['tint']} 0%,
    rgba(255, 255, 255, 0.85) 70%);
  border: 2px solid {style['petal']}55;
  transition: background 0.4s ease, border-color 0.4s ease;
}}
</style>
"""
