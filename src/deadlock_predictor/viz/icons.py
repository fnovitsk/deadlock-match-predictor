"""Small objective icons (SVG, as data URIs) for the win-probability chart.

The game's asset service has no icons for Guardians, Walkers, Shrines or the mid-boss, so
these are drawn here: a white glyph on a round badge in the colour of the team that took
the objective, with a dark ring so they read on the dark page.
"""

from __future__ import annotations

import base64
from functools import cache

# 32×32 glyphs, white on transparent; the badge is drawn around them.
GLYPHS = {
    # A brazier tower: a tapered column with a flame.
    "Lane Guardian": (
        '<path d="M12 26h8l-1.5-11h-5z"/><rect x="10" y="25" width="12" height="3" rx="1"/>'
        '<path d="M16 5c3 3.5 3.5 6 1.2 8.5c.4-2-1.2-3.4-1.2-3.4'
        's-1.6 1.4-1.2 3.4C12.5 11 13 8.5 16 5z"/>'
    ),
    # A walker: a body on two long legs.
    "Walker": (
        '<rect x="10" y="7" width="12" height="9" rx="2.5"/><circle cx="16" cy="11.5" r="1.8" '
        'fill="#000" fill-opacity=".45"/>'
        '<path d="M12 16l-3 11h2.6l2.6-9zM20 16l3 11h-2.6l-2.6-9z"/>'
    ),
    # A keep with battlements.
    "Base Guardian": ('<path d="M8 27V11h3V8h3v3h4V8h3v3h3v16h-6v-6h-4v6z"/>'),
    # A crystal on a pedestal.
    "Shrine": (
        '<path d="M16 4l5 7l-5 9l-5-9z"/><rect x="11" y="22" width="10" height="2.5" rx="1"/>'
        '<rect x="9" y="25.5" width="14" height="2.5" rx="1"/>'
    ),
    # A crown.
    "Patron Downed": (
        '<path d="M6 23l-1-12l6 5l5-8l5 8l6-5l-1 12z"/>'
        '<rect x="6" y="24.5" width="20" height="3" rx="1"/>'
    ),
    # A horned skull.
    "Mid-boss": (
        '<path d="M9 14c0-5 3-8 7-8s7 3 7 8c0 3-1.5 5-3 6v3h-8v-3c-1.5-1-3-3-3-6z"/>'
        '<circle cx="13" cy="14" r="2" fill="#000" fill-opacity=".55"/>'
        '<circle cx="19" cy="14" r="2" fill="#000" fill-opacity=".55"/>'
        '<path d="M9.5 9L5 4l1 7zM22.5 9L27 4l-1 7z"/>'
    ),
}


@cache
def objective_icon(kind: str, color: str) -> str:
    """A data URI of the badge for `kind` (an event name, as in `viz.curve.EVENT_NAMES` or
    'Mid-boss') in `color`."""
    glyph = GLYPHS.get(kind, '<circle cx="16" cy="16" r="6"/>')
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 40 40">'
        f'<circle cx="20" cy="20" r="18.5" fill="{color}" stroke="#0d0f14" stroke-width="3"/>'
        f'<g transform="translate(4 4)" fill="#ffffff">{glyph}</g></svg>'
    )
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()
