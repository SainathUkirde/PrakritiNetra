"""
logo_loader.py
==============
Loads the PrakritiNetra logo and returns it as a base64 data URI for
inline HTML embedding.  Falls back to an SVG placeholder if the PNG is
not found (so the dashboard always starts, even before the logo file is
placed in static/).
"""
from __future__ import annotations
import base64
from pathlib import Path

# Prefer the tight-cropped version (white-padding removed); fall back to original
_LOGO_PATH = (
    Path(__file__).parent / "static" / "logo_cropped.png"
    if (Path(__file__).parent / "static" / "logo_cropped.png").exists()
    else Path(__file__).parent / "static" / "logo.png"
)

# Minimal SVG fallback: eye + leaf motif matching the PrakritiNetra visual
_SVG_FALLBACK = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 80 80" width="52" height="52">'
    '<circle cx="40" cy="40" r="38" fill="none" stroke="#2563eb" stroke-width="3"/>'
    '<ellipse cx="40" cy="36" rx="22" ry="14" fill="none" stroke="#1d4ed8" stroke-width="2.5"/>'
    '<circle cx="40" cy="36" r="9" fill="#2563eb"/>'
    '<circle cx="40" cy="36" r="5" fill="#fff"/>'
    '<ellipse cx="40" cy="56" rx="16" ry="8" fill="#16a34a" opacity="0.85"/>'
    '<ellipse cx="30" cy="54" rx="12" ry="6" fill="#15803d" opacity="0.7" transform="rotate(-20 30 54)"/>'
    '<ellipse cx="50" cy="54" rx="12" ry="6" fill="#15803d" opacity="0.7" transform="rotate(20 50 54)"/>'
    '</svg>'
)


def get_logo_html(height_px: int = 52) -> str:
    """
    Return an <img> or <svg> tag for the PrakritiNetra logo.

    If static/logo.png exists, returns a base64-embedded <img>.
    Otherwise returns the inline SVG fallback.

    We set only `height` and let `width: auto` so the natural aspect ratio
    is preserved — forcing a square box was cropping the top of the circular logo.
    """
    if _LOGO_PATH.exists():
        data = _LOGO_PATH.read_bytes()
        b64  = base64.b64encode(data).decode()
        # Cropped PNG is 1136×1059 → ratio ≈ 1.073 wide:tall
        # Explicit width prevents the flex row from squeezing the logo
        display_w = int(height_px * 1.073)
        return (
            f'<div style="flex-shrink:0; min-width:{display_w}px; '
            f'width:{display_w}px; height:{height_px}px; '
            f'margin-right:14px; overflow:visible;">'
            f'<img src="data:image/png;base64,{b64}" '
            f'style="width:{display_w}px; height:{height_px}px; '
            f'object-fit:fill; display:block; border-radius:4px;" '
            f'alt="PrakritiNetra logo">'
            f'</div>'
        )
    # SVG fallback
    display_w = height_px
    svg_sized = _SVG_FALLBACK.replace('width="52"', f'width="{display_w}"') \
                              .replace('height="52"', f'height="{height_px}"')
    return (
        f'<div style="flex-shrink:0; min-width:{display_w}px; '
        f'width:{display_w}px; height:{height_px}px; margin-right:14px;">'
        f'{svg_sized}</div>'
    )
