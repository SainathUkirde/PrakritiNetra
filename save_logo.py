"""
save_logo.py  — run once to save the attached logo to static/logo.png
Usage:
    python save_logo.py <path_to_downloaded_logo.png>

Or if you have the logo file already, just copy it to:
    forecast_blend/static/logo.png

The dashboard will auto-detect it and use it inline in the header and sidebar.
If the file is absent, a built-in SVG placeholder is shown instead.
"""
import shutil, sys
from pathlib import Path

dest = Path(__file__).parent / "static" / "logo.png"
dest.parent.mkdir(parents=True, exist_ok=True)

if len(sys.argv) > 1:
    src = Path(sys.argv[1])
    if src.exists():
        shutil.copy2(src, dest)
        print(f"Logo saved to {dest}")
    else:
        print(f"Source file not found: {src}")
else:
    print(f"Place the PrakritiNetra logo PNG at:\n  {dest}")
    print("Or run:  python save_logo.py <path_to_logo.png>")
