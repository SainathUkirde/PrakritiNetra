"""
encode_logo.py  — one-time helper
Reads static/logo.png and prints the base64 data URI for embedding in app.py.
Run:  python encode_logo.py
Then copy the printed string into LOGO_B64 in app.py.
"""
import base64, pathlib

logo_path = pathlib.Path(__file__).parent / "static" / "logo.png"
if not logo_path.exists():
    print("ERROR: place the logo at forecast_blend/static/logo.png first.")
else:
    data = logo_path.read_bytes()
    b64  = base64.b64encode(data).decode()
    print(f"data:image/png;base64,{b64}")
