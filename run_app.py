"""Start the API and the web UI together.  Usage: python run_app.py [--port 8000] [--open]"""
import argparse
import threading
import webbrowser

import uvicorn

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--open", action="store_true", help="open the browser when ready")
    a = ap.parse_args()
    if a.open:
        threading.Timer(2.5, lambda: webbrowser.open(f"http://{a.host}:{a.port}")).start()
    uvicorn.run("api.main:app", host=a.host, port=a.port)
