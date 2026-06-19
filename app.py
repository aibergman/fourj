"""Hosted dashboard entry point for FourJ.

Generic Python hosts can run this as `gunicorn app:server`; Docker can run
`python app.py`.
"""

from __future__ import annotations

import os

from fourj.dashboard import create_app

app = create_app()
server = app.server


if __name__ == "__main__":
    app.run(host=os.environ.get("HOST", "0.0.0.0"), port=int(os.environ.get("PORT", "8050")))
