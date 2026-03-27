"""
ASGI entry for ``uvicorn`` (see root ``Procfile``).

Wraps the Flask app in ``api.server`` with WSGI→ASGI.
"""

from __future__ import annotations

from asgiref.wsgi import WsgiToAsgi

from api.server import app as flask_app

app = WsgiToAsgi(flask_app)
