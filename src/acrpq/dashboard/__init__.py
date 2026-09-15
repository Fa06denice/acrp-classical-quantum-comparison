"""Air-traffic-control style dashboard for the ACRP benchmark (OPTIONAL).

A FastAPI backend exposes the ``acrpq`` package as JSON and serves a vanilla-JS
radar front-end. Requires the ``[dashboard]`` extra (FastAPI + uvicorn); the
quantum solver path additionally needs the ``[quantum]`` extra. Nothing here is
imported by the core package.

Run it with::

    acrpq dashboard           # -> http://127.0.0.1:8000
"""

from __future__ import annotations

__all__ = ["create_app", "run"]


def create_app():
    """Build and return the FastAPI application (lazy import of FastAPI)."""
    from .api import create_app as _create

    return _create()


def run(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Launch the dashboard with uvicorn."""
    from .api import run as _run

    _run(host=host, port=port)
