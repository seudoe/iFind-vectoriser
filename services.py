"""
Base URLs of the other services, read from the environment on every call (nothing hardcoded).

  MAIN_PROJECT_URL   the main project (ifind); comma-separated for several origins
  INTERNSCRAPER_URL  internScraper

Both are the browsers/services allowed to call this API from a web page (CORS). Unset = no browser origin is
allowed; server-to-server calls (ifind, internScraper) are not affected by CORS.
"""

import os
from typing import List


def _urls(name: str) -> List[str]:
    return [u for u in (p.strip().rstrip("/") for p in os.environ.get(name, "").split(",")) if u]


def main_project_urls() -> List[str]:
    return _urls("MAIN_PROJECT_URL")


def internscraper_urls() -> List[str]:
    return _urls("INTERNSCRAPER_URL")


def allowed_origins() -> List[str]:
    return [*main_project_urls(), *internscraper_urls()]
