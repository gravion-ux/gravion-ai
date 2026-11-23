"""The hosted licensing authority service (Piece 9).

Piece 1 shipped the entitlement protocol and both ends of it in
process: `MockAuthority` (the reference server logic) and
`HttpAuthorityClient` (the deployment side). This module is the
operated form of the server end: a small, dependency-free HTTP
service (Python standard library only) that

- loads the issuer key (server-side, never distributed),
- loads every issued license document from a licenses directory
  and registers it by hash,
- answers `POST /v1/entitlement` with a signed entitlement token,
  re-verifying the license on every request — signature, expiry,
  deployment binding, module entitlement — exactly like the
  reference authority it wraps,
- answers `GET /health` with the authority's key id and license
  count, and nothing else,
- refuses malformed requests (400) and unentitled ones (403)
  with a JSON error, and 404s everything else.

Privacy is the protocol's, unchanged: a request carries exactly
the four pinned fields (see authority.py); the service logs
nothing about requests and stores nothing beyond the licenses it
was started with.

Honest scope: this is the service implementation, ready to run
(`gravion authority serve`). Where it is publicly hosted, on
whose infrastructure, is an operations decision for the project
owner — the code does not pretend a deployment exists that does
not.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from ..errors import LicenseError
from .authority import MockAuthority, validate_request_payload
from .keys import KeyPair

ENTITLEMENT_PATH = "/v1/entitlement"
HEALTH_PATH = "/health"


def load_license_documents(licenses_dir) -> list[dict]:
    """Load every license document (*.json) from a directory.

    A missing directory is an empty registry (the service starts
    and refuses every request as unknown-license); a malformed
    file is an operator error and raises.
    """
    directory = Path(licenses_dir)
    if not directory.exists():
        return []
    documents = []
    for path in sorted(directory.glob("*.json")):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise LicenseError(
                f"license file {path} is not valid JSON: {exc}"
            ) from exc
        if not isinstance(document, dict):
            raise LicenseError(f"license file {path} is not a JSON object")
        documents.append(document)
    return documents


class AuthorityService:
    """The authority logic behind the HTTP front door: a
    MockAuthority primed with the licenses from a directory."""

    def __init__(self, issuer: KeyPair, licenses_dir):
        self._authority = MockAuthority(issuer)
        self._issuer = issuer
        self.license_count = 0
        for document in load_license_documents(licenses_dir):
            self._authority.register_license(document)
            self.license_count += 1

    @property
    def authority_key_id(self) -> str:
        return self._issuer.key_id

    def health(self) -> dict:
        return {
            "status": "ok",
            "authority": self.authority_key_id,
            "licenses": self.license_count,
        }

    def handle_entitlement(self, payload: dict) -> dict:
        return self._authority.handle_request(payload)


def make_handler(service: AuthorityService):
    """Build a request handler class bound to one service."""

    class AuthorityHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format, *args):  # noqa: A002 - stdlib name
            pass  # the service logs nothing about requests

        def _send_json(self, status: int, document: dict) -> None:
            body = json.dumps(document).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802 - stdlib name
            if self.path == HEALTH_PATH:
                self._send_json(200, service.health())
            else:
                self._send_json(404, {"error": f"unknown path {self.path!r}"})

        def do_POST(self):  # noqa: N802 - stdlib name
            if self.path != ENTITLEMENT_PATH:
                self._send_json(404, {"error": f"unknown path {self.path!r}"})
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("request body must be a JSON object")
                validate_request_payload(payload)
            except (ValueError, LicenseError) as exc:
                self._send_json(400, {"error": f"malformed entitlement request: {exc}"})
                return
            try:
                token = service.handle_entitlement(payload)
            except LicenseError as exc:
                self._send_json(403, {"error": str(exc)})
                return
            self._send_json(200, token)

    return AuthorityHandler


def make_server(
    issuer: KeyPair,
    licenses_dir,
    host: str = "127.0.0.1",
    port: int = 8790,
) -> ThreadingHTTPServer:
    """Create (but do not start) the authority HTTP server."""
    service = AuthorityService(issuer, licenses_dir)
    server = ThreadingHTTPServer((host, port), make_handler(service))
    server.daemon_threads = True
    return server
