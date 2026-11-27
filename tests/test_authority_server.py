import dataclasses
import io
import json
import socket
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from gravion_ai.errors import LicenseError
from gravion_ai.licensing.authority import (
    EntitlementRequest,
    HttpAuthorityClient,
    build_entitlement_request,
    verify_entitlement_token,
)
from gravion_ai.licensing.authority_server import (
    MAX_REQUEST_BYTES,
    AuthorityService,
    load_license_documents,
    make_server,
)
from gravion_ai.runtime import GravionRuntime


@pytest.fixture()
def served(issuer, licensed_home, tmp_path):
    """A running authority service holding one registered license.

    Yields (base_url, config, document): the production config
    from licensed_home, and its license document — which the
    service loaded from a licenses directory, as an operator
    would lay it out.
    """
    config, _mock, document = licensed_home(modules=("chat", "coder"))
    licenses_dir = tmp_path / "licenses"
    licenses_dir.mkdir()
    (licenses_dir / "acme.json").write_text(
        json.dumps(document), encoding="utf-8"
    )
    server = make_server(issuer, licenses_dir, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        yield f"http://127.0.0.1:{port}", config, document
    finally:
        server.shutdown()
        server.server_close()


def _get(url):
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def _post(url, payload):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def test_health_endpoint_reports_authority_and_license_count(served, issuer):
    base_url, _config, _document = served
    status, body = _get(base_url + "/health")
    assert status == 200
    assert body == {"status": "ok", "authority": issuer.key_id, "licenses": 1}


def test_entitlement_round_trip_over_http(served, issuer):
    base_url, config, document = served
    from gravion_ai.licensing.identity import DeploymentIdentity

    identity = DeploymentIdentity.create_or_load(config.state_dir)
    client = HttpAuthorityClient(base_url)
    request = build_entitlement_request(identity.deployment_id, document, "chat")
    token = client.fetch_token(request)
    payload = verify_entitlement_token(
        token, issuer, module="chat", deployment_id=identity.deployment_id
    )
    assert payload["holder"] == "Acme Robotics"
    assert "coder" in payload["modules"]


def test_unknown_license_is_refused(served, issuer):
    base_url, config, _document = served
    from gravion_ai.licensing.identity import DeploymentIdentity
    from gravion_ai.licensing.license_doc import issue_license

    identity = DeploymentIdentity.create_or_load(config.state_dir)
    other = issue_license(
        issuer,
        deployment_id=identity.deployment_id,
        holder="Someone Else",
        tier="tier-2-basic",
        modules=["chat"],
        agreement_reference="GRAVION-CA-2026-999",
        equity_percent=15,
        days=30,
    )
    client = HttpAuthorityClient(base_url)
    request = build_entitlement_request(identity.deployment_id, other, "chat")
    with pytest.raises(LicenseError):
        client.fetch_token(request)


def test_extra_request_field_is_a_400(served):
    base_url, _config, _document = served
    status, body = _post(
        base_url + "/v1/entitlement",
        {
            "deployment_id": "x",
            "license_hash": "y",
            "module": "chat",
            "requested_at": "2026-01-01T00:00:00Z",
            "telemetry": "not allowed",
        },
    )
    assert status == 400
    assert "malformed" in body["error"]


def test_unknown_paths_are_404(served):
    base_url, _config, _document = served
    status, _body = _get(base_url + "/v1/everything")
    assert status == 404
    status, _body = _post(base_url + "/health", {})
    assert status == 404


def test_production_gate_runs_against_the_live_service(served):
    """The full Piece 9 loop: a production deployment's gate
    fetches its entitlement from the hosted service over HTTP and
    allows the module."""
    base_url, config, _document = served
    config = dataclasses.replace(config, authority_url=base_url)
    runtime = GravionRuntime(config)
    decision = runtime.gate.check("chat")
    assert decision.allowed
    # And the token is cached for the offline grace path.
    assert (config.state_dir / "entitlement-token.json").exists()


def test_load_license_documents_missing_dir_and_bad_file(tmp_path):
    assert load_license_documents(tmp_path / "absent") == []
    bad = tmp_path / "bad.json"
    bad.write_text("{oops", encoding="utf-8")
    with pytest.raises(LicenseError, match="not valid JSON"):
        load_license_documents(tmp_path)


def test_service_starts_empty_and_refuses_everything(issuer, tmp_path):
    service = AuthorityService(issuer, tmp_path / "empty")
    assert service.health()["licenses"] == 0
    with pytest.raises(LicenseError, match="unknown license"):
        service.handle_entitlement(
            {
                "deployment_id": "x",
                "license_hash": "0" * 64,
                "module": "chat",
                "requested_at": "2026-01-01T00:00:00Z",
            }
        )


def _raw_post(base_url, content_length):
    host, port = base_url.removeprefix("http://").split(":")
    with socket.create_connection((host, int(port)), timeout=5) as sock:
        sock.sendall(
            (
                "POST /v1/entitlement HTTP/1.1\r\n"
                f"Host: {host}\r\n"
                "Content-Type: application/json\r\n"
                f"Content-Length: {content_length}\r\n"
                "Connection: close\r\n\r\n"
            ).encode("ascii")
        )
        chunks = []
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            chunks.append(chunk)
    return b"".join(chunks).decode("utf-8", errors="replace")


def test_authority_rejects_invalid_and_oversized_content_length(served):
    base_url, _config, _document = served
    assert "400" in _raw_post(base_url, "not-a-number").splitlines()[0]
    assert "400" in _raw_post(base_url, MAX_REQUEST_BYTES + 1).splitlines()[0]


class _FakeAuthorityResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def test_authority_client_caps_response_and_rejects_invalid_utf8():
    request = EntitlementRequest(
        deployment_id="deployment",
        license_hash="0" * 64,
        module="chat",
        requested_at="2026-01-01T00:00:00Z",
    )

    def oversized_opener(_request, timeout=None):
        return _FakeAuthorityResponse(
            b"x" * (HttpAuthorityClient.MAX_RESPONSE_BYTES + 1)
        )

    client = HttpAuthorityClient("http://authority.test", opener=oversized_opener)
    with pytest.raises(LicenseError, match="1 MiB safety limit"):
        client.fetch_token(request)

    def invalid_utf8_opener(_request, timeout=None):
        return _FakeAuthorityResponse(b"\xff")

    client = HttpAuthorityClient("http://authority.test", opener=invalid_utf8_opener)
    with pytest.raises(LicenseError, match="unreachable"):
        client.fetch_token(request)
