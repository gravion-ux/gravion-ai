import io
import json
import socket
import threading
import urllib.request

import pytest

from gravion_ai.bots import (
    BotSlotError,
    HttpBotAdapter,
    bot_status,
    validate_bot_manifest,
)
from gravion_ai.cli import main
from gravion_ai.doctor import doctor_ok, run_doctor
from gravion_ai.files import FileAttachmentError, build_file_context, store_uploads
from gravion_ai.licensing import AuditLog, DeploymentIdentity
from gravion_ai.web import make_server


def _ready_config(make_config):
    config = make_config()
    DeploymentIdentity.create_or_load(config.state_dir)
    AuditLog(config.state_dir / "audit.jsonl").append("deployment.init", {})
    return config


def test_doctor_passes_for_initialized_development_deployment(make_config):
    config = _ready_config(make_config)
    checks = run_doctor(config)
    assert doctor_ok(checks)
    assert {check.name for check in checks} >= {"config", "policy", "identity", "audit"}


def test_doctor_fails_production_without_license(make_config):
    config = make_config(mode="production")
    DeploymentIdentity.create_or_load(config.state_dir)
    AuditLog(config.state_dir / "audit.jsonl").append("deployment.init", {})
    checks = run_doctor(config)
    assert not doctor_ok(checks)
    assert any(check.name == "licensing" and check.status == "FAIL" for check in checks)


def test_cli_doctor_and_endpoint_init_validation(tmp_path, monkeypatch, capsys):
    home = tmp_path / "deploy"
    monkeypatch.setenv("GRAVION_HOME", str(home))
    assert main(["init", "--provider", "endpoint"]) == 1
    assert "base_url" in capsys.readouterr().err
    assert main(["init"]) == 0
    capsys.readouterr()
    assert main(["doctor"]) == 0
    assert "PASS config" in capsys.readouterr().out


def test_store_uploads_sanitizes_names_and_builds_context(make_config):
    config = make_config()
    stored = store_uploads(
        config,
        [
            {
                "name": "../notes.txt",
                "mime": "text/plain",
                "text": "alpha beta gamma",
            },
            {
                "name": "image.bin",
                "mime": "application/octet-stream",
                "data_base64": "AAEC",
            },
        ],
    )
    assert stored[0].stored_name == "notes.txt"
    assert stored[0].text_chars == len("alpha beta gamma")
    assert stored[1].text_chars == 0
    context = build_file_context(stored)
    assert "alpha beta gamma" in context
    assert "binary or non-text" in context
    assert (config.data_dir / "uploads").exists()


def test_store_uploads_rejects_bad_payloads(make_config):
    config = make_config()
    with pytest.raises(FileAttachmentError):
        store_uploads(config, [{"name": "x.txt", "mime": "text/plain"}])
    with pytest.raises(FileAttachmentError):
        store_uploads(
            config,
            [{"name": "x.bin", "mime": "application/octet-stream", "data_base64": "!!"}],
        )


def test_bot_slot_empty_reports_needed_inputs(make_config):
    config = make_config()
    status = bot_status(config)
    assert status["configured"] is False
    assert any("bot code" in item for item in status["needed"])


def test_bot_manifest_validation_and_http_adapter(make_config):
    slot = validate_bot_manifest(
        {
            "name": "My Bot",
            "kind": "http",
            "endpoint_url": "http://127.0.0.1:9000/reply",
        }
    )
    assert slot.name == "My Bot"
    with pytest.raises(BotSlotError):
        validate_bot_manifest({"name": "x", "kind": "command", "endpoint_url": "http://x"})

    class FakeResponse(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.close()
            return False

    def fake_opener(request, timeout=None):
        payload = json.loads(request.data.decode("utf-8"))
        assert payload["messages"][0]["content"] == "hello"
        return FakeResponse(json.dumps({"reply": "bot says hi"}).encode("utf-8"))

    adapter = HttpBotAdapter(slot, opener=fake_opener)
    assert adapter.reply([{"role": "user", "content": "hello"}]) == "bot says hi"


def _serve(config):
    server = make_server(config, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def _post_json(url, payload):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


def test_web_chat_accepts_files_and_uses_context(make_config):
    config = _ready_config(make_config)
    server, base = _serve(config)
    try:
        with urllib.request.urlopen(base + "/", timeout=5) as response:
            html = response.read().decode("utf-8")
        assert "Attach files" in html
        status, body = _post_json(
            base + "/api/chat",
            {
                "module": "chat",
                "prompt": "summarize the attachment",
                "files": [
                    {
                        "name": "notes.md",
                        "mime": "text/markdown",
                        "text": "unique-file-marker-123",
                    }
                ],
            },
        )
        assert status == 200
        assert "unique-file-marker-123" in body["answer"]
        assert body["files"][0]["name"] == "notes.md"
        with urllib.request.urlopen(base + "/api/uploads", timeout=5) as response:
            uploads = json.loads(response.read().decode("utf-8"))["uploads"]
        assert uploads[0]["name"] == "notes.md"
    finally:
        server.shutdown()
        server.server_close()


def test_web_dashboard_status_catalog_and_runs(make_config):
    config = _ready_config(make_config)
    server, base = _serve(config)
    try:
        with urllib.request.urlopen(base + "/", timeout=5) as response:
            html = response.read().decode("utf-8")
        assert "Module workspace" in html
        assert "Model catalog" in html
        assert "Run timeline" in html
        assert "Roadmap — not current capability" in html

        with urllib.request.urlopen(base + "/api/status", timeout=5) as response:
            status = json.loads(response.read().decode("utf-8"))
        core = status["catalog"]["core"]
        assert core["parameter_count"] == 10586543
        assert core["weights_status"] == "included"
        assert core["training_examples"] == 5657
        assert core["answer_classes"] == 779
        # All 12 bundled models (core + 11 specialists) ship included weights.
        assert status["catalog"]["summary"]["weights_included"] == 17
        assert any(
            module["name"] == "chat" and module["description"]
            for module in status["modules"]
        )
        assert status["roadmap"]
        assert all(item["status"] == "Roadmap" for item in status["roadmap"])

        status_code, body = _post_json(
            base + "/api/chat",
            {"module": "chat", "prompt": "hello dashboard", "files": []},
        )
        assert status_code == 200
        assert body["run"]["module"] == "chat"
        assert body["run"]["last_event"] == "run.completed"

        with urllib.request.urlopen(base + "/api/models", timeout=5) as response:
            models = json.loads(response.read().decode("utf-8"))
        assert models["core"]["parameter_count"] == 10586543

        with urllib.request.urlopen(base + "/api/runs", timeout=5) as response:
            runs = json.loads(response.read().decode("utf-8"))
        assert runs["latest"][0]["module"] == "chat"
        assert runs["latest"][0]["last_event"] == "run.completed"

        public_text = html + json.dumps(status)
        for forbidden in ("4K", "100x", "100X"):
            assert forbidden not in public_text
    finally:
        server.shutdown()
        server.server_close()


def test_web_bot_route_reports_empty_slot(make_config):
    config = _ready_config(make_config)
    server, base = _serve(config)
    try:
        request = urllib.request.Request(
            base + "/api/chat",
            data=json.dumps({"module": "bot", "prompt": "hello", "files": []}).encode(
                "utf-8"
            ),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            urllib.request.urlopen(request, timeout=5)
        assert excinfo.value.code == 409
        body = json.loads(excinfo.value.read().decode("utf-8"))
        assert "will not recreate" in body["error"]
        assert body["needed"]
    finally:
        server.shutdown()
        server.server_close()


def test_web_rejects_invalid_content_length(make_config):
    config = _ready_config(make_config)
    server, base = _serve(config)
    try:
        host, port = base.removeprefix("http://").split(":")
        with socket.create_connection((host, int(port)), timeout=5) as sock:
            sock.sendall(
                (
                    "POST /api/chat HTTP/1.1\r\n"
                    f"Host: {host}\r\n"
                    "Content-Type: application/json\r\n"
                    "Content-Length: not-a-number\r\n"
                    "Connection: close\r\n\r\n"
                ).encode("ascii")
            )
            # Read to EOF: the server writes headers and body
            # separately, so a single recv can return the
            # headers alone and miss the error body.
            chunks = []
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                chunks.append(chunk)
            response = b"".join(chunks).decode("utf-8", errors="replace")
        assert "400" in response.splitlines()[0]
        assert "Content-Length is invalid" in response
    finally:
        server.shutdown()
        server.server_close()
