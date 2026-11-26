# Gravion AI web UI

Run the local web workspace:

```bash
gravion web serve --host 127.0.0.1 --port 8791
```

Open `http://127.0.0.1:8791`.

## What it does

- Presents a modern local dashboard for the deployment: workspace,
  module cards, model catalog, run timeline and roadmap status.
- Chat with any implemented Gravion module.
- Attach files in the browser. Text-like files are decoded and added
  to the prompt context; binary files are stored with metadata and
  marked as non-text in the context.
- Store uploads under `<deployment-home>/data/uploads/<batch-id>/`
  with sanitized filenames, SHA-256 hashes and a `manifest.json`.
- Show deployment status, the drop-in bot slot status, catalog
  weights status and recent run events.
- Route the `bot` module to the operator's own bot endpoint when a
  bot slot is configured.

The dashboard separates current capability from roadmap targets.
Roadmap items are labelled as roadmap work, not shipped features.
Gravion Core parameter counts are shown only from the model catalog
when the catalog marks the weights as included; specialist entries
remain labelled according to their catalog weights status.

## API

- `GET /api/status` — dashboard status: deployment, modules,
  catalog, runs, roadmap, bot slot and upload count.
- `GET /api/modules` — module registry cards for the dashboard.
- `GET /api/models` — model catalog payload with weights status.
- `GET /api/runs` — latest run summaries and recent run events.
- `GET /api/uploads` — stored upload metadata.
- `POST /api/files` — store files without running a module.
- `POST /api/chat` — run a module with `{module, prompt, files}`.

File payloads use `{name, mime, text}` for text files or
`{name, mime, data_base64}` for binary files. Limits: 8 files per
request, 2 MiB per file, 8 MiB total, 24,000 context characters.

## Safety

The server binds to loopback by default, caps request bodies, sends
no-store/nosniff headers, sanitizes filenames, and never executes
uploaded content. For public production use, put a TLS reverse
proxy and operator authentication in front of it.
