"""Local web UI for Gravion AI.

A standard-library HTTP server: a modern one-page dashboard, a small
JSON API, file attachments stored under the deployment data
directory, model/catalog status, a run timeline, and the drop-in bot
slot status. It binds to loopback by default; production deployments
put a TLS reverse proxy and their own auth in front.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .bots import HttpBotAdapter, bot_status, load_bot_slot
from .config import GravionConfig
from .core_model import load_bundled_model
from .errors import GravionError, SuspendedError
from .files import build_file_context, list_uploads, store_uploads
from .runtime import GravionRuntime

MAX_REQUEST_BYTES = 12 * 1024 * 1024
MODULES = [
    "chat",
    "support",
    "coder",
    "image",
    "music",
    "video",
    "website-builder",
    "cyber",
    "agentic",
    "all-in-one",
    "bot",
]

ROADMAP = [
    {
        "title": "Specialist weights bundles",
        "status": "Roadmap",
        "detail": (
            "Train and package specialist Gravion weights with held-out "
            "eval gates before presenting them as included."
        ),
    },
    {
        "title": "Long-form video pipeline",
        "status": "Roadmap",
        "detail": (
            "Scene planning and stitching for longer videos. The current "
            "build provides video briefs and a local short-clip "
            "development renderer."
        ),
    },
    {
        "title": "Hosted production operations",
        "status": "Roadmap",
        "detail": (
            "Public hosting, TLS/auth operations, monitoring and support "
            "processes for production deployments."
        ),
    },
    {
        "title": "Larger Gravion-owned training sets",
        "status": "Roadmap",
        "detail": (
            "Expand local corpora and held-out evaluations before scaling "
            "parameter or capability claims."
        ),
    },
]

INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gravion AI — Workspace</title>
<style>
:root{color-scheme:dark;--bg:#05070d;--panel:rgba(13,19,34,.88);--panel2:rgba(18,27,48,.92);--line:rgba(148,176,255,.22);--text:#f4f7ff;--muted:#a9b8d6;--accent:#7dd3fc;--accent2:#a78bfa;--good:#86efac;--warn:#fde68a;--shadow:0 24px 80px rgba(0,0,0,.45)}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:radial-gradient(circle at 15% -10%,rgba(59,130,246,.35),transparent 34rem),radial-gradient(circle at 90% 0%,rgba(167,139,250,.28),transparent 30rem),linear-gradient(180deg,#070b16,#05070d 55%,#03050a);color:var(--text)}
a{color:var(--accent)}button,select,textarea,input{font:inherit}.shell{max-width:1240px;margin:0 auto;padding:24px 18px 56px}.topbar{display:flex;align-items:center;justify-content:space-between;gap:16px;margin-bottom:22px}.brand{display:flex;align-items:center;gap:12px}.mark{width:42px;height:42px;border-radius:14px;background:linear-gradient(135deg,var(--accent),var(--accent2));display:grid;place-items:center;color:#06111f;font-weight:900;box-shadow:0 12px 34px rgba(125,211,252,.25)}.brand strong{display:block;letter-spacing:-.03em;font-size:18px}.brand span{display:block;color:var(--muted);font-size:12px}.actions{display:flex;gap:10px;align-items:center;flex-wrap:wrap}.status-pill{border:1px solid var(--line);background:rgba(255,255,255,.05);border-radius:999px;padding:8px 12px;color:var(--muted);font-size:13px}.ghost{background:rgba(255,255,255,.06);color:var(--text);border:1px solid var(--line);border-radius:999px;padding:9px 14px;cursor:pointer}.ghost:hover{border-color:rgba(125,211,252,.55)}
.hero{display:grid;grid-template-columns:minmax(0,1.35fr) minmax(280px,.65fr);gap:18px;align-items:stretch;margin-bottom:18px}.card{background:var(--panel);border:1px solid var(--line);border-radius:24px;padding:20px;box-shadow:var(--shadow);backdrop-filter:blur(16px)}.hero-main{padding:30px}.eyebrow{color:var(--accent);text-transform:uppercase;letter-spacing:.16em;font-size:12px;font-weight:800}.hero h1{font-size:clamp(36px,6vw,68px);line-height:.95;letter-spacing:-.06em;margin:12px 0}.lede{color:var(--muted);font-size:17px;line-height:1.65;max-width:720px}.hero-points{display:flex;gap:8px;flex-wrap:wrap;margin-top:18px}.chip{border:1px solid var(--line);background:rgba(255,255,255,.05);border-radius:999px;padding:7px 10px;color:#dbeafe;font-size:13px}.metrics{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}.metric{background:var(--panel2);border:1px solid var(--line);border-radius:18px;padding:16px}.metric b{display:block;font-size:28px;letter-spacing:-.04em}.metric span{color:var(--muted);font-size:13px;line-height:1.45}.metric small{display:block;color:#dbeafe;margin-top:6px}
.grid{display:grid;grid-template-columns:minmax(0,1.25fr) minmax(300px,.75fr);gap:18px}.stack{display:grid;gap:18px}h2{margin:0 0 8px;letter-spacing:-.03em}h3{margin:0 0 10px;letter-spacing:-.02em}.small{color:var(--muted);font-size:13px;line-height:1.6}.section-head{display:flex;justify-content:space-between;gap:14px;align-items:flex-end;margin-bottom:14px}.section-head p{margin:0}.badge{display:inline-flex;align-items:center;gap:6px;border:1px solid var(--line);border-radius:999px;padding:4px 9px;font-size:12px;color:var(--muted);background:rgba(255,255,255,.04)}.badge.good{color:var(--good);border-color:rgba(134,239,172,.35)}.badge.warn{color:var(--warn);border-color:rgba(253,230,138,.35)}
label{display:block;color:var(--muted);font-size:13px;margin:14px 0 7px;font-weight:650}select,textarea,input[type=file]{width:100%;background:#0a1020;color:var(--text);border:1px solid var(--line);border-radius:14px;padding:12px;outline:none}select:focus,textarea:focus,input:focus{border-color:rgba(125,211,252,.7);box-shadow:0 0 0 3px rgba(125,211,252,.14)}textarea{min-height:150px;resize:vertical}.primary{background:linear-gradient(135deg,var(--accent),#c4b5fd);color:#06111f;border:0;border-radius:999px;padding:12px 20px;font-weight:850;cursor:pointer;margin-top:16px;box-shadow:0 14px 34px rgba(125,211,252,.22)}.primary:disabled{opacity:.55;cursor:wait}.out{white-space:pre-wrap;background:#070d1b;border:1px solid var(--line);border-radius:16px;padding:16px;min-height:140px;margin-top:16px;line-height:1.6}.pill{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:4px 9px;margin:3px;color:var(--muted);font-size:12px}.kv{display:grid;grid-template-columns:110px 1fr;gap:8px 12px}.kv dt{color:var(--muted)}.kv dd{margin:0;overflow-wrap:anywhere}
.module-grid,.model-grid,.roadmap-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}.module-card,.model-card,.roadmap-card{background:rgba(255,255,255,.045);border:1px solid var(--line);border-radius:18px;padding:16px;text-align:left;color:var(--text)}.module-card{cursor:pointer}.module-card:hover,.module-card.active{border-color:rgba(125,211,252,.7);background:rgba(125,211,252,.09)}.module-card h3,.model-card h3,.roadmap-card h3{margin-bottom:7px}.module-card p,.model-card p,.roadmap-card p{color:var(--muted);line-height:1.55;font-size:14px}.meta{display:flex;gap:7px;flex-wrap:wrap;margin-top:12px}.two{display:grid;grid-template-columns:1fr 1fr;gap:18px}.timeline{display:grid;gap:10px}.run{border-left:3px solid var(--accent);background:rgba(255,255,255,.04);border-radius:0 14px 14px 0;padding:12px 14px}.run.failed{border-color:#fca5a5}.run.completed{border-color:var(--good)}.run b{display:block}.run span{color:var(--muted);font-size:13px;line-height:1.5}.empty{border:1px dashed var(--line);border-radius:16px;padding:16px;color:var(--muted)}footer{color:var(--muted);font-size:13px;margin-top:22px;line-height:1.6}
@media(max-width:980px){.hero,.grid,.two{grid-template-columns:1fr}.module-grid,.model-grid,.roadmap-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.topbar{align-items:flex-start;flex-direction:column}}
@media(max-width:640px){.shell{padding:16px 12px 40px}.hero-main,.card{padding:18px}.metrics,.module-grid,.model-grid,.roadmap-grid{grid-template-columns:1fr}.kv{grid-template-columns:1fr}.section-head{display:block}.primary{width:100%}}
</style>
</head>
<body><div class="shell">
<header class="topbar"><div class="brand"><div class="mark">G</div><div><strong>Gravion AI</strong><span>Local modular AI workspace</span></div></div><div class="actions"><div id="status" class="status-pill">Loading status…</div><button id="refresh" class="ghost" type="button">Refresh</button></div></header>
<section class="hero">
<div class="card hero-main"><div class="eyebrow">Rename · re-policy · run</div><h1>One modular AI platform for the work in front of you.</h1><p class="lede">Run Gravion modules from a local web workspace, attach files, inspect the model catalog honestly, and see what ran in the deployment timeline.</p><div class="hero-points"><span class="chip">Local-first workspace</span><span class="chip">Module routing</span><span class="chip">File context</span><span class="chip">Run events</span><span class="chip">Bot slot</span></div></div>
<div class="metrics" aria-label="Deployment metrics"><div class="metric"><b id="metric-modules">—</b><span>implemented modules</span><small id="metric-modules-note">Loading registry…</small></div><div class="metric"><b id="metric-models">—</b><span>catalog entries</span><small id="metric-models-note">Loading catalog…</small></div><div class="metric"><b id="metric-core">—</b><span>Gravion Core parameters</span><small id="metric-core-note">Verified from the catalog when weights are included.</small></div><div class="metric"><b id="metric-runs">—</b><span>recorded run events</span><small id="metric-runs-note">Timeline updates after module runs.</small></div></div>
</section>
<main class="grid">
<section class="card" aria-labelledby="workspace-title"><div class="section-head"><div><h2 id="workspace-title">Module workspace</h2><p class="small">Choose a module, add context, and run it inside this deployment.</p></div><span class="badge good">Local server</span></div>
<label for="module">Module</label><select id="module"></select>
<label for="prompt">Prompt</label><textarea id="prompt" placeholder="Ask Gravion…"></textarea>
<label for="files">Attach files</label><input id="files" type="file" multiple>
<div id="attached" class="small"></div>
<button id="send" class="primary" type="button">Run module</button>
<div id="out" class="out" role="status">Ready.</div>
</section>
<aside class="stack">
<section class="card"><h3>Deployment</h3><dl id="deployment" class="kv small"></dl></section>
<section class="card"><h3>Model honesty</h3><p class="small">Gravion Core is shown with the catalog’s verified parameter count only when its weights are marked included. Specialist entries stay labelled as packaging-ready when no trained weights ship in this repository. If you connect an endpoint, that model’s parameters belong to the connected model.</p><div id="core-summary" class="small"></div></section>
<section class="card"><h3>Bot slot</h3><div id="bot" class="small">Loading bot slot…</div></section>
</aside>
</main>
<section class="card" style="margin-top:18px"><div class="section-head"><div><h2>Modules</h2><p class="small">Select a card to load it into the workspace.</p></div><span id="module-count" class="badge">Loading…</span></div><div id="modules" class="module-grid"></div></section>
<div class="two" style="margin-top:18px">
<section class="card"><div class="section-head"><div><h2>Model catalog</h2><p class="small">Registry status and weights position, without inflated claims.</p></div><span id="catalog-count" class="badge">Loading…</span></div><div id="models" class="model-grid" style="grid-template-columns:1fr"></div></section>
<section class="card"><div class="section-head"><div><h2>Run timeline</h2><p class="small">Latest module runs recorded by this deployment.</p></div><span id="run-count" class="badge">Loading…</span></div><div id="timeline" class="timeline"></div></section>
</div>
<section class="card" style="margin-top:18px"><div class="section-head"><div><h2>Roadmap — not current capability</h2><p class="small">These are targets for future work. They are not presented as shipped features.</p></div><span class="badge warn">Roadmap</span></div><div id="roadmap" class="roadmap-grid"></div></section>
<footer>Gravion AI runs locally in this workspace. Production deployments should place TLS and operator authentication in front of the server.</footer>
</div>
<script>
const modules=__MODULES__;
const moduleEl=document.getElementById('module'); modules.forEach(m=>{const o=document.createElement('option');o.value=m;o.textContent=m;moduleEl.appendChild(o)});
const filesEl=document.getElementById('files'), attached=document.getElementById('attached'), out=document.getElementById('out'), send=document.getElementById('send');
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt=n=>typeof n==='number'?n.toLocaleString('en-US'):'—';
const weightsLabel=s=>({'included':'Weights included','packaging_ready_no_weights':'Packaging ready · no weights included','planned_no_weights':'Planned · no weights','operator_supplied':'Operator supplied','not_applicable':'Not applicable','unspecified':'Unspecified'}[s]||s);
filesEl.addEventListener('change',()=>{attached.innerHTML=[...filesEl.files].map(f=>`<span class="pill">${esc(f.name)} · ${fmt(f.size)} bytes</span>`).join('')});
async function readFile(file){const textLike=file.type.startsWith('text/')||/\.(txt|md|json|jsonl|csv|py|js|ts|html|css|xml|ya?ml|toml|log)$/i.test(file.name); if(textLike){return {name:file.name,mime:file.type||'text/plain',text:await file.text()}} const buf=await file.arrayBuffer(); let binary=''; const bytes=new Uint8Array(buf); for(let i=0;i<bytes.length;i+=8192){binary+=String.fromCharCode(...bytes.subarray(i,i+8192))} return {name:file.name,mime:file.type||'application/octet-stream',data_base64:btoa(binary)}}
function renderModules(list){const implemented=list.filter(m=>m.status==='implemented'); document.getElementById('metric-modules').textContent=fmt(implemented.length); document.getElementById('metric-modules-note').textContent=`${fmt(list.length)} registered entries including the bot slot`; document.getElementById('module-count').textContent=`${fmt(implemented.length)} implemented`; document.getElementById('modules').innerHTML=list.map(m=>`<button class="module-card ${m.name===moduleEl.value?'active':''}" type="button" data-module="${esc(m.name)}"><h3>${esc(m.name)}</h3><p>${esc(m.description||'Gravion module')}</p><div class="meta"><span class="badge">${esc(m.type||'module')}</span><span class="badge ${m.status==='implemented'?'good':'warn'}">${esc(m.status||'registered')}</span>${m.version?`<span class="badge">v${esc(m.version)}</span>`:''}</div></button>`).join(''); document.querySelectorAll('[data-module]').forEach(card=>card.addEventListener('click',()=>{moduleEl.value=card.dataset.module; renderModules(list); document.getElementById('prompt').focus()}))}
function renderCatalog(catalog){const models=catalog.models||[]; const included=models.filter(m=>m.weights_status==='included'); document.getElementById('metric-models').textContent=fmt(models.length); document.getElementById('metric-models-note').textContent=`${fmt(included.length)} with included weights`; document.getElementById('catalog-count').textContent=`${fmt(models.length)} entries`; const core=catalog.core||{}; document.getElementById('metric-core').textContent=fmt(core.parameter_count); document.getElementById('metric-core-note').textContent=core.parameter_count?`Weights status: ${weightsLabel(core.weights_status)}`:'No included Core parameter count in this catalog.'; document.getElementById('core-summary').innerHTML=core.name?`<p><strong>${esc(core.name)}</strong> · ${esc(weightsLabel(core.weights_status))}${core.parameter_count?` · ${fmt(core.parameter_count)} parameters`:''}${core.training_examples?` · trained from ${fmt(core.training_examples)} Gravion-owned examples`:''}${core.answer_classes?` across ${fmt(core.answer_classes)} answer classes`:''}.</p>`:''; document.getElementById('models').innerHTML=models.map(m=>`<article class="model-card"><h3>${esc(m.name)}</h3><p>${esc(m.notes||'Gravion registry entry.')}</p><div class="meta"><span class="badge ${m.weights_status==='included'?'good':'warn'}">${esc(weightsLabel(m.weights_status))}</span><span class="badge">${esc(m.status)}</span>${m.parameter_count?`<span class="badge">${fmt(m.parameter_count)} parameters</span>`:''}${m.context_window?`<span class="badge">${fmt(m.context_window)} context</span>`:''}</div><p class="small">Capabilities: ${esc((m.capabilities||[]).join(', '))}</p></article>`).join('')}
function renderRuns(runs){const latest=runs.latest||[]; document.getElementById('metric-runs').textContent=fmt(runs.event_count); document.getElementById('metric-runs-note').textContent=latest.length?`Latest: ${latest[0].module||'module'} · ${latest[0].last_event}`:'No module runs recorded yet.'; document.getElementById('run-count').textContent=`${fmt(latest.length)} recent runs`; document.getElementById('timeline').innerHTML=latest.length?latest.map(r=>`<article class="run ${r.last_event==='run.failed'?'failed':r.last_event==='run.completed'?'completed':''}"><b>${esc(r.module||'module')} · ${esc(r.last_event)}</b><span>${esc(r.started_at||'')} → ${esc(r.last_at||'')}<br>Model: ${esc(r.model||'—')} · Events: ${fmt(r.event_count)} · Run: ${esc(String(r.run_id||'').slice(0,12))}</span></article>`).join(''):'<div class="empty">No runs yet. Run a module and its timeline will appear here.</div>'}
function renderRoadmap(items){document.getElementById('roadmap').innerHTML=(items||[]).map(item=>`<article class="roadmap-card"><span class="badge warn">${esc(item.status)}</span><h3>${esc(item.title)}</h3><p>${esc(item.detail)}</p></article>`).join('')}
async function loadStatus(){const r=await fetch('/api/status'); const s=await r.json(); document.getElementById('status').textContent=`${s.name} · ${s.mode} · ${s.provider}/${s.model} · gate ${s.gate_state}`; document.getElementById('deployment').innerHTML=`<dt>Company</dt><dd>${esc(s.company||'(not set)')}</dd><dt>Tier</dt><dd>${esc(s.tier)}</dd><dt>Provider</dt><dd>${esc(s.provider)}</dd><dt>Model</dt><dd>${esc(s.model)}</dd><dt>Gate</dt><dd>${esc(s.gate_state)}</dd><dt>Uploads</dt><dd>${fmt(s.uploads)} stored</dd>`; const b=s.bot; document.getElementById('bot').innerHTML=b.configured?`Configured: ${esc(b.name)} (${esc(b.kind)})`:`Empty slot. Needed: ${esc((b.needed||[]).join('; '))}`; renderModules(s.modules||[]); renderCatalog(s.catalog||{}); renderRuns(s.runs||{}); renderRoadmap(s.roadmap||[])}
document.getElementById('refresh').addEventListener('click',loadStatus);
send.addEventListener('click',async()=>{send.disabled=true; out.textContent='Working…'; try{const files=[]; for(const f of filesEl.files){files.push(await readFile(f))} const r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({module:moduleEl.value,prompt:document.getElementById('prompt').value,files})}); const body=await r.json(); if(!r.ok){throw new Error(body.error||('HTTP '+r.status))} out.textContent=body.answer+(body.files&&body.files.length?`\n\nFiles stored: ${body.files.map(f=>f.name).join(', ')}`:''); await loadStatus()}catch(e){out.textContent='Error: '+e.message}finally{send.disabled=false}});
loadStatus();
</script></body></html>
"""


def _module_payload(runtime: GravionRuntime) -> list[dict]:
    modules = [
        {
            "name": module.name,
            "type": module.type.value,
            "status": module.status.value,
            "version": module.version,
            "description": module.description,
        }
        for module in runtime.registry.list()
    ]
    modules.append(
        {
            "name": "bot",
            "type": "bot-slot",
            "status": "implemented",
            "version": "",
            "description": (
                "Route to the operator's own configured bot endpoint "
                "when a bot slot is present."
            ),
        }
    )
    return modules


def _model_payload(entry) -> dict:
    return {
        "id": entry.id,
        "name": entry.name,
        "capabilities": list(entry.capabilities),
        "tiers": list(entry.tiers),
        "status": entry.status,
        "context_window": entry.context_window,
        "weights_status": entry.weights_status,
        "parameter_count": entry.parameter_count,
        "weights_artifact": entry.weights_artifact,
        "weights_sha256_short": entry.weights_sha256[:12],
        "notes": entry.notes,
    }


def _catalog_payload(runtime: GravionRuntime) -> dict:
    models = [_model_payload(entry) for entry in runtime.catalog.list()]
    core_entry = runtime.catalog.get("gravion-core")
    core = _model_payload(core_entry) if core_entry is not None else {}
    if core_entry is not None and core_entry.weights_included:
        try:
            artifact = load_bundled_model().artifact
            trained_from = artifact.get("trained_from", {})
            if (
                artifact.get("parameter_count") == core_entry.parameter_count
                and isinstance(trained_from, dict)
            ):
                core["training_examples"] = trained_from.get("examples")
                core["answer_classes"] = trained_from.get("classes")
                core["training_features"] = trained_from.get("features")
        except Exception:
            # The catalog remains authoritative for the parameter
            # count; artifact metadata is enrichment only.
            pass
    return {
        "source": runtime.catalog.source,
        "models": models,
        "core": core,
        "summary": {
            "total": len(models),
            "weights_included": sum(
                1 for model in models if model["weights_status"] == "included"
            ),
            "packaging_ready_no_weights": sum(
                1
                for model in models
                if model["weights_status"] == "packaging_ready_no_weights"
            ),
        },
    }


def _run_payload(runtime: GravionRuntime) -> dict:
    recent_events = [event.to_dict() for event in runtime.events.entries()[-20:]]
    return {
        "event_count": runtime.events.count(),
        "latest": runtime.events.latest_runs(12),
        "recent_events": list(reversed(recent_events)),
    }


def _dashboard_status(runtime: GravionRuntime, config: GravionConfig) -> dict:
    status = runtime.status()
    status["bot"] = bot_status(config)
    status["uploads"] = len(list_uploads(config))
    status["modules_available"] = MODULES
    status["modules"] = _module_payload(runtime)
    status["catalog"] = _catalog_payload(runtime)
    status["runs"] = _run_payload(runtime)
    status["roadmap"] = ROADMAP
    return status


def _handler_factory(config: GravionConfig):
    runtime = GravionRuntime(config)

    class GravionWebHandler(BaseHTTPRequestHandler):
        server_version = "GravionAIWeb/1.1"

        def log_message(self, format, *args):  # noqa: A002 - stdlib name
            return

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, payload: dict) -> None:
            self._send(
                status,
                (json.dumps(payload) + "\n").encode("utf-8"),
                "application/json",
            )

        def _read_json(self) -> dict:
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > MAX_REQUEST_BYTES:
                raise GravionError("request body is missing or too large")
            raw = self.rfile.read(length)
            try:
                data = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise GravionError("request body must be valid JSON") from exc
            if not isinstance(data, dict):
                raise GravionError("request body must be a JSON object")
            return data

        def do_GET(self):  # noqa: N802 - stdlib handler name
            if self.path == "/":
                html = INDEX_HTML.replace("__MODULES__", json.dumps(MODULES))
                self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")
                return
            if self.path == "/api/status":
                self._json(200, _dashboard_status(runtime, config))
                return
            if self.path == "/api/modules":
                self._json(200, {"modules": _module_payload(runtime)})
                return
            if self.path == "/api/models":
                self._json(200, _catalog_payload(runtime))
                return
            if self.path == "/api/runs":
                self._json(200, _run_payload(runtime))
                return
            if self.path == "/api/bot/status":
                self._json(200, bot_status(config))
                return
            if self.path == "/api/uploads":
                self._json(200, {"uploads": list_uploads(config)})
                return
            self._json(404, {"error": "not found"})

        def do_POST(self):  # noqa: N802 - stdlib handler name
            try:
                data = self._read_json()
                if self.path == "/api/files":
                    stored = store_uploads(config, data.get("files"))
                    self._json(
                        200,
                        {"files": [item.public_dict() for item in stored]},
                    )
                    return
                if self.path != "/api/chat":
                    self._json(404, {"error": "not found"})
                    return
                module = data.get("module", "chat")
                prompt = data.get("prompt", "")
                if module not in MODULES:
                    raise GravionError(f"unknown module {module!r}")
                if not isinstance(prompt, str) or not prompt.strip():
                    raise GravionError("prompt must be a non-empty string")
                stored = store_uploads(config, data.get("files"))
                context = build_file_context(stored)
                full_prompt = prompt.strip()
                if context:
                    full_prompt += "\n\n" + context
                if module == "bot":
                    slot = load_bot_slot(config)
                    if slot is None:
                        status = bot_status(config)
                        self._json(
                            409,
                            {
                                "error": "bot slot is empty; Gravion will not recreate your bot",
                                "needed": status.get("needed", []),
                            },
                        )
                        return
                    answer = HttpBotAdapter(slot).reply(
                        [
                            {"role": "system", "content": config.system_context()},
                            {"role": "user", "content": full_prompt},
                        ],
                        files=[
                            {**item.public_dict(), "text_preview": item.text_preview}
                            for item in stored
                        ],
                    )
                    try:
                        runtime.events.append(
                            "run.completed",
                            module="bot",
                            model=slot.name,
                            details={"output_chars": len(answer)},
                        )
                    except (OSError, ValueError):
                        pass
                else:
                    answer = runtime.run_module(module, full_prompt)
                latest_runs = runtime.events.latest_runs(1)
                self._json(
                    200,
                    {
                        "answer": answer,
                        "module": module,
                        "files": [item.public_dict() for item in stored],
                        "context_chars": len(context),
                        "run": latest_runs[0] if latest_runs else None,
                    },
                )
            except SuspendedError as exc:
                self._json(403, {"error": str(exc)})
            except GravionError as exc:
                self._json(400, {"error": str(exc)})

    return GravionWebHandler


def make_server(
    config: GravionConfig, *, host: str = "127.0.0.1", port: int = 8791
) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), _handler_factory(config))
