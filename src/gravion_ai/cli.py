"""The `gravion` command line.

    gravion init                     create a deployment
    gravion status                   deployment + gate state
    gravion modules list             the module catalogue
    gravion models list              the Gravion model registry
    gravion chat "hello"             run the chat module
    gravion support "where is..."    run the support module
    gravion coder "review: ..."      run the coder module
    gravion image "a logo for..."    run the image module
    gravion music "a jingle for..."  run the music module
    gravion video "a teaser for..."  run the video module
    gravion video "a teaser..." --out clip.mp4
                                     render a real MP4 locally
    gravion website "a site for..."  run the website-builder module
    gravion website "a site..." --out site-dir
                                     build a real static site locally
    gravion cyber "analyse..."       run the cyber module
    gravion agent "plan an agent..." run the agentic module
    gravion agent "plan..." --plan-out plan.json
                                     write a local agent plan file
    gravion bundles list             show tier bundle completeness
    gravion bundles build --tier tier-1-complete --out b.json
                                     write a tier bundle descriptor
    gravion train validate --job job.json
                                     validate a training job
    gravion train prepare --data examples.jsonl --out dataset/
                                     prepare a fine-tuning dataset
    gravion train run --job job.json run a training job (needs the
                                     hosted training pipeline)
    gravion omni "coder: fix..."     run the all-in-one module
                                     (routes by directive, else chat)
    gravion authority serve --issuer-key issuer.key --licenses DIR
                                     run the hosted licensing
                                     authority service (Piece 9)
    gravion upgrade grant|revoke|status|apply|rollback
                                     governed self-upgrade (§9):
                                     permission, signed update
                                     packages, rollback
    gravion export --out file.zip    full data export (always allowed)

    gravion license identity         create/show the deployment identity
    gravion license issuer-keygen    (owner) create the issuer keypair
    gravion license issue            (owner) sign a license
    gravion license install          install a license into a deployment
    gravion license verify           verify a license file
    gravion license sign-module      (owner) sign a module manifest
    gravion license audit verify     verify the tamper-evident audit log
    gravion license audit show       print audit entries
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .catalog import ModelCatalog
from .config import GravionConfig, ProviderConfig, default_home
from .errors import GravionError
from .licensing import (
    AuditLog,
    DeploymentIdentity,
    KeyPair,
    generate_keypair,
    issue_license,
    load_license,
    verify_license,
)
from .licensing.gate import LICENSE_FILENAME
from .licensing.modules_sig import build_manifest, sign_manifest, write_signed_manifest
from .registry import default_registry
from .runtime import GravionRuntime
from .tiers import TIERS

POLICY_TEMPLATE = """\
# Company policy — {company}

Replace this file with your company's work policy: what the assistant
may and may not do, tone of voice, escalation rules, and any facts it
should always know. Gravion injects this text into every module's
system context, so the same modules behave as *your* company's AI.

## Rules

- (example) Never share internal pricing with customers.
- (example) Escalate refund requests above 500 to a human.
"""


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _home(args) -> Path:
    return Path(args.home).expanduser() if args.home else default_home()


def _load_config(args) -> GravionConfig:
    return GravionConfig.load(_home(args))


# ---------------------------------------------------------------------------
# core commands
# ---------------------------------------------------------------------------

def cmd_init(args) -> int:
    home = _home(args)
    config = GravionConfig(
        name=args.name,
        company=args.company,
        mode=args.mode,
        tier=args.tier,
        provider=ProviderConfig(type=args.provider, model=args.model),
        home=home,
    )
    config.save()
    if not config.policy_path.exists():
        config.policy_path.write_text(
            POLICY_TEMPLATE.format(company=args.company or "your company"),
            encoding="utf-8",
        )
    identity = DeploymentIdentity.create_or_load(config.state_dir)
    AuditLog(config.state_dir / "audit.jsonl").append(
        "deployment.init",
        {"name": config.name, "mode": config.mode, "tier": config.tier},
    )
    print(f"Gravion deployment created at {home}")
    print(f"  name:          {config.name}")
    print(f"  mode:          {config.mode} (development needs no license)")
    print(f"  tier:          {config.tier}")
    print(f"  deployment id: {identity.deployment_id}")
    print(f"  policy file:   {config.policy_path} (edit this — it is your company policy)")
    return 0


def cmd_status(args) -> int:
    runtime = GravionRuntime(_load_config(args))
    info = runtime.status()
    print(f"name:       {info['name']}")
    print(f"company:    {info['company'] or '(not set)'}")
    print(f"mode:       {info['mode']}")
    print(f"tier:       {info['tier']}")
    print(f"provider:   {info['provider']} (model: {info['model']})")
    print(f"gate state: {info['gate_state']}")
    if info["gate_reason"]:
        print(f"gate reason: {info['gate_reason']}")
    implemented = [m["name"] for m in info["modules"] if m["status"] == "implemented"]
    planned = [m["name"] for m in info["modules"] if m["status"] == "planned"]
    print(f"modules:    implemented={implemented} planned={planned}")
    return 0


def cmd_modules_list(args) -> int:
    registry = default_registry()
    for m in registry.list():
        print(f"{m.name:<16} {m.type.value:<16} {m.status.value:<12} {m.description}")
    return 0


def cmd_models_list(args) -> int:
    home = _home(args)
    catalog = (
        ModelCatalog.load_for_home(home)
        if (home / "gravion.json").exists()
        else ModelCatalog.load()
    )
    entries = (
        catalog.models_for(args.capability) if args.capability else catalog.list()
    )
    if not entries:
        print(f"no Gravion models serve capability {args.capability!r}")
        return 0
    for e in entries:
        caps = ",".join(e.capabilities)
        print(f"{e.id:<16} {e.name:<16} {e.status:<10} caps={caps}")
    return 0


def cmd_chat(args) -> int:
    runtime = GravionRuntime(_load_config(args))
    print(runtime.chat(" ".join(args.prompt)))
    return 0


def cmd_support(args) -> int:
    runtime = GravionRuntime(_load_config(args))
    print(runtime.support(" ".join(args.prompt)))
    return 0


def cmd_coder(args) -> int:
    runtime = GravionRuntime(_load_config(args))
    print(runtime.coder(" ".join(args.prompt)))
    return 0


def cmd_image(args) -> int:
    runtime = GravionRuntime(_load_config(args))
    print(runtime.image(" ".join(args.prompt)))
    return 0


def cmd_music(args) -> int:
    runtime = GravionRuntime(_load_config(args))
    print(runtime.music(" ".join(args.prompt)))
    return 0


def cmd_video(args) -> int:
    runtime = GravionRuntime(_load_config(args))
    prompt = " ".join(args.prompt)
    if args.out:
        result = runtime.render_video(prompt, Path(args.out))
        print(f"video written to {result.path}")
        print(f"  duration:   {result.duration_seconds} seconds")
        print(f"  resolution: {result.width}x{result.height}")
        print(f"  provider:   {result.provider}")
        return 0
    print(runtime.video(prompt))
    return 0


def cmd_website(args) -> int:
    runtime = GravionRuntime(_load_config(args))
    prompt = " ".join(args.prompt)
    if args.out:
        result = runtime.build_website(prompt, Path(args.out))
        print(f"website written to {result.path}")
        print(f"  title:    {result.title}")
        print(f"  pages:    {', '.join(result.pages)}")
        print(f"  files:    {', '.join(result.files)}")
        print(f"  provider: {result.provider}")
        return 0
    print(runtime.website_builder(prompt))
    return 0


def cmd_cyber(args) -> int:
    runtime = GravionRuntime(_load_config(args))
    print(runtime.cyber(" ".join(args.prompt)))
    return 0


def cmd_agent(args) -> int:
    runtime = GravionRuntime(_load_config(args))
    prompt = " ".join(args.prompt)
    if args.plan_out:
        result = runtime.plan_agent(prompt, Path(args.plan_out))
        print(f"agent plan written to {result.path}")
        print(f"  task:     {result.plan.task}")
        print(f"  tools:    {', '.join(result.plan.tools) or 'none stated'}")
        print(f"  steps:    {len(result.plan.steps)} ({result.plan.steps_source})")
        print(f"  status:   {result.plan.status}")
        print(f"  provider: {result.provider}")
        return 0
    print(runtime.agentic(prompt))
    return 0


def cmd_omni(args) -> int:
    runtime = GravionRuntime(_load_config(args))
    print(runtime.all_in_one(" ".join(args.prompt)))
    return 0


def cmd_authority_serve(args) -> int:
    from .licensing.authority_server import make_server
    from .licensing.keys import KeyPair

    issuer = KeyPair.from_private_pem(Path(args.issuer_key).read_bytes())
    server = make_server(issuer, args.licenses, host=args.host, port=args.port)
    host, port = server.server_address[:2]
    print(f"gravion licensing authority serving on http://{host}:{port}")
    print(f"  authority key: {issuer.key_id}")
    print(f"  licenses dir:  {args.licenses}")
    print("  POST /v1/entitlement — GET /health — Ctrl-C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nauthority stopped")
    finally:
        server.server_close()
    return 0


def cmd_upgrade(args) -> int:
    from .upgrades import UpgradeService

    config = _load_config(args)
    service = UpgradeService(config)
    if args.action == "grant":
        service.grant()
        print("self-upgrade permission GRANTED (recorded in config + audit log)")
        return 0
    if args.action == "revoke":
        service.revoke()
        print("self-upgrade permission REVOKED (recorded in config + audit log)")
        return 0
    if args.action == "status":
        status = service.status()
        print(f"self-upgrade permission: "
              f"{'granted' if status['permission'] else 'not granted'}")
        if status["backups"]:
            for module, versions in status["backups"].items():
                print(f"  retained {module}: {', '.join(versions)}")
        else:
            print("  no retained module versions")
        return 0
    if args.action == "apply":
        result = service.apply(Path(args.package))
        print(f"updated {result['module']}: "
              f"{result['from_version']} → {result['to_version']} "
              f"({result['kind']})")
        print(f"  changelog: {result['changelog_path']}")
        print(f"  backup:    {result['backup_path']}")
        return 0
    result = service.rollback(args.module)
    print(f"rolled back {result['module']} to {result['restored_version']}")
    return 0


def cmd_bundles(args) -> int:
    from .bundles import all_bundle_specs, build_bundle

    if args.action == "list":
        for spec in all_bundle_specs():
            state = "complete" if spec["complete"] else "INCOMPLETE"
            line = f"{spec['tier_label']}: {state}"
            if spec["gaps"]:
                line += f" — gaps: {', '.join(spec['gaps'])}"
            print(line)
        return 0
    document = build_bundle(args.tier, Path(args.out))
    print(f"bundle written to {args.out}")
    print(f"  tier: {document['bundle']['tier_label']}")
    print(f"  hash: {document['bundle_hash']}")
    return 0


def cmd_train(args) -> int:
    from .training.jobs import load_job, run_training_job, validate_job

    if args.action == "prepare":
        from .training.datasets import prepare_dataset

        report = prepare_dataset(
            args.data, Path(args.out),
            validation_fraction=args.validation_fraction,
        )
        print(f"dataset prepared in {args.out}")
        print(f"  examples:   {report['unique_examples']} unique "
              f"({report['duplicates_removed']} duplicates removed)")
        print(f"  train:      {report['train_count']}")
        print(f"  validation: {report['validation_count']}")
        return 0
    job = validate_job(load_job(args.job))
    if args.action == "validate":
        print(f"training job valid: model {job.model}, data {job.data}, "
              f"base tier {job.base_tier}")
        return 0
    run_training_job(job)  # raises honestly: no backend in this build
    return 0


def cmd_export(args) -> int:
    runtime = GravionRuntime(_load_config(args))
    included = runtime.export_bundle(Path(args.out))
    print(f"export written to {args.out} ({len(included)} files):")
    for name in included:
        print(f"  {name}")
    return 0


# ---------------------------------------------------------------------------
# license commands
# ---------------------------------------------------------------------------

def cmd_license_identity(args) -> int:
    config = _load_config(args)
    existed = DeploymentIdentity.exists(config.state_dir)
    identity = DeploymentIdentity.create_or_load(config.state_dir)
    print(f"deployment id: {identity.deployment_id}")
    print(f"state dir:     {config.state_dir}")
    print("identity existed already" if existed else "new identity created")
    print("the private key never leaves this machine (state/identity.key, mode 0600)")
    return 0


def cmd_license_issuer_keygen(args) -> int:
    out = Path(args.out)
    keypair = generate_keypair()
    keypair.save_private(out / "issuer.key")
    keypair.save_public(out / "issuer.pub")
    print(f"issuer keypair written to {out}/")
    print(f"issuer fingerprint: {keypair.key_id}")
    print("keep issuer.key secret and offline; distribute issuer.pub to deployments")
    return 0


def cmd_license_issue(args) -> int:
    issuer = KeyPair.from_private_pem(Path(args.issuer_key).read_bytes())
    document = issue_license(
        issuer,
        deployment_id=args.deployment_id,
        holder=args.holder,
        tier=args.tier,
        modules=[m.strip() for m in args.modules.split(",") if m.strip()],
        agreement_reference=args.agreement_ref,
        equity_percent=args.equity,
        days=args.days,
    )
    out = Path(args.out) if args.out else _home(args) / "state" / LICENSE_FILENAME
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"license written to {out}")
    print(f"  holder:    {args.holder}")
    print(f"  tier:      {args.tier}")
    print(f"  modules:   {document['payload']['modules']}")
    print(f"  expires:   {document['payload']['expires_at']}")
    print(f"  agreement: {args.agreement_ref} (equity {args.equity}%)")
    return 0


def cmd_license_install(args) -> int:
    config = _load_config(args)
    document = load_license(Path(args.license))
    dest = config.state_dir / LICENSE_FILENAME
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"license installed to {dest}")
    return 0


def cmd_license_verify(args) -> int:
    document = load_license(Path(args.license))
    issuer_public = KeyPair.from_public_pem(Path(args.issuer_pub).read_bytes())
    claims = verify_license(document, issuer_public)
    print("license VALID")
    print(f"  holder:    {claims.holder}")
    print(f"  deployment: {claims.deployment_id}")
    print(f"  tier:      {claims.tier}")
    print(f"  modules:   {sorted(claims.modules)}")
    print(f"  expires:   {claims.expires_at.isoformat()}")
    print(f"  agreement: {claims.agreement_reference} (equity {claims.equity_percent}%)")
    return 0


def cmd_license_sign_module(args) -> int:
    issuer = KeyPair.from_private_pem(Path(args.issuer_key).read_bytes())
    module_dir = Path(args.module_dir)
    defaults = {}
    existing = module_dir / "manifest.json"
    if existing.exists():
        try:
            defaults = json.loads(existing.read_text(encoding="utf-8")).get("payload", {})
        except json.JSONDecodeError:
            defaults = {}
    payload = build_manifest(
        module_dir,
        name=args.name or defaults.get("name") or module_dir.name,
        module_type=args.module_type or defaults.get("type") or "chat",
        version=args.version or defaults.get("version") or "0.1.0",
        entrypoint=args.entrypoint or defaults.get("entrypoint") or "",
    )
    signed = sign_manifest(payload, issuer)
    path = write_signed_manifest(module_dir, signed)
    print(f"signed manifest written to {path}")
    print(f"  module: {payload['name']} v{payload['version']} ({len(payload['files'])} files hashed)")
    return 0


def cmd_license_audit_verify(args) -> int:
    config = _load_config(args)
    log = AuditLog(config.state_dir / "audit.jsonl")
    count = log.verify()
    print(f"audit chain VALID ({count} entries, head {log.head_hash()[:16]}…)")
    return 0


def cmd_license_audit_show(args) -> int:
    config = _load_config(args)
    entries = AuditLog(config.state_dir / "audit.jsonl").entries()
    for entry in entries[-args.limit:]:
        print(f"#{entry['seq']} {entry['ts']} {entry['event']} {json.dumps(entry['details'])}")
    if not entries:
        print("audit log is empty")
    return 0


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gravion", description="Gravion AI platform CLI")
    parser.add_argument("--home", help="deployment home (default: $GRAVION_HOME or ~/.gravion-ai)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init", help="create a deployment")
    p.add_argument("--name", default="Gravion")
    p.add_argument("--company", default="")
    p.add_argument("--mode", choices=["development", "production"], default="development")
    p.add_argument("--tier", choices=sorted(TIERS), default="tier-2-basic")
    p.add_argument("--provider", choices=["gravion-core", "endpoint"], default="gravion-core")
    p.add_argument("--model", default="gravion-core",
                   help="Gravion model id from the registry (e.g. gravion-code for the coder module)")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("status", help="show deployment and gate state")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("modules", help="module catalogue")
    p_sub = p.add_subparsers(dest="modules_command", required=True)
    p_list = p_sub.add_parser("list")
    p_list.set_defaults(func=cmd_modules_list)

    p = sub.add_parser("models", help="Gravion model registry")
    p_sub = p.add_subparsers(dest="models_command", required=True)
    p_list = p_sub.add_parser("list")
    p_list.add_argument("--capability", help="only models serving this capability")
    p_list.set_defaults(func=cmd_models_list)

    p = sub.add_parser("chat", help="run the chat module")
    p.add_argument("prompt", nargs="+")
    p.set_defaults(func=cmd_chat)

    p = sub.add_parser("support", help="run the support module")
    p.add_argument("prompt", nargs="+")
    p.set_defaults(func=cmd_support)

    p = sub.add_parser("coder", help="run the coder module")
    p.add_argument("prompt", nargs="+")
    p.set_defaults(func=cmd_coder)

    p = sub.add_parser("image", help="run the image module")
    p.add_argument("prompt", nargs="+")
    p.set_defaults(func=cmd_image)

    p = sub.add_parser("music", help="run the music module")
    p.add_argument("prompt", nargs="+")
    p.set_defaults(func=cmd_music)

    p = sub.add_parser("video", help="run the video module")
    p.add_argument("prompt", nargs="+")
    p.add_argument("--out", help="render a real MP4 locally to this path "
                                 "(local development renderer) instead of "
                                 "returning the model's reply")
    p.set_defaults(func=cmd_video)

    p = sub.add_parser("website", help="run the website-builder module")
    p.add_argument("prompt", nargs="+")
    p.add_argument("--out", help="build a real static website locally "
                                 "into this directory (local "
                                 "development builder) instead of "
                                 "returning the model's reply")
    p.set_defaults(func=cmd_website)

    p = sub.add_parser("cyber", help="run the cyber module")
    p.add_argument("prompt", nargs="+")
    p.set_defaults(func=cmd_cyber)

    p = sub.add_parser("agent", help="run the agentic module")
    p.add_argument("prompt", nargs="+")
    p.add_argument("--plan-out", help="write a structured agent plan "
                                      "locally to this JSON path "
                                      "(local development planner; "
                                      "nothing is executed) instead of "
                                      "returning the model's reply")
    p.set_defaults(func=cmd_agent)

    p = sub.add_parser("omni", help="run the all-in-one module "
                                    "(routes by '<module>:' directive, "
                                    "default chat)")
    p.add_argument("prompt", nargs="+")
    p.set_defaults(func=cmd_omni)

    p = sub.add_parser("authority", help="run the licensing authority service")
    authority_sub = p.add_subparsers(dest="action", required=True)
    ap = authority_sub.add_parser("serve", help="serve entitlement requests")
    ap.add_argument("--issuer-key", required=True,
                    help="path to the issuer private key (server-side only)")
    ap.add_argument("--licenses", required=True,
                    help="directory of issued license documents (*.json)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8790)
    p.set_defaults(func=cmd_authority_serve)

    p = sub.add_parser("upgrade", help="governed self-upgrade (§9)")
    upgrade_sub = p.add_subparsers(dest="action", required=True)
    upgrade_sub.add_parser("grant", help="grant the standing permission")
    upgrade_sub.add_parser("revoke", help="revoke the standing permission")
    upgrade_sub.add_parser("status", help="show permission + retained versions")
    up = upgrade_sub.add_parser("apply", help="apply a signed update package")
    up.add_argument("--package", required=True)
    up = upgrade_sub.add_parser("rollback", help="restore the latest retained version")
    up.add_argument("--module", required=True)
    p.set_defaults(func=cmd_upgrade)

    p = sub.add_parser("bundles", help="inspect and build tier bundles")
    bundle_sub = p.add_subparsers(dest="action", required=True)
    bundle_sub.add_parser("list", help="show each tier's completeness")
    bp = bundle_sub.add_parser("build", help="write a tier bundle descriptor")
    bp.add_argument("--tier", required=True)
    bp.add_argument("--out", required=True)
    p.set_defaults(func=cmd_bundles)

    p = sub.add_parser("train", help="validate or run a training job")
    train_sub = p.add_subparsers(dest="action", required=True)
    for action in ("validate", "run"):
        tp = train_sub.add_parser(action)
        tp.add_argument("--job", required=True, help="path to the job JSON")
    tp = train_sub.add_parser(
        "prepare", help="prepare a fine-tuning dataset (validate, "
                        "dedupe, split, report)"
    )
    tp.add_argument("--data", required=True,
                    help="a .jsonl file or a directory of .jsonl files")
    tp.add_argument("--out", required=True, help="output directory")
    tp.add_argument("--validation-fraction", type=float, default=0.2)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("export", help="export the full deployment bundle")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("license", help="licensing commands")
    p_sub = p.add_subparsers(dest="license_command", required=True)

    p_id = p_sub.add_parser("identity", help="create/show the deployment identity")
    p_id.set_defaults(func=cmd_license_identity)

    p_kg = p_sub.add_parser("issuer-keygen", help="(owner) create the issuer keypair")
    p_kg.add_argument("--out", required=True)
    p_kg.set_defaults(func=cmd_license_issuer_keygen)

    p_issue = p_sub.add_parser("issue", help="(owner) sign a license")
    p_issue.add_argument("--issuer-key", required=True)
    p_issue.add_argument("--deployment-id", required=True)
    p_issue.add_argument("--holder", required=True)
    p_issue.add_argument("--tier", choices=sorted(TIERS), required=True)
    p_issue.add_argument("--modules", required=True, help="comma-separated module names, or *")
    p_issue.add_argument("--agreement-ref", required=True)
    p_issue.add_argument("--equity", type=float, required=True, help="partnership equity % (10-20)")
    p_issue.add_argument("--days", type=int, default=365)
    p_issue.add_argument("--out")
    p_issue.set_defaults(func=cmd_license_issue)

    p_inst = p_sub.add_parser("install", help="install a license into this deployment")
    p_inst.add_argument("--license", required=True)
    p_inst.set_defaults(func=cmd_license_install)

    p_ver = p_sub.add_parser("verify", help="verify a license file")
    p_ver.add_argument("--license", required=True)
    p_ver.add_argument("--issuer-pub", required=True)
    p_ver.set_defaults(func=cmd_license_verify)

    p_sign = p_sub.add_parser("sign-module", help="(owner) sign a module manifest")
    p_sign.add_argument("--module-dir", required=True)
    p_sign.add_argument("--issuer-key", required=True)
    p_sign.add_argument("--name")
    p_sign.add_argument("--module-type", dest="module_type")
    p_sign.add_argument("--version")
    p_sign.add_argument("--entrypoint")
    p_sign.set_defaults(func=cmd_license_sign_module)

    p_audit = p_sub.add_parser("audit", help="audit log commands")
    p_audit_sub = p_audit.add_subparsers(dest="audit_command", required=True)
    p_av = p_audit_sub.add_parser("verify")
    p_av.set_defaults(func=cmd_license_audit_verify)
    p_as = p_audit_sub.add_parser("show")
    p_as.add_argument("--limit", type=int, default=20)
    p_as.set_defaults(func=cmd_license_audit_show)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except GravionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
