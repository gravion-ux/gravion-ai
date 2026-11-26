"""Governed self-upgrade (Piece 10) — ARCHITECTURE.md §9, implemented.

Gravion may improve itself — bug fixes and enhancements — under a
strict governor. This module is that governor; every rule in §9 is
enforced here, in code:

1. **One-time permission.** `config.self_upgrade_permission` is
   the standing grant, recorded in config and in the audit log
   (`upgrade.permission_granted` / `upgrade.permission_revoked`).
   No grant, no self-changes — `apply` refuses and audits the
   refusal. Revocable at any time.
2. **Signed updates only.** An update package is an issuer-signed
   document (`update.json`, the same canonical signing as module
   manifests) plus the new files, each pinned by SHA-256 in the
   signed payload. Unsigned, tampered or hash-mismatched packages
   are refused + audited.
3. **Scoped.** The payload's kind must be `bugfix` or
   `enhancement`, and the target must be an existing, implemented
   module, at the version the package says it updates from. The
   governor never touches licensing terms, tier, policy or
   identity — there is simply no code path here that can.
4. **Logged and reversible.** Before anything changes, the
   current module directory is snapshotted under
   `state/updates/backups/<module>/<version>/`. The apply is
   audited with before/after file hashes (`upgrade.applied`); a
   failed post-update self-check restores the snapshot
   automatically (`upgrade.rollback`), and `rollback()` restores
   the latest snapshot manually.
5. **Transparent.** The human-readable changelog is written into
   the deployment (`updates/CHANGELOG-<module>-<version>.md`)
   *before* the new files are activated, never after.

The self-check is: every updated file's hash matches the signed
payload, every Python file in the module still compiles, and —
when the module directory is the live imported package — the
entrypoint module reloads cleanly.
"""

from __future__ import annotations

import importlib
import json
import py_compile
import shutil
from pathlib import Path

from .canonical import canonical_json
from .config import GravionConfig
from .errors import UpgradeError
from .licensing.audit import AuditLog
from .licensing.keys import KeyPair, b64decode, b64encode
from .licensing.modules_sig import hash_file
from .licensing.timeutil import to_iso, utcnow
from .registry import ModuleRegistry, ModuleStatus, default_registry

PACKAGE_FILENAME = "update.json"
UPDATE_KINDS = ("bugfix", "enhancement")
_PAYLOAD_FIELDS = frozenset(
    {"module", "from_version", "to_version", "kind", "changelog", "files"}
)


# -- update packages (issuer-side construction + loading) --------------------


def write_update_package(
    package_dir,
    issuer: KeyPair,
    *,
    module: str,
    from_version: str,
    to_version: str,
    kind: str,
    changelog: str,
    source_dir,
) -> dict:
    """Build a signed update package from a directory of new files.

    Issuer-side tooling (the project owner signs update packages
    like module manifests; the private key never ships). Every
    file under `source_dir` is copied into the package's `files/`
    tree and pinned by SHA-256 in the signed payload.
    """
    if kind not in UPDATE_KINDS:
        raise UpgradeError(
            f"update kind must be one of {UPDATE_KINDS}, got {kind!r}"
        )
    package_dir = Path(package_dir)
    source_dir = Path(source_dir)
    files_root = package_dir / "files"
    if files_root.exists():
        shutil.rmtree(files_root)
    files: dict[str, str] = {}
    for path in sorted(source_dir.rglob("*")):
        if path.is_file():
            rel = path.relative_to(source_dir).as_posix()
            target = files_root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            files[rel] = hash_file(target)
    if not files:
        raise UpgradeError(f"no files to package under {source_dir}")
    payload = {
        "module": module,
        "from_version": from_version,
        "to_version": to_version,
        "kind": kind,
        "changelog": changelog,
        "files": files,
    }
    document = {
        "payload": payload,
        "signature": b64encode(issuer.sign(canonical_json(payload))),
    }
    package_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / PACKAGE_FILENAME).write_text(
        json.dumps(document, indent=2) + "\n", encoding="utf-8"
    )
    return document


def _validate_relpath(rel: str) -> None:
    path = Path(rel)
    if path.is_absolute() or ".." in path.parts or not rel.strip():
        raise UpgradeError(f"unsafe file path in update package: {rel!r}")


def load_update_package(package_dir) -> dict:
    """Load and structurally validate an update package document."""
    package_dir = Path(package_dir)
    path = package_dir / PACKAGE_FILENAME
    if not path.exists():
        raise UpgradeError(f"no update package at {package_dir}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        payload = document["payload"]
        signature = document["signature"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise UpgradeError(f"malformed update package: {exc}") from exc
    if not isinstance(payload, dict) or set(payload) != _PAYLOAD_FIELDS:
        raise UpgradeError(
            f"update payload must contain exactly {sorted(_PAYLOAD_FIELDS)}"
        )
    if payload["kind"] not in UPDATE_KINDS:
        raise UpgradeError(
            f"update kind must be one of {UPDATE_KINDS}, "
            f"got {payload['kind']!r} — self-upgrade covers bug fixes "
            "and enhancements only"
        )
    if not isinstance(payload["files"], dict) or not payload["files"]:
        raise UpgradeError("update payload names no files")
    for rel in payload["files"]:
        _validate_relpath(rel)
    if not isinstance(signature, str) or not signature:
        raise UpgradeError("update package is unsigned")
    return document


def verify_update_package(package_dir, issuer_public: KeyPair) -> dict:
    """Verify a package's signature and every file's hash.

    Returns the payload on success; raises UpgradeError on any
    mismatch — an unsigned, tampered or incomplete package never
    gets further than this function.
    """
    package_dir = Path(package_dir)
    document = load_update_package(package_dir)
    payload = document["payload"]
    try:
        signature = b64decode(document["signature"])
    except (TypeError, ValueError) as exc:
        raise UpgradeError(f"malformed update signature: {exc}") from exc
    try:
        issuer_public.verify(canonical_json(payload), signature)
    except Exception as exc:
        raise UpgradeError(
            f"update package signature does not verify: {exc}"
        ) from exc
    files_root = package_dir / "files"
    for rel, expected in payload["files"].items():
        target = files_root / rel
        if not target.exists():
            raise UpgradeError(f"update package is missing file {rel!r}")
        actual = hash_file(target)
        if actual != expected:
            raise UpgradeError(
                f"update file {rel!r} does not match its signed hash"
            )
    return payload


# -- the governor ---------------------------------------------------------------


def _dir_hashes(module_dir: Path) -> dict[str, str]:
    """SHA-256 of every file in a module directory (audit record)."""
    return {
        path.relative_to(module_dir).as_posix(): hash_file(path)
        for path in sorted(module_dir.rglob("*"))
        if path.is_file()
    }


class UpgradeService:
    """Applies signed update packages to a deployment's modules,
    under the §9 governor. One service per deployment config."""

    def __init__(
        self,
        config: GravionConfig,
        *,
        registry: ModuleRegistry | None = None,
        audit: AuditLog | None = None,
        issuer_public: KeyPair | None = None,
    ):
        self._config = config
        self._registry = registry or default_registry()
        self._audit = audit or AuditLog(config.state_dir / "audit.jsonl")
        if issuer_public is None:
            from .runtime import load_issuer_public

            issuer_public = load_issuer_public(config)
        self._issuer_public = issuer_public

    # -- permission ----------------------------------------------------

    def grant(self) -> None:
        self._config.self_upgrade_permission = True
        self._config.save()
        self._audit.append("upgrade.permission_granted", {})

    def revoke(self) -> None:
        self._config.self_upgrade_permission = False
        self._config.save()
        self._audit.append("upgrade.permission_revoked", {})

    def status(self) -> dict:
        backups: dict[str, list[str]] = {}
        root = self._backups_root
        if root.exists():
            for module_dir in sorted(root.iterdir()):
                if module_dir.is_dir():
                    backups[module_dir.name] = sorted(
                        p.name for p in module_dir.iterdir() if p.is_dir()
                    )
        return {
            "permission": self._config.self_upgrade_permission,
            "backups": backups,
        }

    # -- paths ---------------------------------------------------------

    @property
    def _backups_root(self) -> Path:
        return self._config.state_dir / "updates" / "backups"

    def _backup_dir(self, module: str, version: str) -> Path:
        return self._backups_root / module / version

    # -- apply / rollback ------------------------------------------------

    def _refuse(self, reason: str, **details) -> UpgradeError:
        self._audit.append("upgrade.refused", {"reason": reason, **details})
        return UpgradeError(reason)

    def apply(self, package_dir) -> dict:
        """Apply a signed update package under the governor.

        Refusals (no permission, bad signature, out of scope,
        version mismatch) are audited and raised before anything
        on disk changes. A failed self-check after activation
        rolls back automatically and is audited as such.
        """
        package_dir = Path(package_dir)
        if not self._config.self_upgrade_permission:
            raise self._refuse(
                "self-upgrade permission is not granted for this "
                "deployment (see `gravion upgrade grant`)"
            )
        if self._issuer_public is None:
            raise self._refuse(
                "no issuer public key installed at state/issuer.pub; "
                "cannot verify update packages"
            )
        try:
            payload = verify_update_package(package_dir, self._issuer_public)
        except UpgradeError as exc:
            raise self._refuse(str(exc)) from exc

        module_name = payload["module"]
        try:
            manifest = self._registry.get(module_name)
        except Exception:
            raise self._refuse(
                f"update targets unknown module {module_name!r}"
            ) from None
        if manifest.status is not ModuleStatus.IMPLEMENTED or manifest.path is None:
            raise self._refuse(
                f"update target {module_name!r} is not an implemented "
                "filesystem module"
            )
        if payload["from_version"] != manifest.version:
            raise self._refuse(
                f"update is for {module_name} {payload['from_version']}, "
                f"but the installed version is {manifest.version}"
            )

        module_dir = Path(manifest.path)
        before = _dir_hashes(module_dir)

        # Rule 5: the changelog lands in the deployment BEFORE
        # activation.
        changelog_path = (
            self._config.home
            / "updates"
            / f"CHANGELOG-{module_name}-{payload['to_version']}.md"
        )
        changelog_path.parent.mkdir(parents=True, exist_ok=True)
        changelog_path.write_text(
            f"# {module_name} {payload['from_version']} → "
            f"{payload['to_version']} ({payload['kind']})\n\n"
            f"Applied: {to_iso(utcnow())}\n\n{payload['changelog']}\n",
            encoding="utf-8",
        )

        # Rule 4: snapshot the current version before touching it.
        backup_dir = self._backup_dir(module_name, payload["from_version"])
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        shutil.copytree(module_dir, backup_dir)

        files_root = package_dir / "files"
        for rel in payload["files"]:
            target = module_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(files_root / rel, target)

        problem = self._self_check(module_dir, payload, manifest.entrypoint)
        if problem is not None:
            self._restore(backup_dir, module_dir)
            self._audit.append(
                "upgrade.rollback",
                {
                    "module": module_name,
                    "reason": f"self-check failed: {problem}",
                    "automatic": True,
                },
            )
            raise UpgradeError(
                f"update to {module_name} {payload['to_version']} failed "
                f"its self-check ({problem}); the previous version was "
                "restored automatically"
            )

        after = _dir_hashes(module_dir)
        result = {
            "module": module_name,
            "from_version": payload["from_version"],
            "to_version": payload["to_version"],
            "kind": payload["kind"],
            "changelog_path": str(changelog_path),
            "backup_path": str(backup_dir),
        }
        self._audit.append(
            "upgrade.applied",
            {**result, "before_hashes": before, "after_hashes": after},
        )
        return result

    def _self_check(self, module_dir: Path, payload: dict, entrypoint: str):
        """Post-activation verification. Returns a problem string,
        or None when the updated module is sound."""
        for rel, expected in payload["files"].items():
            target = module_dir / rel
            if not target.exists() or hash_file(target) != expected:
                return f"file {rel!r} does not match the signed payload"
        for path in sorted(module_dir.rglob("*.py")):
            try:
                py_compile.compile(str(path), doraise=True)
            except py_compile.PyCompileError as exc:
                return f"{path.name} no longer compiles: {exc.msg}"
        # When the module directory is the live imported package,
        # prove the entrypoint still loads. (In a staged/test
        # layout the hash + compile checks above are the check.)
        module_name = entrypoint.split(":")[0]
        try:
            package = importlib.import_module(module_name.rsplit(".", 1)[0])
        except Exception:
            return None
        if Path(package.__file__).parent.resolve() == module_dir.resolve():
            try:
                importlib.reload(importlib.import_module(module_name))
            except Exception as exc:
                return f"entrypoint reload failed: {exc}"
        return None

    def _restore(self, backup_dir: Path, module_dir: Path):
        if module_dir.exists():
            shutil.rmtree(module_dir)
        shutil.copytree(backup_dir, module_dir)

    def rollback(self, module_name: str) -> dict:
        """Restore the latest retained version of a module."""
        root = self._backups_root / module_name
        if not root.exists():
            raise UpgradeError(
                f"no retained versions for module {module_name!r}"
            )
        versions = sorted(p.name for p in root.iterdir() if p.is_dir())
        if not versions:
            raise UpgradeError(
                f"no retained versions for module {module_name!r}"
            )
        latest = versions[-1]
        manifest = self._registry.get(module_name)
        if manifest.path is None:
            raise UpgradeError(
                f"module {module_name!r} has no filesystem directory"
            )
        self._restore(root / latest, Path(manifest.path))
        result = {"module": module_name, "restored_version": latest}
        self._audit.append(
            "upgrade.rollback", {**result, "automatic": False}
        )
        return result
