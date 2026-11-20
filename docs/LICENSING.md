# Gravion AI Licensing

Gravion AI licensing is built as **two separate layers**. They are
designed independently, by different disciplines, and they meet at
exactly one point: a reference number.

| Layer | Owned by | Lives in |
|---|---|---|
| **Legal entitlement** | Lawyers / the signed commercial agreement | `LICENSE-COMMERCIAL.md` (template) and each customer's signed agreement |
| **Technical enforcement** | Engineers / this codebase | `src/gravion_ai/licensing/` |

The technical layer never interprets the contract. It checks
cryptographic facts: *is this license genuine, current, issued for
this deployment, and does it cover this module?* The legal layer never
depends on the software behaving — the customer's rights (including
data export) are contractual, and the software is built to honour them
even when it is the thing being argued about.

> This document and `LICENSE-COMMERCIAL.md` are **templates and design
> records, pending review by a qualified lawyer**. They are not legal
> advice and not, by themselves, a contract.

---

## Part I — Legal entitlement (summary of the agreement terms)

1. **Development and evaluation are free.** Anyone may run Gravion AI
   in development mode, with no agreement and no license, for as long
   as they are developing or evaluating.
2. **Production use requires a commercial agreement.** Before a
   deployment serves real users or real business, the operator signs
   the Gravion AI Commercial Agreement. No agreement, no production —
   this is a contractual obligation first, and a technical gate second.
3. **Partnership equity.** The agreement includes a negotiated equity
   partnership in the customer's venture of **10–20%**, fixed per
   customer in the signed agreement. Every issued license mirrors the
   agreed percentage so the records cannot drift apart.
4. **No destructive kill switch — contractual commitment.** The
   licensor commits that Gravion AI contains no mechanism to delete,
   encrypt, corrupt or withhold the customer's data or deployment, and
   that loss of entitlement leads at most to *suspension* (§III.5).
5. **Data export is a right, not a feature.** On suspension, expiry or
   termination, the customer may export everything — configuration,
   policy, registry, audit log and all deployment data — at any time,
   using tooling that ships with the product and works while suspended.
6. **Audit transparency.** The tamper-evident licensing audit log
   (§III.4) is the shared record both parties can verify independently.

## Part II — What the technical layer enforces

| Requirement | Mechanism | Code |
|---|---|---|
| Cryptographic deployment identity | Ed25519 keypair per deployment; public fingerprint = deployment ID; private key local-only, mode 0600, never transmitted | `licensing/identity.py` |
| Signed license verification | Issuer-signed JSON license: deployment, holder, tier, modules, validity, agreement ref + equity %; verified before production runs | `licensing/license_doc.py` |
| Signed module verification | Issuer-signed manifest of per-file SHA-256 hashes; verified before a module loads in production; tampering refused + audited | `licensing/modules_sig.py` |
| Tamper-evident licensing audit | Append-only JSONL, each entry hash-chained to the previous; `verify()` localises any edit/reorder/deletion | `licensing/audit.py` |
| Server-side commercial entitlement | Signed, short-TTL entitlement tokens from the licensing authority (§III); local checks alone are never the root of trust | `licensing/authority.py`, `licensing/gate.py` |
| Transparent, privacy-respecting verification | Request = exactly 4 metadata fields; no content, no user data, no telemetry; field set pinned in code and tests | `licensing/authority.py` |
| Safe suspension + data export | Gate states development / active / suspended; suspension blocks runs only; export works in every state; private key never exported | `licensing/gate.py`, `runtime.py` |

## Part III — The entitlement protocol (specification)

### III.1 Request

`POST /v1/entitlement` with a JSON body containing **exactly**:

```json
{
  "deployment_id": "gravion-dep-<40 hex>",
  "license_hash": "<sha256 of the license payload, hex>",
  "module": "chat",
  "requested_at": "2026-11-20T10:00:00Z"
}
```

The authority **rejects** requests with missing or additional fields.
`license_hash` is a hash, not the license: the authority looks the
license up in its own registry, so the document itself never travels.

### III.2 Server-side verification

For each request the authority re-verifies, from its own records:

1. the license exists (by hash) and its issuer signature is valid;
2. it is inside its validity window;
3. it is issued for the requesting deployment;
4. it entitles the requested module.

### III.3 Response — the entitlement token

A signed document (Ed25519, authority key):

```json
{
  "payload": {
    "deployment_id": "…", "license_hash": "…",
    "holder": "…", "tier": "tier-3-intermediate",
    "modules": ["chat", "support"],
    "issued_at": "…", "expires_at": "…",
    "authority": "<authority key fingerprint>"
  },
  "signature": "<base64>"
}
```

Default TTL is **900 seconds**. The deployment verifies the token's
signature and expiry locally and caches it; while a cached token is
valid it authorises runs **even if the authority is unreachable**.

### III.4 Audit events

`deployment.init`, `gate.allow`, `gate.suspend`, `module.verified`,
`module.tamper`, `export.bundle` — each with metadata only (module,
reason, holder), chained as described in Part II.

### III.5 Suspension semantics

A production check fails → the deployment is **suspended for that
module run**:

- module code does not execute;
- the reason is returned to the operator and written to the audit log;
- **no data is touched** — nothing deleted, nothing encrypted, nothing
  withheld;
- export, status, identity and audit commands keep working;
- installing a valid license (or the authority issuing a token) returns
  the deployment to *active* with no data migration of any kind.

### III.6 Reference implementations in Piece 1

- `MockAuthority` — the authority's verification logic in-process,
  used by the test-suite and offline development. It performs the same
  checks as §III.2; only the transport is absent.
- `MockAuthorityClient` / `HttpAuthorityClient` — the two client
  transports (in-process, and HTTP per §III.1).
- The **hosted authority service** is deliberately *not* in Piece 1:
  the protocol above is the stable contract; operating the service is
  a later piece (see ARCHITECTURE.md roadmap).

## Part IV — Key management notes

- **Issuer key** (licensor): signs licenses, module manifests and —
  via the authority — entitlement tokens. Kept offline except on the
  authority. Generated with `gravion license issuer-keygen`.
- **Deployment key** (customer): generated on the deployment machine
  by `gravion license identity`; the private half never leaves it.
  Deployments install only the issuer's *public* key
  (`state/issuer.pub`) to verify licenses, manifests and tokens.
- Losing a deployment private key means a new identity and a re-issued
  license (an agreement-level event, recorded in the audit log of the
  new deployment). Losing the issuer key is a licensor incident:
  licenses signed by it remain verifiable against the published
  fingerprint until rotation.
