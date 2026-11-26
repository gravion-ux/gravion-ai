# Gravion AI production runbook

This runbook covers the local production shape of Gravion AI: what
must be true before a deployment serves real users, and what still
belongs to operations rather than code.

## Pre-flight

Run the readiness checks before every production start:

```bash
gravion doctor
gravion status
gravion license audit verify
```

`gravion doctor` is read-only. It checks config validity, company
policy, deployment identity, audit-chain integrity, the model
catalog, provider configuration, and — in production mode — the
issuer public key and installed license. Fix every FAIL before
serving traffic. WARN items need an operator decision recorded in
the deployment notes.

## Configuration rules enforced by the code

- Unknown config fields are rejected; typos fail loudly instead of
  being ignored.
- Endpoint providers need an absolute HTTP(S) base URL with no
  embedded credentials, query string or fragment.
- Remote endpoints must use HTTPS in production. Plain HTTP is
  accepted only for loopback endpoints.
- API keys must come from the environment variable named by
  `provider.api_key_env`. Production config refuses a stored
  `provider.api_key` in `gravion.json`.
- Timeouts are bounded to 1–300 seconds.

## Licensing operations

Production use requires the signed commercial agreement and a
license issued for the deployment. Keep the issuer private key
offline except on the licensing authority host. Deployments install
only `state/issuer.pub`, their license, and short-lived entitlement
tokens.

Suspension is safe by design: module runs stop, data is not deleted
or withheld, and `gravion export` remains available. Never add a
destructive kill switch. If entitlement lapses, restore service by
installing a valid license or letting the authority issue a fresh
token; no data migration is required.

## Web UI deployment

`gravion web serve` binds to `127.0.0.1:8791` by default. For real
users, place it behind a TLS reverse proxy with operator-managed
authentication and request-size limits. The built-in server already
caps request bodies, stores uploads under `data/uploads/`, sanitizes
filenames, and never executes uploaded content.

## Backups and export

Back up the whole deployment home, especially `state/` and `data/`.
`gravion export --out deployment.zip` produces the customer export
bundle in every gate state; the private identity key is never
exported. Test restores by loading the config, verifying the audit
chain, and running `gravion doctor` on the restored copy.

## Honest production blockers

- **Trained weights:** Gravion Core in this build includes a local
  sparse n-gram embedding + direct linear softmax classifier (10,586,543
  parameters counted from stored quantized weights) trained only
  from the bundled Gravion corpus (5,657 examples, 779 answer
  classes), with a deterministic offline fallback. Held-out eval:
  1,238/1,530 paraphrased prompts answered correctly (80.9%). Each
  specialist registry entry also includes local sparse n-gram
  classifier weights trained only from its bundled Gravion-owned
  seed corpus, with counts and hashes computed from the stored
  artifacts and a deterministic offline fallback. If
  a deployment connects an endpoint, the parameter count belongs
  to that connected model, not to Gravion.
- **Hosting:** the licensing authority service and web UI are code;
  public hosting, TLS certificates, DNS, monitoring and incident
  response are operations work.
- **Payments and contracts:** the commercial agreement template
  needs lawyer review, and payment/equity administration is outside
  this codebase.
- **Training execution:** dataset preparation, job validation, and
  Gravion Core and specialist local training are local; larger
  operations training remains bundle or endpoint work.
