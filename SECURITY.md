# Security Policy

Gravion AI's licensing and entitlement layer is its commercial core,
so the security of this repository is treated seriously. This policy
covers vulnerabilities **in this repository** — the framework, the
module runtime, and above all the licensing subsystem.

## Reporting a vulnerability

Please **do not open a public issue** for a vulnerability. Report it
privately, either way:

- GitHub's private vulnerability reporting:
  <https://github.com/gravion-ux/gravion-ai/security/advisories/new>
- Email: priyanshukumarthakur937@gmail.com

Please include:

- A one- or two-sentence description of the issue.
- The version or commit SHA it affects.
- The smallest reproduction you have — a failing test is ideal.
- The impact you believe it has.

**Never include real private keys, issuer keys, deployment
identities, or production license files in a report.** Reproduce with
test keys generated locally.

You can expect an acknowledgement within a few days.

## Disclosure timing

Please allow **90 days** for a fix to be released before any public
disclosure. Licensing issues may need coordinated timing with
deployments; that can be discussed — the goal is that no deployment
is left exposed.

## Scope

In scope — this repository itself, especially:

- Any bypass of the entitlement gate, signature verification for
  licenses or module manifests, or the hash-chained audit log (a way
  to alter history without detection).
- Development mode gaining network behaviour or account requirements
  it should not have — dev mode is free and fully offline by design.
- A module escaping its declared scope, or planned/stub capability
  being made to appear functional in a way that misleads an operator.

Out of scope:

- Vulnerabilities in third-party models, providers, or systems a
  Gravion deployment is connected to — those belong to their owners.
- Findings that require an attacker to already hold the issuer's
  private key. Protecting that key is the deployment owner's
  responsibility; the bundled manifests ship unsigned on purpose.

## Supported versions

Only the current `main` branch and the latest release are supported.
Older commits are kept for history, not for backports.
