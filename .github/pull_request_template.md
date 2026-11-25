## What changed

<!-- One or two sentences. What is different after this pull request? -->

## Why

<!-- Link the issue it closes, or state the problem it solves. Say
     which roadmap piece in ARCHITECTURE.md it belongs to, if any. -->

## How it was tested

<!-- The exact commands you ran and what they printed, e.g.
     `python -m pytest -q` (all passing) and the CLI smoke test:
     `gravion init --name Gravion`, `gravion status`,
     `gravion modules list`, `gravion chat "hello gravion"`
     in development mode with a throwaway GRAVION_HOME. -->

## Checklist

- [ ] Full test suite passes locally and CI is green (Python 3.10, 3.11, 3.12 + smoke test)
- [ ] Tests added or updated for the change (bug fixes include a test that failed before the fix)
- [ ] Licensing untouched — or, if `src/gravion_ai/licensing/`, the entitlement gate, signed manifests, or the audit chain changed, an issue/design discussion happened first and is linked above
- [ ] Development mode still runs free and fully offline; no new runtime dependency beyond `cryptography`
- [ ] No private keys, license files, deployment identities, or secrets anywhere in the diff
- [ ] No planned capability made to pretend it works — stubs still raise honestly
- [ ] Docs updated (README / docs / examples) if behaviour changed
- [ ] One logical change in this PR
