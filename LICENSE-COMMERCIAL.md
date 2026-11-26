# Gravion AI — Commercial Agreement (Template)

> ⚠️ **TEMPLATE — PENDING LEGAL REVIEW.** This document records the
> commercial terms Gravion AI is designed around, in plain language,
> so the product and its licensing technology can be built against
> them. It is **not** a signed contract, **not** legal advice, and
> must be reviewed and adapted by a qualified lawyer before use with
> any customer.


The source code in this repository is MIT-licensed (see `LICENSE`). This commercial agreement additionally governs running Gravion AI as a production deployment; development and evaluation use needs no commercial agreement.

---

## 1. Parties and product

This agreement is between the **Licensor** (the Gravion AI project
owner) and the **Customer** named in the signature block, and covers
the Customer's use of the Gravion AI platform — its modules, models,
and licensing system — in the tier recorded in Schedule A.

## 2. Development and evaluation

The Customer may use Gravion AI in development mode, without this
agreement and without charge, for development and evaluation purposes
for an unlimited period.

## 3. Production use requires this agreement

The Customer must not run Gravion AI in **production** — serving real
users, customers, or business operations — before this agreement is
signed by both parties. On signing, the Licensor issues a signed
license document (see `docs/LICENSING.md`) bound to the Customer's
deployment identity, tier, and entitled modules.

## 4. Partnership equity

In consideration of the licence and the platform's ongoing
development, the Customer grants the Licensor an equity partnership in
the Customer's venture of **between 10% and 20%**, the exact figure
negotiated per customer and recorded in Schedule A. The issued license
document mirrors the agreed percentage and this agreement's reference
number.

## 5. No destructive kill switch

The Licensor commits that Gravion AI contains **no mechanism** to
delete, encrypt, corrupt, or withhold the Customer's data, policy,
configuration, or deployment, under any circumstances — including
expiry, suspension, dispute, or termination of this agreement.

## 6. Suspension

If entitlement lapses (expiry, non-payment of agreed terms, or
termination), the platform may **suspend**: module execution stops,
and nothing else happens. Suspension is not deletion. During
suspension the Customer keeps full read and export access (§7), and
restoring entitlement restores service with no data migration.

## 7. Data export

The Customer may at any time — during the term, during suspension, and
after termination — export the complete deployment: configuration,
company policy, model registry, module list, public identity material,
license, audit log, and all data created by or stored in the
deployment, using the export tooling shipped with the product
(`gravion export`). Export is a contractual right and is not gated by
entitlement.

## 8. Audit

Both parties may independently verify the deployment's tamper-evident
licensing audit log. Neither party can alter past entries without
detection.

## 9. Self-upgrade permission

If Schedule A records the Customer's one-time permission, Gravion AI
may apply signed bug-fix and enhancement updates as described in
ARCHITECTURE.md §9 — logged, reversible, and never extending to
licensing terms, tier, policy, or identity. The Customer may revoke
the permission at any time, effective immediately.

## 10. General

- This template, the signed Schedule A (tier, modules, equity %,
  agreement reference, term), and `docs/LICENSING.md` together describe
  the licensing relationship; in case of conflict, the signed
  agreement prevails over the technical documents.
- Nothing in the technical enforcement layer limits the Customer's
  rights under §§5–7.

**Schedule A** (completed at signing): Customer legal name ·
deployment ID · tier · entitled modules · equity % · agreement
reference · term and expiry · self-upgrade permission (yes/no) ·
signatures.
