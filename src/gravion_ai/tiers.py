"""The four Gravion AI tiers.

Tiers are *bundle / training levels over the same module system* — the
modules do not change between tiers; what changes is how trained and
ready each module is when it ships, and what the commercial agreement
covers. Every tier is renameable (default name "Gravion"), takes a
company policy, and runs on the built-in Gravion Core engine or any
inference endpoint the deployment configures with its own API key.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Tier:
    id: str
    label: str
    training_level: str
    summary: str


TIERS: dict[str, Tier] = {
    "tier-1-complete": Tier(
        id="tier-1-complete",
        label="Tier 1 — Complete",
        training_level="fully trained, ready to use",
        summary=(
            "A complete, ready AI. Give it a name, teach it your company "
            "and work policy, and use it. No training work required."
        ),
    ),
    "tier-2-basic": Tier(
        id="tier-2-basic",
        label="Tier 2 — Basic",
        training_level="basic training included",
        summary=(
            "Basic-trained variants of every module type — a working "
            "starting point you continue training on your own data, "
            "with any API key or your own model."
        ),
    ),
    "tier-3-intermediate": Tier(
        id="tier-3-intermediate",
        label="Tier 3 — Intermediate",
        training_level="intermediate training included",
        summary=(
            "Already trained to a level that takes a startup years to "
            "reach in-house. Name it, inject your policy, use it — and "
            "keep improving it from a high baseline."
        ),
    ),
    "tier-4-advanced": Tier(
        id="tier-4-advanced",
        label="Tier 4 — Advanced",
        training_level="advanced training included",
        summary=(
            "The top tier: advanced-trained modules on the full Gravion "
            "platform — the strongest foundation Gravion offers for "
            "serious AI products, with headroom to train further to "
            "best-in-class."
        ),
    ),
}

DEFAULT_TIER = "tier-2-basic"


def get_tier(tier_id: str) -> Tier:
    try:
        return TIERS[tier_id]
    except KeyError:
        raise ValueError(
            f"unknown tier {tier_id!r}; expected one of {sorted(TIERS)}"
        ) from None
