"""Gravion Core's local trained model.

Gravion Core has two layers:

- a trained local model, learned only from local Gravion dataset
  files; and
- a deterministic offline fallback used when no trained weights are
  available or when the trained model has no confident match.

Format v3 (current) is a compact sparse embedding/linear model:
explicit word n-gram features from the training prompts are scored by
a learned direct feature-to-class matrix plus a low-dimensional
learned embedding classifier, trained with sampled-softmax SGD.
Weights are stored int8-quantized (base64) with per-matrix scales;
every stored quantized weight is a real trained parameter and the
parameter count is computed from the stored weights, never asserted
by hand. Bigram transition weights learned from the completions are
kept for ``continue_text``.

Format v2 (legacy, load-only) is the earlier hashed embedding model,
and format v1 (legacy, load-only) is the earlier TF-IDF retrieval +
n-gram model. Both stay loadable so older artifacts keep working.

No third-party weights are used anywhere in this module.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import random
import re
from array import array
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .errors import TrainingError

CORE_MODEL_ID = "gravion-core"
SPECIALIST_MODEL_IDS = (
    "gravion-chat",
    "gravion-support",
    "gravion-review",
    "gravion-code",
    "gravion-builder",
    "gravion-vision",
    "gravion-video",
    "gravion-audio",
    "gravion-shield",
    "gravion-agent",
    "gravion-omni",
)
LOCAL_MODEL_IDS = (CORE_MODEL_ID, *SPECIALIST_MODEL_IDS)
CORE_MODEL_FORMAT = "gravion-core-local-v3"
LEGACY_V2_MODEL_FORMAT = "gravion-core-local-v2"
LEGACY_MODEL_FORMAT = "gravion-core-local-v1"
CORE_ARCHITECTURE = "sparse n-gram embedding + direct linear softmax classifier"
DEFAULT_MATCH_THRESHOLD = 0.30

DEFAULT_DIM = 16
DEFAULT_EPOCHS = 128
DEFAULT_NEGATIVES = 32
DEFAULT_MAX_FEATURES = None
_LEARNING_RATE = 0.8
_L2 = 1e-6
_INIT_SEED = 20261126

_WEIGHTS_CACHE: dict[str, tuple[list[float], list[float], list[float], int, int]] = {}

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_START = "<s>"
_END = "</s>"


def tokenize(text: str) -> list[str]:
    """Tokenize text for the local Gravion Core model."""

    if not isinstance(text, str):
        raise TypeError(f"text must be a string, got {type(text).__name__}")
    return _TOKEN_RE.findall(text.lower())


def _ngrams(tokens: Sequence[str]) -> list[str]:
    terms = list(tokens)
    terms.extend(f"{tokens[i]} {tokens[i + 1]}" for i in range(len(tokens) - 1))
    return terms


def _normalized_text(text: str) -> str:
    return " ".join(tokenize(text))


def _bucket(term: str, buckets: int) -> int:
    digest = hashlib.md5(term.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") % buckets


def _features(prompt: str, buckets: int) -> list[int]:
    return sorted({_bucket(term, buckets) for term in _ngrams(tokenize(prompt))})


def _feature_terms(prompt: str) -> list[str]:
    tokens = tokenize(prompt)
    terms = set(tokens)
    terms.update(f"{tokens[i]} {tokens[i + 1]}" for i in range(len(tokens) - 1))
    terms.update(
        f"{tokens[i]} {tokens[i + 1]} {tokens[i + 2]}"
        for i in range(len(tokens) - 2)
    )
    return sorted(terms)


def _dataset_sha256(examples: Sequence[tuple[str, str]]) -> str:
    payload = "\n".join(f"{prompt}\n{completion}" for prompt, completion in examples)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


# ---------------------------------------------------------------------------
# Quantization helpers
# ---------------------------------------------------------------------------


def _quantize(values: Sequence[float]) -> tuple[str, float]:
    max_abs = max((abs(v) for v in values), default=0.0)
    scale = max_abs / 127.0 if max_abs > 0 else 1.0
    raw = bytes(
        max(-128, min(127, round(v / scale))) & 0xFF for v in values
    )
    return base64.b64encode(raw).decode("ascii"), scale


def _dequantize(data: str, scale: float, expected: int, source: str) -> list[float]:
    try:
        raw = base64.b64decode(data.encode("ascii"), validate=True)
    except (ValueError, UnicodeEncodeError) as exc:
        raise TrainingError(f"{source}: quantized weights are not valid base64") from exc
    if len(raw) != expected:
        raise TrainingError(
            f"{source}: quantized weights have {len(raw)} values, expected {expected}"
        )
    return [((b - 256) if b > 127 else b) * scale for b in raw]


def _dequantize_array(data: str, scale: float, expected: int, source: str) -> array:
    try:
        raw = base64.b64decode(data.encode("ascii"), validate=True)
    except (ValueError, UnicodeEncodeError) as exc:
        raise TrainingError(f"{source}: quantized weights are not valid base64") from exc
    if len(raw) != expected:
        raise TrainingError(
            f"{source}: quantized weights have {len(raw)} values, expected {expected}"
        )
    return array("f", (((b - 256) if b > 127 else b) * scale for b in raw))


def _validate_quantized_block(
    artifact: Mapping[str, object], name: str, expected: int, source: str
) -> None:
    block = artifact.get(name)
    if not isinstance(block, dict):
        raise TrainingError(f"{source}: {name} weights are missing")
    data = block.get("data")
    scale = block.get("scale")
    if not isinstance(data, str) or not _is_number(scale) or float(scale) <= 0:
        raise TrainingError(f"{source}: {name} weights are invalid")
    try:
        raw = base64.b64decode(data.encode("ascii"), validate=True)
    except (ValueError, UnicodeEncodeError) as exc:
        raise TrainingError(f"{source}: quantized weights are not valid base64") from exc
    if len(raw) != expected:
        raise TrainingError(
            f"{source}: quantized weights have {len(raw)} values, expected {expected}"
        )


# ---------------------------------------------------------------------------
# Parameter counting and hashing
# ---------------------------------------------------------------------------


def _transition_parameter_count(artifact: Mapping[str, object]) -> int:
    total = 0
    transitions = artifact.get("transitions", {})
    if isinstance(transitions, Mapping):
        for next_weights in transitions.values():
            if isinstance(next_weights, Mapping):
                total += sum(1 for value in next_weights.values() if _is_number(value))
    unigram_weights = artifact.get("unigram_weights", {})
    if isinstance(unigram_weights, Mapping):
        total += sum(1 for value in unigram_weights.values() if _is_number(value))
    return total


def count_parameters(artifact: Mapping[str, object]) -> int:
    """Count numeric weights stored in a Gravion Core artifact.

    For v3 artifacts the counted parameters are the stored quantized
    direct feature-to-class weights, embedding weights, classifier
    weights, classifier biases, and the bigram/unigram transition
    weights. For v2 artifacts the counted parameters are the stored
    quantized embedding weights, classifier weights, classifier
    biases, and transition weights. Class strings, prompts, hashes,
    and metadata are not counted.
    """

    if artifact.get("format") == CORE_MODEL_FORMAT:
        features = artifact.get("features")
        dim = artifact.get("dim")
        classes = artifact.get("classes")
        if (
            not isinstance(features, list)
            or not isinstance(dim, int)
            or not isinstance(classes, list)
        ):
            return 0
        total = len(features) * len(classes) + len(features) * dim + len(classes) * dim
        bias = artifact.get("bias", [])
        if isinstance(bias, list):
            total += sum(1 for value in bias if _is_number(value))
        return total + _transition_parameter_count(artifact)

    if artifact.get("format") == LEGACY_V2_MODEL_FORMAT:
        buckets = artifact.get("buckets")
        dim = artifact.get("dim")
        classes = artifact.get("classes")
        if not isinstance(buckets, int) or not isinstance(dim, int) or not isinstance(classes, list):
            return 0
        total = buckets * dim + len(classes) * dim
        bias = artifact.get("bias", [])
        if isinstance(bias, list):
            total += sum(1 for value in bias if _is_number(value))
        return total + _transition_parameter_count(artifact)

    # Legacy v1 counting.
    total = 0
    idf = artifact.get("idf", {})
    if isinstance(idf, Mapping):
        total += sum(1 for value in idf.values() if _is_number(value))
    examples = artifact.get("examples", [])
    if isinstance(examples, list):
        for example in examples:
            if isinstance(example, Mapping):
                vector = example.get("prompt_vector", {})
                if isinstance(vector, Mapping):
                    total += sum(1 for value in vector.values() if _is_number(value))
    return total + _transition_parameter_count(artifact)


def _weights_payload(artifact: Mapping[str, object]) -> dict[str, object]:
    if artifact.get("format") == CORE_MODEL_FORMAT:
        return {
            "features": artifact.get("features", []),
            "dim": artifact.get("dim"),
            "classes": artifact.get("classes", []),
            "direct": artifact.get("direct", {}),
            "embedding": artifact.get("embedding", {}),
            "classifier": artifact.get("classifier", {}),
            "bias": artifact.get("bias", []),
            "transitions": artifact.get("transitions", {}),
            "unigram_weights": artifact.get("unigram_weights", {}),
        }
    if artifact.get("format") == LEGACY_V2_MODEL_FORMAT:
        return {
            "buckets": artifact.get("buckets"),
            "dim": artifact.get("dim"),
            "classes": artifact.get("classes", []),
            "embedding": artifact.get("embedding", {}),
            "classifier": artifact.get("classifier", {}),
            "bias": artifact.get("bias", []),
            "transitions": artifact.get("transitions", {}),
            "unigram_weights": artifact.get("unigram_weights", {}),
        }
    return {
        "vocabulary": artifact.get("vocabulary", []),
        "idf": artifact.get("idf", {}),
        "examples": artifact.get("examples", []),
        "transitions": artifact.get("transitions", {}),
        "unigram_weights": artifact.get("unigram_weights", {}),
    }


def _weights_sha256(artifact: Mapping[str, object]) -> str:
    payload = json.dumps(_weights_payload(artifact), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Artifact validation
# ---------------------------------------------------------------------------


def _validate_transitions(artifact: Mapping[str, object], source: str) -> None:
    transitions = artifact.get("transitions")
    if not isinstance(transitions, dict) or not transitions:
        raise TrainingError(f"{source}: transition weights are missing")
    for previous, next_weights in transitions.items():
        if not isinstance(previous, str) or not isinstance(next_weights, dict) or not next_weights:
            raise TrainingError(f"{source}: transition weights are invalid")
        if not all(isinstance(k, str) and _is_number(v) for k, v in next_weights.items()):
            raise TrainingError(f"{source}: transition weights are invalid")
    unigram_weights = artifact.get("unigram_weights")
    if not isinstance(unigram_weights, dict) or not unigram_weights:
        raise TrainingError(f"{source}: unigram weights are missing")
    if not all(isinstance(k, str) and _is_number(v) for k, v in unigram_weights.items()):
        raise TrainingError(f"{source}: unigram weights are invalid")


def _validate_exact_prompts(
    artifact: dict[str, object], classes: list, source: str
) -> None:
    # Exact-prompt keys are looked up by normalized prompt text, so a
    # key must be a non-empty normalized prompt: an empty key would
    # make every empty or punctuation-only prompt a confident match,
    # and an unnormalized key could never be reached.
    exact = artifact.get("exact_prompts", {})
    if not isinstance(exact, dict) or not all(
        isinstance(k, str)
        and k
        and k == _normalized_text(k)
        and isinstance(v, int)
        and 0 <= v < len(classes)
        for k, v in exact.items()
    ):
        raise TrainingError(f"{source}: exact prompt map is invalid")


def _validate_common(artifact: dict[str, object], source: str) -> None:
    expected_count = count_parameters(artifact)
    if artifact.get("parameter_count") != expected_count:
        raise TrainingError(
            f"{source}: parameter_count does not match stored weights "
            f"({artifact.get('parameter_count')!r} != {expected_count})"
        )
    if artifact.get("weights_sha256") != _weights_sha256(artifact):
        raise TrainingError(f"{source}: weights_sha256 does not match stored weights")


def _validate_v3(artifact: dict[str, object], source: str) -> dict[str, object]:
    features = artifact.get("features")
    if (
        not isinstance(features, list)
        or not features
        or not all(isinstance(feature, str) and feature for feature in features)
        or len(set(features)) != len(features)
    ):
        raise TrainingError(f"{source}: feature vocabulary is missing or invalid")
    dim = artifact.get("dim")
    if not isinstance(dim, int) or dim <= 0:
        raise TrainingError(f"{source}: dim is missing or invalid")
    classes = artifact.get("classes")
    if not isinstance(classes, list) or not classes or not all(
        isinstance(c, str) and c.strip() for c in classes
    ):
        raise TrainingError(f"{source}: answer classes are missing or invalid")
    _validate_quantized_block(artifact, "direct", len(features) * len(classes), source)
    _validate_quantized_block(artifact, "embedding", len(features) * dim, source)
    _validate_quantized_block(artifact, "classifier", len(classes) * dim, source)
    bias = artifact.get("bias")
    if not isinstance(bias, list) or len(bias) != len(classes) or not all(
        _is_number(v) for v in bias
    ):
        raise TrainingError(f"{source}: classifier bias is invalid")
    _validate_exact_prompts(artifact, classes, source)
    vocabulary = artifact.get("vocabulary")
    if not isinstance(vocabulary, list) or not vocabulary or not all(
        isinstance(v, str) for v in vocabulary
    ):
        raise TrainingError(f"{source}: vocabulary is missing or invalid")
    _validate_transitions(artifact, source)
    _validate_common(artifact, source)
    return artifact


def _validate_v2(artifact: dict[str, object], source: str) -> dict[str, object]:
    buckets = artifact.get("buckets")
    dim = artifact.get("dim")
    if not isinstance(buckets, int) or buckets <= 0 or not isinstance(dim, int) or dim <= 0:
        raise TrainingError(f"{source}: buckets/dim are missing or invalid")
    classes = artifact.get("classes")
    if not isinstance(classes, list) or not classes or not all(
        isinstance(c, str) and c.strip() for c in classes
    ):
        raise TrainingError(f"{source}: answer classes are missing or invalid")
    for name, expected in (("embedding", buckets * dim), ("classifier", len(classes) * dim)):
        block = artifact.get(name)
        if not isinstance(block, dict):
            raise TrainingError(f"{source}: {name} weights are missing")
        data = block.get("data")
        scale = block.get("scale")
        if not isinstance(data, str) or not _is_number(scale) or float(scale) <= 0:
            raise TrainingError(f"{source}: {name} weights are invalid")
        _dequantize(data, float(scale), expected, source)
    bias = artifact.get("bias")
    if not isinstance(bias, list) or len(bias) != len(classes) or not all(
        _is_number(v) for v in bias
    ):
        raise TrainingError(f"{source}: classifier bias is invalid")
    _validate_exact_prompts(artifact, classes, source)
    vocabulary = artifact.get("vocabulary")
    if not isinstance(vocabulary, list) or not vocabulary or not all(
        isinstance(v, str) for v in vocabulary
    ):
        raise TrainingError(f"{source}: vocabulary is missing or invalid")
    _validate_transitions(artifact, source)
    _validate_common(artifact, source)
    return artifact


def _validate_v1(artifact: dict[str, object], source: str) -> dict[str, object]:
    vocabulary = artifact.get("vocabulary")
    if not isinstance(vocabulary, list) or not all(isinstance(v, str) for v in vocabulary):
        raise TrainingError(f"{source}: vocabulary must be a list of strings")
    idf = artifact.get("idf")
    if not isinstance(idf, dict) or not idf or not all(_is_number(v) for v in idf.values()):
        raise TrainingError(f"{source}: idf weights are missing or invalid")
    examples = artifact.get("examples")
    if not isinstance(examples, list) or not examples:
        raise TrainingError(f"{source}: trained examples are missing")
    for index, example in enumerate(examples):
        if not isinstance(example, dict):
            raise TrainingError(f"{source}: example {index} must be an object")
        if (
            not isinstance(example.get("prompt"), str)
            or not tokenize(example["prompt"])
        ):
            raise TrainingError(f"{source}: example {index} has no prompt")
        if not isinstance(example.get("completion"), str) or not example["completion"].strip():
            raise TrainingError(f"{source}: example {index} has no completion")
        vector = example.get("prompt_vector")
        if not isinstance(vector, dict) or not vector or not all(_is_number(v) for v in vector.values()):
            raise TrainingError(f"{source}: example {index} prompt vector is invalid")
    _validate_transitions(artifact, source)
    _validate_common(artifact, source)
    return artifact


def _validate_artifact(artifact: object, source: str) -> dict[str, object]:
    if not isinstance(artifact, dict):
        raise TrainingError(f"{source}: Gravion Core weights must be a JSON object")
    if artifact.get("model_id") not in LOCAL_MODEL_IDS:
        raise TrainingError(
            f"{source}: weights are not for a bundled local Gravion model"
        )
    if artifact.get("format") == CORE_MODEL_FORMAT:
        return _validate_v3(artifact, source)
    if artifact.get("format") == LEGACY_V2_MODEL_FORMAT:
        return _validate_v2(artifact, source)
    if artifact.get("format") == LEGACY_MODEL_FORMAT:
        return _validate_v1(artifact, source)
    raise TrainingError(f"{source}: unsupported Gravion local weights format")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GravionCoreLocalModel:
    """A loaded Gravion Core trained model (v3 current, v1/v2 legacy)."""

    artifact: dict[str, object]
    source: str = "<memory>"
    _weights_cache: object | None = field(default=None, repr=False, compare=False)

    @property
    def model_id(self) -> str:
        return str(self.artifact["model_id"])

    @property
    def parameter_count(self) -> int:
        return int(self.artifact["parameter_count"])

    @property
    def weights_sha256(self) -> str:
        return str(self.artifact["weights_sha256"])

    @property
    def is_v3(self) -> bool:
        return self.artifact.get("format") == CORE_MODEL_FORMAT

    @property
    def is_v2(self) -> bool:
        return self.artifact.get("format") == LEGACY_V2_MODEL_FORMAT

    # -- v3 inference ----------------------------------------------------

    def _v3_weights(
        self,
    ) -> tuple[array, array, array, list[float], dict[str, int], int, int]:
        if self._weights_cache is not None:
            return self._weights_cache  # type: ignore[return-value]
        artifact = self.artifact
        features = artifact["features"]
        classes = artifact["classes"]
        assert isinstance(features, list) and isinstance(classes, list)
        dim = int(artifact["dim"])  # type: ignore[arg-type]
        direct_block = artifact["direct"]
        embedding_block = artifact["embedding"]
        classifier_block = artifact["classifier"]
        assert isinstance(direct_block, dict)
        assert isinstance(embedding_block, dict)
        assert isinstance(classifier_block, dict)
        direct = _dequantize_array(
            str(direct_block["data"]),
            float(direct_block["scale"]),
            len(features) * len(classes),
            self.source,
        )
        embedding = _dequantize_array(
            str(embedding_block["data"]),
            float(embedding_block["scale"]),
            len(features) * dim,
            self.source,
        )
        classifier = _dequantize_array(
            str(classifier_block["data"]),
            float(classifier_block["scale"]),
            len(classes) * dim,
            self.source,
        )
        bias = [float(v) for v in artifact["bias"]]  # type: ignore[union-attr]
        feature_index = {str(feature): index for index, feature in enumerate(features)}
        weights = (direct, embedding, classifier, bias, feature_index, dim, len(classes))
        object.__setattr__(self, "_weights_cache", weights)
        return weights

    def _v3_score(self, prompt: str) -> tuple[str | None, float]:
        classes = self.artifact["classes"]
        assert isinstance(classes, list)
        exact = self.artifact.get("exact_prompts", {})
        assert isinstance(exact, dict)
        normalized_prompt = _normalized_text(prompt)
        # An empty normalized prompt (empty or punctuation-only text)
        # is never a confident match, even if an in-memory artifact
        # was built without validation and carries an empty key.
        hit = exact.get(normalized_prompt) if normalized_prompt else None
        if isinstance(hit, int):
            return str(classes[hit]), 1.0

        tokens = tokenize(prompt)
        vocabulary = self.artifact.get("vocabulary", [])
        assert isinstance(vocabulary, list)
        normalized = _normalized_text(prompt)
        if self.model_id == CORE_MODEL_ID:
            doc_question = normalized.startswith(
                (
                    "what is",
                    "explain",
                    "gravion",
                    "what does gravion say about",
                    "how does",
                    "tell me about",
                    "what should i know about",
                )
            ) or "gravion" in tokens
            if len(tokens) < 3 or not set(tokens).intersection(vocabulary) or not doc_question:
                return None, 0.0
        elif len(tokens) < 2 or not set(tokens).intersection(vocabulary):
            return None, 0.0

        direct, embedding, classifier, bias, feature_index, dim, num_classes = (
            self._v3_weights()
        )
        terms = _feature_terms(prompt)
        feature_ids = [
            feature_index[term] for term in terms if term in feature_index
        ]
        if not feature_ids:
            return None, 0.0
        # Evidence coverage: the share of the prompt's n-gram terms the
        # model actually knows. Margin alone overstates confidence when
        # only a couple of terms fire (e.g. "a website for our bakery"
        # matched a resume answer at 0.79 on 3 of 12 terms), so the
        # returned confidence is scaled by coverage below.
        coverage = len(feature_ids) / len(terms)

        hidden = [0.0] * dim
        for feature_id in feature_ids:
            base = feature_id * dim
            for j in range(dim):
                hidden[j] += embedding[base + j]
        inv = 1.0 / len(feature_ids)
        hidden = [value * inv for value in hidden]

        direct_logits = [0.0] * num_classes
        for feature_id in feature_ids:
            base = feature_id * num_classes
            for c in range(num_classes):
                direct_logits[c] += direct[base + c]

        logits: list[float] = []
        for c in range(num_classes):
            base = c * dim
            total = bias[c] + direct_logits[c] * inv
            for j in range(dim):
                total += classifier[base + j] * hidden[j]
            logits.append(total)

        best = max(range(num_classes), key=lambda i: logits[i])
        top = logits[best]
        if top <= 0:
            return str(classes[best]), 0.0
        second = max(
            (value for index, value in enumerate(logits) if index != best),
            default=0.0,
        )
        margin = max(0.0, top - second)
        return str(classes[best]), (margin / (1.0 + margin)) * coverage

    # -- v2 inference ----------------------------------------------------

    def _v2_weights(self) -> tuple[list[float], list[float], list[float], int, int]:
        cached = _WEIGHTS_CACHE.get(self.weights_sha256)
        if cached is not None:
            return cached
        artifact = self.artifact
        buckets = int(artifact["buckets"])  # type: ignore[arg-type]
        dim = int(artifact["dim"])  # type: ignore[arg-type]
        classes = artifact["classes"]
        assert isinstance(classes, list)
        embedding_block = artifact["embedding"]
        classifier_block = artifact["classifier"]
        assert isinstance(embedding_block, dict) and isinstance(classifier_block, dict)
        embedding = _dequantize(
            str(embedding_block["data"]), float(embedding_block["scale"]), buckets * dim, self.source
        )
        classifier = _dequantize(
            str(classifier_block["data"]), float(classifier_block["scale"]), len(classes) * dim, self.source
        )
        bias = [float(v) for v in artifact["bias"]]  # type: ignore[union-attr]
        result = (embedding, classifier, bias, buckets, dim)
        _WEIGHTS_CACHE[self.weights_sha256] = result
        return result

    def _v2_score(self, prompt: str) -> tuple[str | None, float]:
        classes = self.artifact["classes"]
        assert isinstance(classes, list)
        exact = self.artifact.get("exact_prompts", {})
        assert isinstance(exact, dict)
        normalized_prompt = _normalized_text(prompt)
        hit = exact.get(normalized_prompt) if normalized_prompt else None
        if isinstance(hit, int):
            return str(classes[hit]), 1.0

        vocabulary = self.artifact.get("vocabulary", [])
        assert isinstance(vocabulary, list)
        tokens = tokenize(prompt)
        if len(tokens) < 3 or not set(tokens).intersection(vocabulary):
            return None, 0.0

        embedding, classifier, bias, buckets, dim = self._v2_weights()
        features = _features(prompt, buckets)
        if not features:
            return None, 0.0
        hidden = [0.0] * dim
        for bucket in features:
            base = bucket * dim
            for j in range(dim):
                hidden[j] += embedding[base + j]
        scale = 1.0 / len(features)
        hidden = [value * scale for value in hidden]

        logits: list[float] = []
        for c in range(len(classes)):
            base = c * dim
            total = bias[c]
            for j in range(dim):
                total += classifier[base + j] * hidden[j]
            logits.append(total)
        top = max(logits)
        exps = [math.exp(value - top) for value in logits]
        denom = sum(exps)
        best = max(range(len(classes)), key=lambda i: logits[i])
        return str(classes[best]), exps[best] / denom

    # -- v1 inference ----------------------------------------------------

    def _v1_prompt_vector(self, prompt: str) -> dict[str, float]:
        idf = self.artifact["idf"]
        assert isinstance(idf, dict)
        counts = Counter(_ngrams(tokenize(prompt)))
        vector = {
            term: count * float(idf[term])
            for term, count in counts.items()
            if term in idf
        }
        norm = math.sqrt(sum(value * value for value in vector.values()))
        if norm == 0:
            return {}
        return {term: value / norm for term, value in vector.items()}

    def _v1_score(self, prompt: str) -> tuple[str | None, float]:
        normalized = _normalized_text(prompt)
        if not normalized:
            return None, 0.0
        examples = self.artifact["examples"]
        assert isinstance(examples, list)
        for example in examples:
            assert isinstance(example, dict)
            if _normalized_text(str(example["prompt"])) == normalized:
                return str(example["completion"]), 1.0
        vector = self._v1_prompt_vector(prompt)
        if not vector:
            return None, 0.0
        best_completion: str | None = None
        best_score = 0.0
        for example in examples:
            assert isinstance(example, dict)
            example_vector = example["prompt_vector"]
            assert isinstance(example_vector, dict)
            score = sum(
                value * float(example_vector.get(term, 0.0))
                for term, value in vector.items()
            )
            if score > best_score:
                best_score = score
                best_completion = str(example["completion"])
        return best_completion, best_score

    # -- shared API --------------------------------------------------------

    def score(self, prompt: str) -> tuple[str | None, float]:
        """Return the best completion and its confidence score."""

        if self.is_v3:
            return self._v3_score(prompt)
        if self.is_v2:
            return self._v2_score(prompt)
        return self._v1_score(prompt)

    def predict(
        self, prompt: str, *, threshold: float = DEFAULT_MATCH_THRESHOLD
    ) -> str | None:
        """Return a trained completion for a confident match, else None."""

        completion, score = self.score(prompt)
        if completion is not None and score >= threshold:
            return completion
        return None

    def continue_text(self, seed: str, *, max_tokens: int = 12) -> str:
        """Continue text with the learned bigram transition weights."""

        if max_tokens <= 0:
            return ""
        transitions = self.artifact["transitions"]
        assert isinstance(transitions, dict)
        tokens = tokenize(seed)
        previous = tokens[-1] if tokens else _START
        generated: list[str] = []
        for _ in range(max_tokens):
            next_weights = transitions.get(previous)
            if not isinstance(next_weights, dict) or not next_weights:
                break
            next_token = max(next_weights.items(), key=lambda item: (float(item[1]), item[0]))[0]
            if next_token == _END:
                break
            generated.append(next_token)
            previous = next_token
        return " ".join(generated)

    def to_dict(self) -> dict[str, object]:
        return dict(self.artifact)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _clean_examples(examples: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    cleaned: list[tuple[str, str]] = []
    # Prompts are matched by their normalized text (lowercase word
    # tokens), so duplicates and conflicts are judged on that form:
    # "Hello!" and "hello" are the same prompt to the model. The
    # same normalized prompt with two different completions is
    # contradictory training data — reject it instead of silently
    # keeping whichever completion came last. A prompt or completion
    # with no word tokens at all ("!!!") can never be learned or
    # matched honestly, so it is rejected here with a clear error
    # rather than failing later inside artifact validation.
    by_normalized: dict[str, str] = {}
    for prompt, completion in examples:
        if not isinstance(prompt, str) or not tokenize(prompt):
            raise TrainingError(
                "local Gravion Core training requires prompts with "
                "at least one word or number"
            )
        if not isinstance(completion, str) or not tokenize(completion):
            raise TrainingError(
                "local Gravion Core training requires completions with "
                "at least one word or number"
            )
        pair = (prompt.strip(), completion.strip())
        normalized = _normalized_text(pair[0])
        previous = by_normalized.get(normalized)
        if previous is not None:
            if previous != pair[1]:
                raise TrainingError(
                    "local Gravion Core training has conflicting "
                    f"completions for the same prompt {normalized!r}"
                )
            continue
        by_normalized[normalized] = pair[1]
        cleaned.append(pair)
    if not cleaned:
        raise TrainingError("local Gravion Core training requires at least one example")
    return cleaned


def _transition_weights(
    cleaned: Sequence[tuple[str, str]],
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    transition_counts: dict[str, Counter[str]] = defaultdict(Counter)
    unigram_counts: Counter[str] = Counter()
    for _, completion in cleaned:
        tokens = tokenize(completion)
        unigram_counts.update(tokens)
        sequence = [_START, *tokens, _END]
        for previous, next_token in zip(sequence, sequence[1:], strict=False):
            transition_counts[previous][next_token] += 1
    transitions = {
        previous: {
            next_token: round(count / sum(next_counts.values()), 8)
            for next_token, count in sorted(next_counts.items())
        }
        for previous, next_counts in sorted(transition_counts.items())
    }
    total_unigrams = sum(unigram_counts.values()) or 1
    unigram_weights = {
        token: round(count / total_unigrams, 8)
        for token, count in sorted(unigram_counts.items())
    }
    return transitions, unigram_weights


def train_local_core(
    examples: Iterable[tuple[str, str]],
    *,
    model_id: str = CORE_MODEL_ID,
    source: str = "<dataset>",
    buckets: int | None = None,
    dim: int = DEFAULT_DIM,
    epochs: int = DEFAULT_EPOCHS,
    negatives: int = DEFAULT_NEGATIVES,
    max_features: int | None = None,
) -> GravionCoreLocalModel:
    """Train Gravion Core's sparse embedding/linear model.

    Explicit word n-gram features from the training prompts are scored
    by a learned direct feature-to-class matrix plus a low-dimensional
    learned embedding classifier over the distinct completions.
    Training is sampled-softmax SGD with a fixed deterministic
    initialization, so the same dataset always produces the same
    weights. The legacy ``buckets`` argument is accepted as a feature
    cap for callers of the earlier hashed model.
    """

    if model_id not in LOCAL_MODEL_IDS:
        raise TrainingError(f"local Gravion training does not support {model_id!r}")
    if dim <= 0:
        raise TrainingError("local Gravion Core training requires dim > 0")
    if epochs <= 0:
        raise TrainingError("local Gravion Core training requires epochs > 0")
    if negatives < 0:
        raise TrainingError("local Gravion Core training requires negatives >= 0")
    if max_features is None and buckets is not None:
        max_features = buckets
    if max_features is None:
        max_features = DEFAULT_MAX_FEATURES
    if max_features is not None and max_features <= 0:
        raise TrainingError("local Gravion Core training requires max_features > 0")

    cleaned = _clean_examples(examples)
    classes = sorted({completion for _, completion in cleaned})
    class_index = {completion: index for index, completion in enumerate(classes)}
    num_classes = len(classes)

    feature_counts: Counter[str] = Counter()
    feature_class_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for prompt, completion in cleaned:
        for term in set(_feature_terms(prompt)):
            feature_counts[term] += 1
            feature_class_counts[term][completion] += 1
    features = sorted(
        feature_counts,
        key=lambda term: (
            -max(feature_class_counts[term].values()),
            -feature_counts[term],
            term,
        ),
    )
    if max_features is not None:
        features = features[:max_features]
    if not features:
        raise TrainingError("local Gravion Core training requires prompt features")
    feature_index = {term: index for index, term in enumerate(features)}
    num_features = len(features)

    rng = random.Random(_INIT_SEED)
    direct = array("f", [0.0]) * (num_features * num_classes)
    embedding = array(
        "f", ((rng.random() - 0.5) * 0.02 for _ in range(num_features * dim))
    )
    classifier = array(
        "f", ((rng.random() - 0.5) * 0.02 for _ in range(num_classes * dim))
    )
    bias = array("f", [0.0]) * num_classes

    prepared = [
        (
            [
                feature_index[term]
                for term in _feature_terms(prompt)
                if term in feature_index
            ],
            class_index[completion],
        )
        for prompt, completion in cleaned
    ]
    order = list(range(len(prepared)))
    for epoch in range(epochs):
        rng.shuffle(order)
        lr = _LEARNING_RATE / (1.0 + 0.08 * epoch)
        for index in order:
            feature_ids, target = prepared[index]
            if not feature_ids:
                continue
            inv = 1.0 / len(feature_ids)
            hidden = [0.0] * dim
            for feature_id in feature_ids:
                base = feature_id * dim
                for j in range(dim):
                    hidden[j] += embedding[base + j]
            hidden = [value * inv for value in hidden]

            if num_classes <= negatives + 1:
                candidates = list(range(num_classes))
            else:
                candidates = [target]
                seen = {target}
                while len(candidates) < negatives + 1:
                    candidate = rng.randrange(num_classes)
                    if candidate not in seen:
                        seen.add(candidate)
                        candidates.append(candidate)

            logits: list[float] = []
            for c in candidates:
                direct_sum = 0.0
                for feature_id in feature_ids:
                    direct_sum += direct[feature_id * num_classes + c]
                base = c * dim
                total = bias[c] + direct_sum * inv
                for j in range(dim):
                    total += classifier[base + j] * hidden[j]
                logits.append(total)
            top = max(logits)
            exps = [math.exp(value - top) for value in logits]
            denom = sum(exps)
            probs = [value / denom for value in exps]

            grad_hidden = [0.0] * dim
            for position, c in enumerate(candidates):
                grad = probs[position] - (1.0 if c == target else 0.0)
                if grad == 0.0:
                    continue
                bias[c] -= lr * grad
                direct_step = lr * grad * inv
                for feature_id in feature_ids:
                    weight_index = feature_id * num_classes + c
                    direct[weight_index] -= direct_step + lr * _L2 * direct[weight_index]
                base = c * dim
                for j in range(dim):
                    grad_hidden[j] += grad * classifier[base + j]
                    classifier[base + j] -= lr * (
                        grad * hidden[j] + _L2 * classifier[base + j]
                    )
            embedding_step = lr * inv
            for feature_id in feature_ids:
                base = feature_id * dim
                for j in range(dim):
                    embedding[base + j] -= (
                        embedding_step * grad_hidden[j]
                        + lr * _L2 * embedding[base + j]
                    )

    direct_data, direct_scale = _quantize(direct)
    embedding_data, embedding_scale = _quantize(embedding)
    classifier_data, classifier_scale = _quantize(classifier)
    transitions, unigram_weights = _transition_weights(cleaned)
    exact_prompts = {
        _normalized_text(prompt): class_index[completion] for prompt, completion in cleaned
    }

    artifact: dict[str, object] = {
        "format": CORE_MODEL_FORMAT,
        "model_id": model_id,
        "architecture": CORE_ARCHITECTURE,
        "trained_from": {
            "source": source,
            "examples": len(cleaned),
            "classes": num_classes,
            "features": num_features,
            "dataset_sha256": _dataset_sha256(cleaned),
        },
        "features": features,
        "dim": dim,
        "classes": classes,
        "direct": {"encoding": "int8-base64", "scale": direct_scale, "data": direct_data},
        "embedding": {"encoding": "int8-base64", "scale": embedding_scale, "data": embedding_data},
        "classifier": {"encoding": "int8-base64", "scale": classifier_scale, "data": classifier_data},
        "bias": [round(float(value), 8) for value in bias],
        "vocabulary": sorted(
            {token for prompt, completion in cleaned for token in tokenize(prompt) + tokenize(completion)}
        ),
        "exact_prompts": exact_prompts,
        "transitions": transitions,
        "unigram_weights": unigram_weights,
    }
    artifact["parameter_count"] = count_parameters(artifact)
    artifact["weights_sha256"] = _weights_sha256(artifact)
    return GravionCoreLocalModel(_validate_artifact(artifact, source), source=source)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_weights(model: GravionCoreLocalModel, path: Path | str) -> dict[str, object]:
    """Save a trained Gravion Core model artifact and return its report."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(model.to_dict(), indent=2, sort_keys=True) + "\n"
    destination.write_text(text, encoding="utf-8")
    artifact = model.artifact
    if model.is_v3:
        examples = artifact["trained_from"]["examples"]  # type: ignore[index]
        vocabulary_size = len(artifact["features"])  # type: ignore[arg-type]
    elif model.is_v2:
        examples = artifact["trained_from"]["examples"]  # type: ignore[index]
        vocabulary_size = artifact["buckets"]
    else:
        examples = len(artifact["examples"])  # type: ignore[arg-type]
        vocabulary_size = len(artifact["vocabulary"])  # type: ignore[arg-type]
    return {
        "model": model.model_id,
        "weights_path": str(destination),
        "parameter_count": model.parameter_count,
        "weights_sha256": model.weights_sha256,
        "file_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "examples": examples,
        "vocabulary_size": vocabulary_size,
    }


def load_weights(
    path: Path | str, *, model_id: str | None = None
) -> GravionCoreLocalModel:
    """Load and validate a Gravion local weights artifact."""

    source = Path(path)
    try:
        artifact = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise TrainingError(f"Gravion Core weights not found: {source}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainingError(f"Gravion Core weights are not valid JSON: {exc}") from exc
    model = GravionCoreLocalModel(
        _validate_artifact(artifact, str(source)), source=str(source)
    )
    if model_id is not None and model.model_id != model_id:
        raise TrainingError(
            f"{source}: weights are for {model.model_id!r}, expected {model_id!r}"
        )
    return model


def weights_filename(model_id: str = CORE_MODEL_ID) -> str:
    if model_id not in LOCAL_MODEL_IDS:
        raise TrainingError(f"unknown bundled local Gravion model {model_id!r}")
    return f"{model_id}-weights.json"


def seed_filename(model_id: str = CORE_MODEL_ID) -> str:
    if model_id not in LOCAL_MODEL_IDS:
        raise TrainingError(f"unknown bundled local Gravion model {model_id!r}")
    return f"{model_id}-seed.jsonl"


def bundled_seed_path(model_id: str = CORE_MODEL_ID) -> Path:
    return Path(
        str(resources.files("gravion_ai").joinpath("data", seed_filename(model_id)))
    )


def load_bundled_model(model_id: str = CORE_MODEL_ID) -> GravionCoreLocalModel:
    """Load a bundled Gravion local weights artifact."""

    resource = resources.files("gravion_ai").joinpath(
        "data", weights_filename(model_id)
    )
    try:
        artifact = json.loads(resource.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainingError(f"bundled {model_id} weights are unavailable: {exc}") from exc
    model = GravionCoreLocalModel(
        _validate_artifact(artifact, f"bundled {model_id} weights"),
        source=f"bundled {model_id} weights",
    )
    if model.model_id != model_id:
        raise TrainingError(
            f"bundled weights are for {model.model_id!r}, expected {model_id!r}"
        )
    return model


def deployment_weights_path(
    home: Path | str, model_id: str = CORE_MODEL_ID
) -> Path:
    """The deployment-local Gravion weights path under data/models."""

    return Path(home) / "data" / "models" / weights_filename(model_id)
