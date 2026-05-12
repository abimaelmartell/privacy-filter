"""Core runtime loading and one-shot text prediction helpers."""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
import tiktoken

from .._common.constants import (
    OUTPUT_MODES,
    REDACTED_OUTPUT_LABEL,
    REDACTED_OUTPUT_PLACEHOLDER,
)
from .._common.env import get_env_bool
from .decoding import ViterbiCRFDecoder
from .._common.label_space import resolve_label_space_from_config
from .spans import (
    decode_text_with_offsets,
    discard_overlapping_spans_by_label,
    labels_to_spans,
    token_spans_to_char_spans,
    trim_char_spans_whitespace,
)
from .sequence_labeling import (
    ExampleAggregation,
    LabelInfo,
    TokenizedExample,
    build_label_info,
    example_to_windows,
)
from .._model.model import Transformer


@dataclass(frozen=True)
class InferenceRuntime:
    """Loaded model runtime and decode-time metadata for one OPF instance."""

    checkpoint: str
    model: Transformer
    encoding: tiktoken.Encoding
    label_info: LabelInfo
    device: torch.device
    n_ctx: int
    trim_span_whitespace: bool
    discard_overlapping_predicted_spans: bool
    output_mode: str
    active_encoding_name: str
    pad_token_id: int
    bidirectional_context: bool
    category_version: str


@dataclass(frozen=True)
class DetectedSpan:
    """One detected character span ready for rendering or serialization."""

    label: str
    start: int
    end: int
    text: str
    placeholder: str


@dataclass(frozen=True)
class PredictionResult:
    """Raw inference output before higher-level API serialization."""

    text: str
    spans: tuple[DetectedSpan, ...]
    decoded_mismatch: bool


def build_detection_summary(
    *,
    output_mode: str,
    labels: Sequence[str],
    decoded_mismatch: bool,
) -> dict[str, object]:
    """Build a compact summary for structured prediction output."""
    by_label: dict[str, int] = {}
    for label in labels:
        by_label[label] = by_label.get(label, 0) + 1
    return {
        "output_mode": output_mode,
        "span_count": len(labels),
        "by_label": dict(sorted(by_label.items(), key=lambda item: item[0])),
        "decoded_mismatch": decoded_mismatch,
    }


def _apply_output_mode_to_detected_spans(
    spans: Sequence[DetectedSpan],
    *,
    output_mode: str,
) -> list[DetectedSpan]:
    """Apply typed vs redacted output rendering to detected spans."""
    if output_mode == "typed":
        return list(spans)
    if output_mode != "redacted":
        raise ValueError(f"Unsupported output_mode: {output_mode!r}")
    return [
        DetectedSpan(
            label=REDACTED_OUTPUT_LABEL,
            start=span.start,
            end=span.end,
            text=span.text,
            placeholder=REDACTED_OUTPUT_PLACEHOLDER,
        )
        for span in spans
    ]


def _load_checkpoint_config(checkpoint_dir: str) -> dict[str, object]:
    """Load and validate the checkpoint JSON config object."""
    config_path = Path(checkpoint_dir) / "config.json"
    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(
            f"Checkpoint config at {config_path} must contain a JSON object"
        )
    return payload


def _resolve_n_ctx(
    checkpoint_config: dict[str, object],
    override_n_ctx: int | None,
    device: torch.device,
) -> int:
    """Resolve the effective context length for the current runtime."""
    if override_n_ctx is not None:
        if override_n_ctx <= 0:
            raise ValueError("n_ctx must be positive")
        return override_n_ctx
    if device.type == "cpu":
        # CPU full-eval/demo should default to a safer context size.
        return 4096

    for field_name in (
        "default_n_ctx",
        "initial_context_length",
        "max_position_embeddings",
    ):
        if field_name not in checkpoint_config:
            continue
        value = checkpoint_config[field_name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                f"Checkpoint config field {field_name} must be a positive integer"
            )
        if value <= 0:
            raise ValueError(f"Checkpoint config field {field_name} must be positive")
        return value

    return 4096


def _validate_checkpoint_dir(checkpoint_dir: str) -> None:
    """Ensure a checkpoint directory exists and contains the expected files."""
    path = Path(checkpoint_dir)
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    config_path = path / "config.json"
    if not config_path.exists() or not config_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint config: {config_path}")
    if not any(path.glob("*.safetensors")):
        raise FileNotFoundError(
            f"Checkpoint directory has no .safetensors files: {checkpoint_dir}"
        )


def _label_placeholder(label: str) -> str:
    """Convert a span label into the placeholder token shown to users."""
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", label.upper()).strip("_")
    if not normalized:
        normalized = "REDACTED"
    return f"<{normalized}>"


def _select_non_overlapping_spans(spans: Sequence[DetectedSpan]) -> list[DetectedSpan]:
    """Keep a left-to-right non-overlapping subset of detected spans."""
    ordered = sorted(
        spans, key=lambda span: (span.start, -(span.end - span.start), span.label)
    )
    kept: list[DetectedSpan] = []
    cursor = 0
    for span in ordered:
        if span.start < cursor:
            continue
        if span.end <= span.start:
            continue
        kept.append(span)
        cursor = span.end
    return kept


def load_inference_runtime(
    *,
    checkpoint: str,
    device_name: str,
    n_ctx_override: int | None = None,
    trim_span_whitespace: bool,
    discard_overlapping_predicted_spans: bool,
    output_mode: str,
) -> InferenceRuntime:
    """Load model, tokenizer, label space, and runtime metadata for inference."""
    if output_mode not in OUTPUT_MODES:
        raise ValueError(f"Unsupported output_mode: {output_mode!r}")
    _validate_checkpoint_dir(checkpoint)
    device = torch.device(device_name)
    checkpoint_config = _load_checkpoint_config(checkpoint)
    n_ctx = _resolve_n_ctx(checkpoint_config, n_ctx_override, device)
    encoding_name = checkpoint_config.get("encoding")
    if not isinstance(encoding_name, str) or not encoding_name:
        raise ValueError("Checkpoint config field encoding must be a non-empty string")
    encoding = tiktoken.get_encoding(encoding_name)
    pad_token_id = int(encoding.eot_token)
    config_context = str(Path(checkpoint) / "config.json")
    resolved_category_version, _span_class_names, resolved_ner_class_names = (
        resolve_label_space_from_config(checkpoint_config, context=config_context)
    )
    label_info = build_label_info(resolved_ner_class_names)
    model = Transformer.from_checkpoint(
        checkpoint,
        device=device,
    )
    bidirectional_context = checkpoint_config["bidirectional_context"]
    model.eval()
    if get_env_bool("OPF_TORCH_COMPILE"):
        compile_mode = os.environ.get("OPF_TORCH_COMPILE_MODE", "default")
        model = torch.compile(model, mode=compile_mode)
    return InferenceRuntime(
        checkpoint=checkpoint,
        model=model,
        encoding=encoding,
        label_info=label_info,
        device=device,
        n_ctx=n_ctx,
        trim_span_whitespace=trim_span_whitespace,
        discard_overlapping_predicted_spans=discard_overlapping_predicted_spans,
        output_mode=output_mode,
        active_encoding_name=encoding_name,
        pad_token_id=pad_token_id,
        bidirectional_context=bidirectional_context,
        category_version=resolved_category_version,
    )


@torch.inference_mode()
def predict_text(
    runtime: InferenceRuntime,
    text: str,
    *,
    decoder: ViterbiCRFDecoder | None,
) -> PredictionResult:
    """Run one text through the model and return decoded detected spans."""
    token_ids = tuple(
        int(tok) for tok in runtime.encoding.encode(text, allowed_special="all")
    )
    if not token_ids:
        return PredictionResult(text=text, spans=(), decoded_mismatch=False)

    example_id = "demo-example"
    background = int(runtime.label_info.background_token_label)
    example = TokenizedExample(
        tokens=token_ids,
        labels=tuple(background for _ in token_ids),
        example_id=example_id,
        text=text,
    )
    aggregation = ExampleAggregation(
        logprob_logsumexp=[], counts=[], labels=[], token_ids=[]
    )

    for window in example_to_windows(
        example,
        runtime.n_ctx,
    ):
        if not window.tokens:
            continue
        window_tokens = torch.tensor(
            [list(window.tokens)],
            device=runtime.device,
            dtype=torch.int32,
        )
        attention_mask = torch.ones_like(window_tokens, dtype=torch.bool)
        logits = runtime.model(window_tokens, attention_mask=attention_mask)
        log_probs = F.log_softmax(logits.float(), dim=-1)[0].cpu()
        if log_probs.shape[0] != len(window.tokens):
            raise ValueError("Logprob output length does not match window length")

        # Fast path for the common non-overlapping configuration
        # (example_to_windows produces stride == window_size, so every
        # token sits in exactly one window). Skip the per-token Python
        # loop entirely and bulk-extend the aggregation lists. On
        # long inputs this removes O(N_tokens) iterations per window,
        # each running ensure_capacity / logaddexp / record_token_id
        # on count == 1 data — purely no-op work.
        first_offset = int(window.offsets[0])
        if (
            all(window.mask)
            and first_offset == aggregation.length
            and tuple(int(o) for o in window.offsets)
            == tuple(range(first_offset, first_offset + len(window.tokens)))
        ):
            aggregation.logprob_logsumexp.extend(log_probs.unbind(0))
            aggregation.counts.extend([1] * len(window.tokens))
            aggregation.labels.extend([None] * len(window.tokens))
            aggregation.token_ids.extend(int(t) for t in window.tokens)
            aggregation.length = first_offset + len(window.tokens)
            continue

        for token_pos, is_valid in enumerate(window.mask):
            if not bool(is_valid):
                continue
            token_idx = int(window.offsets[token_pos])
            if token_idx < 0:
                continue
            aggregation.ensure_capacity(token_idx)
            score_vec = log_probs[token_pos]
            existing = aggregation.logprob_logsumexp[token_idx]
            if existing is None:
                aggregation.logprob_logsumexp[token_idx] = score_vec.clone()
            else:
                aggregation.logprob_logsumexp[token_idx] = torch.logaddexp(
                    existing, score_vec
                )
            aggregation.counts[token_idx] += 1
            aggregation.record_token_id(
                token_idx, int(window.tokens[token_pos]), example_id
            )
            aggregation.length = max(aggregation.length, token_idx + 1)

    token_positions: list[int] = []
    token_score_vectors: list[torch.Tensor] = []
    for token_idx in range(aggregation.length):
        if token_idx >= len(aggregation.logprob_logsumexp):
            continue
        score_sum = aggregation.logprob_logsumexp[token_idx]
        count = aggregation.counts[token_idx]
        if score_sum is None or count <= 0:
            continue
        avg_logprob = score_sum - math.log(float(count))
        token_positions.append(token_idx)
        token_score_vectors.append(avg_logprob)

    if not token_score_vectors:
        return PredictionResult(text=text, spans=(), decoded_mismatch=False)

    stacked_scores = torch.stack(token_score_vectors, dim=0)
    if decoder is not None:
        decoded_labels = decoder.decode(stacked_scores)
        if len(decoded_labels) != len(token_positions):
            decoded_labels = stacked_scores.argmax(dim=1).tolist()
    else:
        decoded_labels = stacked_scores.argmax(dim=1).tolist()
    predicted_labels_by_index = {
        token_idx: int(label)
        for token_idx, label in zip(token_positions, decoded_labels)
    }
    predicted_token_spans = labels_to_spans(
        predicted_labels_by_index, runtime.label_info
    )

    decoded_text, char_starts, char_ends = decode_text_with_offsets(
        token_ids, runtime.encoding
    )
    decoded_mismatch = decoded_text != text
    source_text = decoded_text if decoded_mismatch else text

    predicted_char_spans = token_spans_to_char_spans(
        predicted_token_spans, char_starts, char_ends
    )
    if runtime.trim_span_whitespace:
        predicted_char_spans = trim_char_spans_whitespace(
            predicted_char_spans, source_text
        )
    if runtime.discard_overlapping_predicted_spans:
        predicted_char_spans = discard_overlapping_spans_by_label(predicted_char_spans)

    detected: list[DetectedSpan] = []
    for label_idx, start, end in predicted_char_spans:
        if not (0 <= start < end <= len(source_text)):
            continue
        label = (
            str(runtime.label_info.span_class_names[label_idx])
            if 0 <= int(label_idx) < len(runtime.label_info.span_class_names)
            else f"label_{label_idx}"
        )
        detected.append(
            DetectedSpan(
                label=label,
                start=int(start),
                end=int(end),
                text=source_text[start:end],
                placeholder=_label_placeholder(label),
            )
        )

    display_spans = _apply_output_mode_to_detected_spans(
        _select_non_overlapping_spans(detected),
        output_mode=runtime.output_mode,
    )
    return PredictionResult(
        text=source_text,
        spans=tuple(display_spans),
        decoded_mismatch=decoded_mismatch,
    )
