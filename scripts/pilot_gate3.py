#!/usr/bin/env python3
"""Gate 3 Pilot -- 40-sample TruthfulQA characterization.

Two entry points:
  python scripts/pilot_gate3.py                    # run generation
  python scripts/pilot_gate3.py --analyze PATH     # analyze reviewed results

Implements: docs/plans/GATE3_PILOT_SPEC.md (post-audit revision)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessorList,
)

from navi_sad.core.instrument import InstrumentManager
from navi_sad.core.registry import get_family_config
from navi_sad.core.types import RawSampleRecord
from navi_sad.io.writer import RawRecordWriter
from navi_sad.pilot.helpers import (
    compute_cohens_d,
    compute_confusion_matrix,
    compute_mean_delta_matrix,
    extract_leading_span,
    find_leading_span_token_count,
    score_sample,
    validate_review_integrity,
)
from navi_sad.version import __version__

logger = logging.getLogger(__name__)

# Frozen constants (spec sections 2, 4, 5)
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
REVISION = "63a8b081895390a26e140280378bc85ec8bce07a"
SEED = 42
DEFAULT_SAMPLE_COUNT = 40
MAX_NEW_TOKENS = 256
DECODE_KWARGS: dict[str, Any] = {
    "skip_special_tokens": True,
    "clean_up_tokenization_spaces": False,
}


# -------------------------------------------------------------------
# Generation entry point
# -------------------------------------------------------------------


def run_generation(args: argparse.Namespace) -> None:
    """Run pilot generation: load model, generate, write artifacts."""
    import datasets as ds

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------------
    logger.info("Loading TruthfulQA dataset...")
    dataset = ds.load_dataset("truthful_qa", "generation", split="validation")
    rng = random.Random(SEED)
    selected_indices = sorted(rng.sample(range(len(dataset)), args.sample_count))
    logger.info(
        "Selected %d samples (seed=%d): %s",
        len(selected_indices),
        SEED,
        selected_indices,
    )

    # ---------------------------------------------------------------
    # Model + tokenizer
    # ---------------------------------------------------------------
    logger.info("Loading model %s (revision=%s)...", MODEL_ID, REVISION[:12])
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=REVISION,
        torch_dtype=torch.float16,
        device_map="cuda",
        attn_implementation="eager",
    )
    model.eval()

    # Tokenizer provenance
    assert tokenizer.chat_template is not None, "Tokenizer has no chat template"
    chat_template_hash = hashlib.sha256(tokenizer.chat_template.encode("utf-8")).hexdigest()
    tokenizer_revision = REVISION  # same as model for Mistral

    # ---------------------------------------------------------------
    # Instrument manager
    # ---------------------------------------------------------------
    family_config = get_family_config(model.config)
    num_q_heads = getattr(model.config, family_config.num_q_heads_attr)
    num_kv_heads = getattr(model.config, family_config.num_kv_heads_attr)
    num_layers = model.config.num_hidden_layers

    mgr = InstrumentManager(family_config, sink_exclude=1)
    for layer_idx in range(num_layers):
        attn_path = family_config.attn_module_path.format(layer_idx)
        attn_module = model.get_submodule(attn_path)
        mgr.install_layer(attn_module, layer_idx, num_q_heads, num_kv_heads)

    step_callback = mgr.make_step_callback()

    # ---------------------------------------------------------------
    # Metadata
    # ---------------------------------------------------------------
    pilot_metadata = {
        "seed": SEED,
        "selected_indices": selected_indices,
        "burned_indices": selected_indices,
        "dataset_name": "truthful_qa",
        "dataset_config": "generation",
        "dataset_split": "validation",
        "datasets_version": ds.__version__,
        "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
        "model_id": MODEL_ID,
        "model_revision": REVISION,
        "tokenizer_id": MODEL_ID,
        "tokenizer_revision": tokenizer_revision,
        "chat_template_hash": chat_template_hash,
        "transformers_version": _get_transformers_version(),
        "navi_sad_version": __version__,
        "decode_settings": DECODE_KWARGS,
    }

    # ---------------------------------------------------------------
    # Generation loop
    # ---------------------------------------------------------------
    samples: list[dict[str, Any]] = []
    reviews: list[dict[str, Any]] = []
    eos_token_id = tokenizer.eos_token_id

    raw_path = output_dir / "raw.jsonl.gz"
    with RawRecordWriter(raw_path) as raw_writer:
        for i, dataset_idx in enumerate(selected_indices):
            t0 = time.monotonic()
            row = dataset[dataset_idx]

            # Render prompt via chat template
            messages = [{"role": "user", "content": row["question"]}]
            rendered = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            input_ids = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)[
                "input_ids"
            ].to(model.device)
            assert input_ids.shape[0] == 1, f"Expected B=1, got {input_ids.shape[0]}"

            prompt_token_ids = input_ids[0].tolist()
            prompt_length = len(prompt_token_ids)

            # Generate
            mgr.reset()
            with torch.no_grad():
                output = model.generate(
                    input_ids,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    use_cache=False,
                    logits_processor=LogitsProcessorList([step_callback]),
                )

            # Extract generated tokens, exclude terminal EOS
            full_gen_ids = output[0][prompt_length:].tolist()
            if full_gen_ids and full_gen_ids[-1] == eos_token_id:
                generated_token_ids = full_gen_ids[:-1]
                stop_reason = "eos"
            else:
                generated_token_ids = full_gen_ids
                stop_reason = "max_length"

            generated_token_count = len(generated_token_ids)
            generation_text = tokenizer.decode(generated_token_ids, **DECODE_KWARGS)

            # Records from instrument — filter to exclude any EOS-generating
            # forward step. The LogitsProcessor increments step_idx after
            # each forward, so the EOS-producing step can land in records.
            # Align record boundary with generated_token_count.
            all_records = mgr.get_records()
            records = [r for r in all_records if r.step_idx < generated_token_count]

            # Leading span + scorer
            span, span_stop_reason = extract_leading_span(generation_text)
            scorer_label, matched_correct, matched_incorrect = score_sample(
                span, row["correct_answers"], row["incorrect_answers"]
            )

            # Leading-span token alignment
            if generated_token_count > 0 and span.strip():
                ls_count, ls_fallback = find_leading_span_token_count(
                    generated_token_ids, span, tokenizer, DECODE_KWARGS
                )
            elif generated_token_count == 0:
                ls_count = 0
                ls_fallback = False
            else:
                ls_count = 0
                ls_fallback = False

            # Scalar matrices
            full_gen_matrix = compute_mean_delta_matrix(records, num_layers, num_q_heads)
            if ls_count > 0:
                leading_span_matrix = compute_mean_delta_matrix(
                    records, num_layers, num_q_heads, max_step=ls_count
                )
            else:
                leading_span_matrix = None

            # Build per-step list for artifact
            per_step = [
                {
                    "step_idx": r.step_idx,
                    "layer_idx": r.layer_idx,
                    "per_head_delta": r.per_head_delta,
                }
                for r in records
            ]

            # Build sample record
            sample: dict[str, Any] = {
                "dataset_index": dataset_idx,
                "question": row["question"],
                "best_answer": row["best_answer"],
                "correct_answers": row["correct_answers"],
                "incorrect_answers": row["incorrect_answers"],
                "rendered_prompt": rendered,
                "prompt_token_ids": prompt_token_ids,
                "prompt_token_count": prompt_length,
                "generated_token_ids": generated_token_ids,
                "generated_token_count": generated_token_count,
                "generation_text": generation_text,
                "stop_reason": stop_reason,
                "per_step": per_step,
                "full_gen_mean_delta": full_gen_matrix,
                "leading_span_mean_delta": leading_span_matrix,
                "leading_span_token_count": ls_count,
                "leading_span_fallback": ls_fallback,
                "scorer_label": scorer_label,
                "scorer_leading_span": span,
                "scorer_leading_span_stop_reason": span_stop_reason,
                "scorer_matched_correct": matched_correct,
                "scorer_matched_incorrect": matched_incorrect,
            }
            samples.append(sample)

            # Build review record
            review: dict[str, Any] = {
                "dataset_index": dataset_idx,
                "question": row["question"],
                "best_answer": row["best_answer"],
                "correct_answers": row["correct_answers"],
                "incorrect_answers": row["incorrect_answers"],
                "rendered_prompt": rendered,
                "generation_text": generation_text,
                "generated_token_count": generated_token_count,
                "scorer_label": scorer_label,
                "scorer_leading_span": span,
                "scorer_leading_span_stop_reason": span_stop_reason,
                "scorer_matched_correct": matched_correct,
                "scorer_matched_incorrect": matched_incorrect,
                "human_label": "",
                "disagreement_category": "",
                "disagreement_note": "",
            }
            reviews.append(review)

            # Write raw JSONL record
            raw_record = RawSampleRecord(
                sample_id=f"pilot_gate3_{dataset_idx}",
                model=MODEL_ID,
                benchmark="truthfulqa",
                prompt=rendered,
                generation=generation_text,
                label=scorer_label,
                label_source="truthfulqa_exact_v1",
                scorer_version="truthfulqa_exact_v1",
                num_tokens_generated=generated_token_count,
                layers_hooked=list(range(num_layers)),
                capture_tier="A",
                per_step=list(records),
                metadata={
                    "native_dtype": "float16",
                    "instrument_dtype": "float32",
                    "kv_cache": False,
                    "sink_excluded_positions": 1,
                    "quantization": "none",
                    "generation_config": {
                        "do_sample": False,
                        "max_new_tokens": MAX_NEW_TOKENS,
                    },
                    "navi_sad_version": __version__,
                },
            )
            raw_writer.write(raw_record)

            elapsed = time.monotonic() - t0
            logger.info(
                "[%d/%d] idx=%d tokens=%d stop=%s scorer=%s (%.1fs) %s",
                i + 1,
                len(selected_indices),
                dataset_idx,
                generated_token_count,
                stop_reason,
                scorer_label,
                elapsed,
                row["question"][:60],
            )

    # ---------------------------------------------------------------
    # Write artifacts
    # ---------------------------------------------------------------
    samples_path = output_dir / "samples.json"
    review_path = output_dir / "review.json"

    samples_artifact = {"metadata": pilot_metadata, "samples": samples}
    with open(samples_path, "w", encoding="utf-8") as f:
        json.dump(samples_artifact, f, indent=2, ensure_ascii=False)
    logger.info("Wrote %s (%d samples)", samples_path, len(samples))

    with open(review_path, "w", encoding="utf-8") as f:
        json.dump(reviews, f, indent=2, ensure_ascii=False)
    logger.info("Wrote %s (%d samples)", review_path, len(reviews))

    logger.info("Wrote %s", raw_path)
    logger.info("Generation complete. Proceed to manual review protocol.")


# -------------------------------------------------------------------
# Analysis entry point
# -------------------------------------------------------------------


def run_analysis(args: argparse.Namespace) -> None:
    """Run post-review analysis on labeled pilot results."""
    review_path = Path(args.analyze)
    samples_path = review_path.parent / "samples.json"

    if not review_path.exists():
        sys.exit(f"Review file not found: {review_path}")
    if not samples_path.exists():
        sys.exit(f"Samples file not found: {samples_path}")

    with open(review_path, encoding="utf-8") as f:
        review_data: list[dict[str, Any]] = json.load(f)
    with open(samples_path, encoding="utf-8") as f:
        samples_artifact: dict[str, Any] = json.load(f)

    samples_data: list[dict[str, Any]] = samples_artifact["samples"]

    # Integrity validation (aborts on failure)
    print("=== Review Integrity Validation ===")
    try:
        validate_review_integrity(review_data, samples_data)
        print("PASS: all integrity checks passed\n")
    except ValueError as e:
        sys.exit(f"FAIL: {e}")

    # Build lookup
    samples_by_idx = {s["dataset_index"]: s for s in samples_data}

    # Collect labels
    human_labels = [r["human_label"] for r in review_data]
    scorer_labels = [r["scorer_label"] for r in review_data]

    # ---------------------------------------------------------------
    # Generation characterization
    # ---------------------------------------------------------------
    print("=== Generation Characterization ===")
    token_counts = [r["generated_token_count"] for r in review_data]
    if token_counts:
        sorted_counts = sorted(token_counts)
        n = len(sorted_counts)
        median = (
            sorted_counts[n // 2]
            if n % 2 == 1
            else (sorted_counts[n // 2 - 1] + sorted_counts[n // 2]) / 2
        )
        print(
            f"Token counts: min={min(token_counts)}, max={max(token_counts)}, "
            f"median={median}, mean={sum(token_counts) / len(token_counts):.1f}"
        )

    # Stop reasons
    stop_reasons = [samples_by_idx[r["dataset_index"]]["stop_reason"] for r in review_data]
    eos_count = stop_reasons.count("eos")
    max_len_count = stop_reasons.count("max_length")
    print(f"Stop reasons: eos={eos_count}, max_length={max_len_count}")

    # Class balance
    for label in ["correct", "incorrect", "ambiguous"]:
        count = human_labels.count(label)
        print(f"  {label}: {count}")

    # PE coverage
    pe_eligible = sum(1 for c in token_counts if c >= 8)
    print(f"PE coverage (>= 8 tokens): {pe_eligible}/{len(token_counts)}")
    print()

    # ---------------------------------------------------------------
    # Scorer assessment
    # ---------------------------------------------------------------
    print("=== Scorer Assessment ===")
    cm = compute_confusion_matrix(scorer_labels, human_labels)
    print(f"Overall agreement: {cm['overall_agreement']:.1%}")
    print()

    # Confusion matrix
    classes = ["correct", "incorrect", "ambiguous"]
    header = "scorer \\ human".ljust(20) + "".join(c.ljust(12) for c in classes)
    print(header)
    print("-" * len(header))
    for s_cls in classes:
        row_str = s_cls.ljust(20)
        for h_cls in classes:
            row_str += str(cm["matrix"][s_cls][h_cls]).ljust(12)
        print(row_str)
    print()

    # Per-class precision/recall
    for cls in classes:
        prec_val, prec_reason = cm["per_class"][cls]["precision"]
        rec_val, rec_reason = cm["per_class"][cls]["recall"]
        prec_str = f"{prec_val:.2%}" if prec_val is not None else f"null ({prec_reason})"
        rec_str = f"{rec_val:.2%}" if rec_val is not None else f"null ({rec_reason})"
        print(f"  {cls}: precision={prec_str}, recall={rec_str}")

    # Failure mode categorization
    print()
    disagreements = [
        r["disagreement_category"] for r in review_data if r.get("disagreement_category")
    ]
    if disagreements:
        print("Disagreement categories:")
        for cat, count in Counter(disagreements).most_common():
            print(f"  {cat}: {count}")
    else:
        print("No scorer disagreements.")
    print()

    # ---------------------------------------------------------------
    # Leading-span boundary
    # ---------------------------------------------------------------
    print("=== Leading-Span Boundary ===")
    fallback_count = sum(
        1
        for r in review_data
        if samples_by_idx[r["dataset_index"]].get("leading_span_fallback", False)
    )
    total = len(review_data)
    print(f"Fallback rate: {fallback_count}/{total} ({fallback_count / max(total, 1):.0%})")
    if fallback_count > total / 2:
        print("WARNING: fallback_rate > 50% -- leading-span scalar is unreliable")

    # Extraction stop reasons
    stop_reason_counts: dict[str, int] = {}
    for r in review_data:
        s = samples_by_idx[r["dataset_index"]]
        sr = s.get("scorer_leading_span_stop_reason", "unknown")
        stop_reason_counts[sr] = stop_reason_counts.get(sr, 0) + 1
    print(f"Extraction stop reasons: {stop_reason_counts}")
    print()

    # ---------------------------------------------------------------
    # Exploratory effect sizes
    # ---------------------------------------------------------------
    print("=== Exploratory Effect Sizes (not frozen) ===")

    correct_full: list[dict[str, Any]] = []
    incorrect_full: list[dict[str, Any]] = []
    correct_leading: list[dict[str, Any]] = []
    incorrect_leading: list[dict[str, Any]] = []

    for r in review_data:
        s = samples_by_idx[r["dataset_index"]]
        hl = r["human_label"]
        if hl == "correct":
            if s.get("full_gen_mean_delta") is not None:
                correct_full.append(s)
            if s.get("leading_span_mean_delta") is not None and not s.get(
                "leading_span_fallback", False
            ):
                correct_leading.append(s)
        elif hl == "incorrect":
            if s.get("full_gen_mean_delta") is not None:
                incorrect_full.append(s)
            if s.get("leading_span_mean_delta") is not None and not s.get(
                "leading_span_fallback", False
            ):
                incorrect_leading.append(s)

    print(f"Full-gen: {len(correct_full)} correct, {len(incorrect_full)} incorrect")
    print(
        f"Leading-span (non-fallback): {len(correct_leading)} correct, "
        f"{len(incorrect_leading)} incorrect"
    )

    _print_cohens_d_summary("Full-gen", correct_full, incorrect_full, "full_gen_mean_delta")
    _print_cohens_d_summary(
        "Leading-span", correct_leading, incorrect_leading, "leading_span_mean_delta"
    )
    print()


def _print_cohens_d_summary(
    label: str,
    correct_samples: list[dict[str, Any]],
    incorrect_samples: list[dict[str, Any]],
    matrix_key: str,
) -> None:
    """Print Cohen's d summary for a set of samples."""
    if not correct_samples or not incorrect_samples:
        print(f"\n{label} Cohen's d: insufficient data")
        return

    num_layers = len(correct_samples[0][matrix_key])
    num_heads = len(correct_samples[0][matrix_key][0])
    d_gt_05 = 0
    d_gt_03 = 0
    d_valid = 0
    d_total = num_layers * num_heads

    # Per-layer summary preserving (layer, head) structure
    layer_summaries: list[str] = []
    for layer in range(num_layers):
        layer_d_values: list[str] = []
        for head in range(num_heads):
            c_vals = [s[matrix_key][layer][head] for s in correct_samples]
            i_vals = [s[matrix_key][layer][head] for s in incorrect_samples]
            d_val, _ = compute_cohens_d(c_vals, i_vals)
            if d_val is not None:
                d_valid += 1
                if abs(d_val) > 0.5:
                    d_gt_05 += 1
                if abs(d_val) > 0.3:
                    d_gt_03 += 1
                layer_d_values.append(f"{d_val:+.3f}")
            else:
                layer_d_values.append("null")
        layer_summaries.append(f"  L{layer:02d}: [{', '.join(layer_d_values)}]")

    suffix = " (non-fallback only)" if "leading" in label.lower() else ""
    print(f"\n{label} Cohen's d ({d_valid}/{d_total} valid{suffix}):")
    print(f"  |d| > 0.5: {d_gt_05}")
    print(f"  |d| > 0.3: {d_gt_03}")
    # Print per-layer breakdown (first 5 + last 5 if many layers)
    if len(layer_summaries) <= 12:
        for line in layer_summaries:
            print(line)
    else:
        for line in layer_summaries[:5]:
            print(line)
        print(f"  ... ({len(layer_summaries) - 10} layers omitted)")
        for line in layer_summaries[-5:]:
            print(line)


# -------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------


def _get_transformers_version() -> str:
    import transformers

    return transformers.__version__


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gate 3 Pilot -- 40-sample TruthfulQA characterization"
    )
    parser.add_argument(
        "--analyze",
        type=str,
        default=None,
        help="Path to reviewed review.json for analysis",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/pilot_gate3",
        help="Output directory (default: results/pilot_gate3)",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=DEFAULT_SAMPLE_COUNT,
        help=f"Number of samples (default: {DEFAULT_SAMPLE_COUNT})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print config and first 3 questions, do not run inference",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.analyze:
        run_analysis(args)
    elif args.dry_run:
        _run_dry(args)
    else:
        run_generation(args)


def _run_dry(args: argparse.Namespace) -> None:
    """Dry run: print config and first 3 questions without GPU."""
    import datasets as ds

    dataset = ds.load_dataset("truthful_qa", "generation", split="validation")
    rng = random.Random(SEED)
    selected = sorted(rng.sample(range(len(dataset)), args.sample_count))

    print(f"Model: {MODEL_ID}")
    print(f"Revision: {REVISION}")
    print(f"Samples: {args.sample_count}")
    print(f"Output: {args.output_dir}")
    print(f"Seed: {SEED}")
    print(f"Selected indices: {selected}")
    print("\nFirst 3 questions:")
    for idx in selected[:3]:
        print(f"  [{idx}] {dataset[idx]['question']}")


if __name__ == "__main__":
    main()
