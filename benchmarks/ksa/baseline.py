# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""T00 official Transformers baseline; no import of vLLM or model modifications."""

import argparse
import csv
import gc
import hashlib
import importlib.metadata as metadata
import json
import logging
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

OFFICIAL_REVISION = "9998565d0aa2907cc99e3ab0717b8bf246e77876"
HF_REVISION = "6b60859be46422fc5949a0e69d2a338c4a618c90"
SHORT = [1, 7, 8, 9, 15, 16, 17, 1023, 1024, 1025, 1031, 1032, 1033]
LONG = [4096, 16384, 32768, 65536, 130944]
PROMPT = "The archive records that the secret code is 73921. Remember this fact. "
GENERATION = {
    "do_sample": False,
    "use_cache": True,
    "eos_token_id": None,
    "pad_token_id": 151643,
    "max_new_tokens": 128,
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_hash(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def command(*args):
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr)
    return result.stdout.strip()


def install_transformers_compat():
    """Bridge KSA's factory-style decorator to newer Transformers releases."""
    from transformers.utils import generic

    original = generic.check_model_inputs
    if getattr(original, "_ksa_compat", False):
        return

    def compatible(func=None):
        if func is None:
            return lambda wrapped: original(wrapped)
        return original(func)

    compatible._ksa_compat = True
    generic.check_model_inputs = compatible


def make_inputs(tokenizer):
    unit = tokenizer.encode(PROMPT, add_special_tokens=False)
    cases = {}
    for length in SHORT + LONG:
        ids = (unit * ((length + 32) // len(unit) + 1))[: length + 24]
        cases[f"length-{length}"] = {
            "input_ids": ids[:length],
            "teacher_ids": ids[length:],
        }
    for name, prompt in [
        ("english", "The capital of France is"),
        ("chinese", "请简要介绍长上下文模型。"),
    ]:
        cases[name] = {
            "input_ids": tokenizer.encode(prompt, add_special_tokens=False),
            "teacher_ids": unit[:24],
        }
    # Deterministic retrieval input; the expected answer is recorded, not assumed.
    needle = tokenizer.encode("\nThe secret code is 73921.\n", add_special_tokens=False)
    filler = tokenizer.encode("An ordinary archive entry. ", add_special_tokens=False)
    question = tokenizer.encode(
        "\nWhat is the secret code? Answer:", add_special_tokens=False
    )
    for length in [4096, 32768, 65536]:
        count = length - len(needle) - len(question)
        body = (filler * (count // len(filler) + 1))[:count]
        split = count // 4
        cases[f"retrieval-{length}"] = {
            "input_ids": body[:split] + needle + body[split:] + question,
            "teacher_ids": unit[:24],
            "expected_answer": "73921",
        }
    return cases


def collect(args, torch, tokenizer):
    hashes = {
        p.name: file_hash(p)
        for p in sorted(args.model.iterdir())
        if p.is_file() and p.suffix in {".json", ".txt", ".py", ".safetensors"}
    }
    packages = dict(
        sorted((d.metadata["Name"], d.version) for d in metadata.distributions())
    )
    manifest_path = Path(__file__).resolve().parents[2] / "docs/ksa/model-manifest.json"
    expected_hashes = json.loads(manifest_path.read_text())["files"]
    inputs = make_inputs(tokenizer)
    lock = {
        "model_hashes": hashes,
        "packages": packages,
        "official_revision": OFFICIAL_REVISION,
        "hf_release_revision": HF_REVISION,
        "input_hash": digest(inputs),
        "generation": GENERATION,
        "profile": "official-flex-cu128",
        "generation_cache": "explicit_official_Qwen3RingBufferCache",
        "dtype": "bfloat16",
        "torch_cuda": torch.version.cuda,
        "harness_sha256": file_hash(Path(__file__)),
        "gpu": torch.cuda.get_device_name(),
        "seed": 0,
        "allow_tf32": False,
    }
    kernel_distribution = metadata.distribution("summary_attn")
    lock["kernel_hashes"] = {
        str(p): file_hash(Path(kernel_distribution.locate_file(p)))
        for p in kernel_distribution.files
        if str(p).endswith(".py")
    }
    baseline_id = "T00-" + digest(lock)[:20]
    environment = {
        "schema_version": 1,
        "ticket": "T00",
        "baseline_id": baseline_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_sha": command("git", "rev-parse", "HEAD"),
        "git_branch": command("git", "branch", "--show-current"),
        "git_dirty": bool(command("git", "status", "--porcelain")),
        "os": platform.platform(),
        "python": sys.version,
        "commands": [sys.argv],
        "reference_code_revision": OFFICIAL_REVISION,
        "gpu_name": torch.cuda.get_device_name(),
        "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "compute_capability": torch.cuda.get_device_capability(),
        "driver_version": command(
            "nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"
        ),
        "dtype": "bfloat16",
        "kv_dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "lock": lock,
        "tokenizer_hashes": {
            k: v
            for k, v in hashes.items()
            if "token" in k or k in {"vocab.json", "merges.txt"}
        },
        "model_hashes": hashes,
        "backends": {
            "transformers": {
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "transformers": metadata.version("transformers"),
                "summary_kernel": "summary_attn 0.3.0 / official FlexAttention",
            },
            "vllm": None,
        },
    }
    try:
        environment["cuda_toolkit"] = command("nvcc", "--version")
    except (OSError, RuntimeError):
        environment["cuda_toolkit"] = None
    write_json(args.output / "inputs.json", inputs)
    write_json(args.output / "environment.json", environment)
    write_json(args.output / "baseline-lock.json", lock)
    if hashes != expected_hashes:
        raise RuntimeError(
            "Model files differ from docs/ksa/model-manifest.json; "
            "review changes and establish a new baseline before running"
        )
    return environment, inputs


def metrics(torch, expected, actual):
    a, b = expected.float(), actual.float()
    if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
        return {
            "finite": False,
            "logits_max_abs_error": None,
            "logits_rmse": None,
            "logprobs_max_abs_error": None,
            "top1_agreement": None,
            "first_mismatch": None,
            "min_top1_margin": None,
        }
    delta = a - b
    top_a, top_b = a.argmax(-1), b.argmax(-1)
    mismatches = (top_a != top_b).nonzero()
    first_mismatch = None
    if len(mismatches):
        index = tuple(mismatches[0].tolist())
        first_mismatch = {
            "row_index": list(index),
            "expected_token": top_a[index].item(),
            "actual_token": top_b[index].item(),
            "top1_margin": a[index].topk(2).values.diff().abs().item(),
        }
    return {
        "logits_max_abs_error": delta.abs().max().item(),
        "logits_rmse": delta.square().mean().sqrt().item(),
        "logprobs_max_abs_error": (a.log_softmax(-1) - b.log_softmax(-1))
        .abs()
        .max()
        .item(),
        "top1_agreement": (top_a == top_b).float().mean().item(),
        "first_mismatch": first_mismatch,
        "min_top1_margin": (a.topk(2).values.diff(dim=-1).abs().min().item()),
        "finite": bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
    }


def semantics_probe(torch, args):
    """Analytic probe: zero Q/K, summary V=9 gives summary output exactly 1."""
    from summary_attn import summary_attn_func

    results = []
    for window in [128, 16768]:
        q = torch.zeros((1, 18, 32, 128), device="cuda", dtype=torch.bfloat16)
        k = torch.zeros((1, 18, 8, 128), device="cuda", dtype=torch.bfloat16)
        v = torch.zeros_like(k)
        v[:, [8, 17]] = 9
        summary = torch.arange(18, device="cuda") % 9 == 8
        output, _ = summary_attn_func(q, k, v, 8, 1, window, summary_pos=summary)
        expected = torch.zeros_like(q)
        expected[:, [8, 17]] = 1
        error = (output - expected).abs().max().item()
        results.append(
            {
                "window": window,
                "max_abs_error": error,
                "status": "pass" if error <= 0.01 else "fail",
            }
        )
    write_json(
        args.output / "semantics.json",
        {
            "probe": "summary sees self and 8 text rows; text excludes local summaries",
            "analytic_atol": 0.01,
            "cases": results,
        },
    )
    if any(r["status"] != "pass" for r in results):
        raise RuntimeError("Official kernel failed analytic summary visibility probe")


def teacher(torch, model, ids, continuation):
    output = model(input_ids=ids, use_cache=True, logits_to_keep=1)
    rows = [output.logits[0, -1].float().cpu()]
    cache = output.past_key_values
    for token in continuation:
        token = torch.tensor([[token]], device="cuda")
        prepared = model.prepare_inputs_for_generation(
            token, past_key_values=cache, use_cache=True
        )
        output = model(**prepared, logits_to_keep=1)
        cache = output.past_key_values
        rows.append(output.logits[0, -1].float().cpu())
    return torch.stack(rows)


def generate(model, ids):
    # HF 4.57.1 otherwise creates DynamicCache before the remote model sees None.
    remote = sys.modules[type(model).__module__]
    cache = remote.Qwen3RingBufferCache(model.config, model.model._sliding_chunk_nums)
    return model.generate(ids, past_key_values=cache, **GENERATION)


def correctness(torch, model, tokenizer, inputs, args, common):
    report = {
        **common,
        "schema_version": 1,
        "tolerance_id": None,
        "thresholds": None,
        "status": "awaiting_calibration_review",
        "cases": [],
    }
    for name, case in inputs.items():
        logging.info("Correctness %s", name)
        row = {
            "case_id": name,
            "input_tokens": len(case["input_ids"]),
            "output_tokens": len(case["teacher_ids"]),
            "backend_pair": "HF/HF",
            "mode": "teacher_forced",
            "status": "not_run",
            "reason": None,
            "logits_max_abs_error": None,
            "logits_rmse": None,
            "logprobs_max_abs_error": None,
            "top1_agreement": None,
            "first_mismatch": None,
        }
        try:
            ids = torch.tensor([case["input_ids"]], device="cuda")
            reference = teacher(torch, model, ids, case["teacher_ids"])
            repeats = [
                metrics(
                    torch, reference, teacher(torch, model, ids, case["teacher_ids"])
                )
                for _ in range(2)
            ]
            # Compare the same teacher sequence via full prefill, selected text rows.
            full = torch.tensor(
                [case["input_ids"] + case["teacher_ids"]], device="cuda"
            )
            positions = torch.arange(ids.shape[1] - 1, full.shape[1], device="cuda")
            output = model(input_ids=full, use_cache=False, logits_to_keep=positions)
            prefill = output.logits[0].float().cpu()
            del output
            comparison = metrics(torch, reference, prefill)
            row.update(comparison)
            row["repeat_metrics"] = repeats
            row["status"] = "not_run" if comparison["finite"] else "fail"
            row["reason"] = "Measurements collected; thresholds not yet frozen."
            torch.save(
                {
                    "decode_logits": reference,
                    "prefill_logits": prefill,
                    "text_positions": positions.cpu(),
                },
                args.output / "outputs" / f"{name}.pt",
            )
            generated = generate(model, ids)[0, ids.shape[1] :].cpu().tolist()
            generated_again = generate(model, ids)[0, ids.shape[1] :].cpu().tolist()
            row["generation_repeat_equal"] = generated == generated_again
            if generated != generated_again:
                row["generated_repeat_ids"] = generated_again
            row["generated_ids"] = generated
            row["generated_text"] = tokenizer.decode(generated)
            if "expected_answer" in case:
                row["retrieval_answer_present"] = (
                    case["expected_answer"] in row["generated_text"]
                )
        except Exception as exc:
            logging.exception("Correctness failed: %s", name)
            row.update(status="error", reason=f"{type(exc).__name__}: {exc}")
            row["output_tokens"] = None
        report["cases"].append(row)
        write_json(args.output / "correctness.json", report)
        gc.collect()
        torch.get_device_module("cuda").empty_cache()


def timed_generation(torch, model, ids):
    """Wall time includes host dispatch, argmax and official summary insertion."""
    torch.accelerator.synchronize()
    start = time.perf_counter()
    output = model(input_ids=ids, use_cache=True, logits_to_keep=1)
    token = output.logits[:, -1].argmax(-1, keepdim=True)
    cache = output.past_key_values
    torch.accelerator.synchronize()
    ttft = (time.perf_counter() - start) * 1000
    latencies, boundary = [], []
    for step in range(GENERATION["max_new_tokens"] - 1):
        start = time.perf_counter()
        prepared = model.prepare_inputs_for_generation(
            token, past_key_values=cache, use_cache=True
        )
        output = model(**prepared, logits_to_keep=1)
        token = output.logits[:, -1].argmax(-1, keepdim=True)
        cache = output.past_key_values
        torch.accelerator.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        latencies.append(elapsed)
        boundary.append((ids.shape[1] + step + 1) % 8 == 0)
    return {
        "ttft_ms": ttft,
        "tpot_steady_ms": statistics.mean(
            t for t, b in zip(latencies, boundary) if not b
        ),
        "tpot_boundary_ms": statistics.mean(
            t for t, b in zip(latencies, boundary) if b
        ),
        "output_tokens_per_s": 128000 / (ttft + sum(latencies)),
        "peak_memory_bytes": torch.get_device_module("cuda").max_memory_allocated(),
        "decode_ms": latencies,
        "boundary": boundary,
    }


def performance(torch, model, inputs, args, common, load_ms):
    columns = list(common) + [
        "case_id",
        "backend",
        "mode",
        "input_tokens",
        "output_tokens",
        "concurrency",
        "repetition",
        "status",
        "ttft_ms",
        "tpot_steady_ms",
        "tpot_boundary_ms",
        "output_tokens_per_s",
        "peak_memory_bytes",
        "active_kv_bytes",
        "allocated_kv_pool_bytes",
        "load_ms",
        "compile_capture_ms",
        "reason",
    ]
    with (args.output / "performance.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for length in LONG:
            logging.info("Performance length %s", length)
            ids = torch.tensor([inputs[f"length-{length}"]["input_ids"]], device="cuda")
            for rep in range(-1, 5):
                row = {
                    **common,
                    "case_id": f"length-{length}",
                    "backend": "transformers",
                    "mode": "offline",
                    "input_tokens": length,
                    "output_tokens": 128,
                    "concurrency": 1,
                    "repetition": rep,
                    "load_ms": load_ms,
                    "reason": "KV pages unavailable; rep=-1 includes cold compilation.",
                }
                try:
                    torch.get_device_module("cuda").reset_peak_memory_stats()
                    result = timed_generation(torch, model, ids)
                    write_json(
                        args.output / "outputs" / f"timing-{length}-{rep}.json", result
                    )
                    row.update({k: v for k, v in result.items() if k in columns})
                    row["status"] = "warmup" if rep < 0 else "pass"
                except Exception as exc:
                    logging.exception("Performance failed")
                    row.update(
                        status="oom"
                        if isinstance(exc, torch.cuda.OutOfMemoryError)
                        else "error",
                        reason=f"{type(exc).__name__}: {exc}",
                    )
                    row["output_tokens"] = None
                writer.writerow(row)
                stream.flush()
                gc.collect()
                torch.get_device_module("cuda").empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=["all", "correctness", "performance"], default="all"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "outputs").mkdir()
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(args.output / "run.log", encoding="utf-8"),
        ],
    )
    write_json(
        args.output / "environment.json",
        {
            "schema_version": 1,
            "ticket": "T00",
            "baseline_id": None,
            "os": platform.platform(),
            "python": sys.version,
            "commands": [sys.argv],
            "status": "initializing",
        },
    )
    write_json(
        args.output / "correctness.json",
        {"status": "not_run", "cases": [], "thresholds": None, "tolerance_id": None},
    )
    (args.output / "performance.csv").write_text("status,reason\nnot_run,not started\n")
    status = "BLOCKED"
    try:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        install_transformers_compat()

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is required; CPU results are not an official baseline"
            )
        torch.manual_seed(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        config = AutoConfig.from_pretrained(
            args.model, trust_remote_code=True, local_files_only=True
        )
        if not hasattr(config, "rope_parameters"):
            config.rope_parameters = {
                "rope_type": "default",
                "rope_theta": config.rope_theta,
            }
        env, inputs = collect(args, torch, tokenizer)
        import summary_attn.interface as kernel

        if kernel._check_cute_available():
            raise RuntimeError(
                "Official FlexAttention profile requires an isolated environment "
                "without flash-attn-cute"
            )
        with torch.inference_mode():
            semantics_probe(torch, args)
        torch.accelerator.synchronize()
        start = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            config=config,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
        ).eval()
        torch.accelerator.synchronize()
        load_ms = (time.perf_counter() - start) * 1000
        common = {
            "ticket": "T00",
            "git_sha": env["git_sha"],
            "baseline_id": env["baseline_id"],
        }
        with torch.inference_mode():
            if args.mode in {"all", "performance"}:
                performance(torch, model, inputs, args, common, load_ms)
            if args.mode in {"all", "correctness"}:
                correctness(torch, model, tokenizer, inputs, args, common)
        status = "AWAITING_SERVER_REVIEW"
    except Exception as exc:
        logging.exception("Baseline blocked")
        write_json(
            args.output / "failure.json",
            {"status": "blocked", "reason": f"{type(exc).__name__}: {exc}"},
        )
        raise
    finally:
        (args.output / "summary.md").write_text(
            f"# T00\n\nStatus: {status}\n\n"
            "No acceptance or frozen tolerances are implied by completing this run.\n"
            "Review repeat variation, prefill/decode errors and OOM rows.\n"
            "Concurrency 4/8, paged KV, chunked prefill, CUDA graphs: N/A.\n"
            "Previous ticket: N/A. Pinned official CuTe does not support sm120.\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
