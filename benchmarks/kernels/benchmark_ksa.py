# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KSA attention timing, separating metadata and reference gather overhead."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from vllm.model_executor.models.ksa_attention import (
    KSAAttentionMetadata,
    paged_attention,
)
from vllm.model_executor.models.ksa_decode import cached_layout, retained_layout
from vllm.model_executor.models.ksa_prefill import prefill_attention, visibility_mask


def measure(fn):
    for _ in range(3):
        fn()
    torch.accelerator.synchronize()
    result = []
    for _ in range(5):
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        for _ in range(10):
            fn()
        end.record()
        end.synchronize()
        result.append(start.elapsed_time(end) / 10)
    return result


@torch.inference_mode()
def run(length, batch, phase, window):
    torch.manual_seed(0)
    start, count = (0, length) if phase == "prefill" else (length - 1, 1)
    pos, _, summary = cached_layout(start, count, "cuda")
    old_pos, old_summary, _ = retained_layout(start, window, "cuda")
    n, m = len(old_pos), len(pos)
    pool = torch.randn(batch * (n + 8), 2, 8, 128, device="cuda", dtype=torch.bfloat16)
    selected = torch.randperm(len(pool), device="cuda")[: batch * n].view(batch, n)
    states = [
        dict(
            pos=pos,
            summary=summary,
            old_lengths={window: n},
            cache=SimpleNamespace(
                layouts={window: (old_pos, old_summary)},
                page_pool=SimpleNamespace(read_slots={"r": [selected[i]]}),
                request=SimpleNamespace(request_id="r"),
            ),
        )
        for i in range(batch)
    ]
    metadata = KSAAttentionMetadata.from_states([window], states)
    q = torch.randn(batch * m, 32, 128, device="cuda", dtype=torch.bfloat16)
    k, v = torch.randn(2, batch * m, 8, 128, device="cuda", dtype=torch.bfloat16)
    positions, flags = pos.repeat(batch), summary.repeat(batch)
    mask = visibility_mask(
        pos,
        summary,
        window,
        torch.cat((old_pos, pos)),
        torch.cat((old_summary, summary)),
    )

    def gather():
        old = pool[selected.flatten()].view(batch, n, 2, 8, 128)
        return [
            (
                torch.cat((old[i, :, 0], k[i * m : (i + 1) * m])),
                torch.cat((old[i, :, 1], v[i * m : (i + 1) * m])),
            )
            for i in range(batch)
        ]

    history = gather()

    def reference(history=history):
        return torch.cat(
            [
                prefill_attention(q[i * m : (i + 1) * m], key, value, mask)
                for i, (key, value) in enumerate(history)
            ]
        )

    def kernel():
        return paged_attention(q, k, v, pool, metadata, 0, positions, flags, m)

    torch.testing.assert_close(kernel(), reference(), atol=0.008, rtol=0.008)
    return dict(
        length=length,
        batch=batch,
        phase=phase,
        window=window,
        triton_ms=measure(kernel),
        reference_attention_ms=measure(reference),
        reference_gather_ms=measure(gather),
        reference_with_gather_ms=measure(lambda: reference(gather())),
        metadata_ms=measure(lambda: metadata.stage(states)),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for length in (1024, 4096):
        for batch in (1, 4, 8):
            for phase in ("prefill", "decode"):
                for window in (128, 16768):
                    row = run(length, batch, phase, window)
                    rows.append(row)
                    args.output.write_text(
                        json.dumps(
                            dict(
                                gpu=torch.cuda.get_device_name(),
                                torch=torch.__version__,
                                timings=rows,
                            ),
                            indent=2,
                        )
                        + "\n"
                    )
                    print(length, batch, phase, window, flush=True)


if __name__ == "__main__":
    main()
