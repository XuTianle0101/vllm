# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise a running KSA HTTP service: concurrency, SSE and disconnect cleanup."""

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx
from baseline import digest
from prefill import BASELINE_ID, TOLERANCE_ID


async def run(args):
    baseline = json.loads((args.baseline / "correctness.json").read_text())
    lock = json.loads((args.baseline / "baseline-lock.json").read_text())
    calibration = json.loads((args.baseline / "calibration.json").read_text())
    if (
        baseline["baseline_id"] != BASELINE_ID
        or baseline["tolerance_id"] != TOLERANCE_ID
        or "T00-" + digest(lock)[:20] != BASELINE_ID
        or "T00-tol-" + digest(calibration)[:20] != TOLERANCE_ID
    ):
        raise ValueError("frozen baseline identity mismatch")
    report = {
        "url": args.url,
        "status": "running",
        "concurrency": [],
        "baseline_id": baseline["baseline_id"],
        "tolerance_id": baseline["tolerance_id"],
    }
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        async with httpx.AsyncClient(base_url=args.url, timeout=300) as client:
            body = dict(
                model="ksa",
                prompt="Hello world. " * 40,
                max_tokens=16,
                temperature=0,
                ignore_eos=True,
                prompt_logprobs=True,
            )
            response = await client.post("/v1/completions", json=body)
            response.raise_for_status()
            expected = response.json()
            assert expected["usage"]["completion_tokens"] == 16
            assert (
                len(expected["prompt_logprobs"]) == expected["usage"]["prompt_tokens"]
            )
            for concurrency in (1, 4, 8):
                start = time.perf_counter()
                responses = await asyncio.gather(
                    *[
                        client.post("/v1/completions", json=body)
                        for _ in range(concurrency)
                    ]
                )
                elapsed = time.perf_counter() - start
                max_error = 0.0
                for response in responses:
                    response.raise_for_status()
                    data = response.json()
                    assert data["choices"] == expected["choices"]
                    assert data["usage"] == expected["usage"]
                    error = max(
                        abs(a - b)
                        for a, b in zip(
                            data["prompt_logprobs"][1:], expected["prompt_logprobs"][1:]
                        )
                    )
                    assert error <= baseline["thresholds"]["logprobs_max_abs_error"]
                    max_error = max(max_error, error)
                report["concurrency"].append(
                    dict(
                        concurrency=concurrency,
                        prompt_logprobs_max_abs_error=max_error,
                        elapsed_s=elapsed,
                        output_tokens_per_s=16 * concurrency / elapsed,
                    )
                )
            chunks = []
            async with client.stream(
                "POST", "/v1/completions", json={**body, "stream": True}
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line == "data: [DONE]":
                        break
                    if line.startswith("data: "):
                        chunks.append(json.loads(line[6:]))
                else:
                    raise AssertionError("SSE stream did not terminate with [DONE]")
            assert (
                "".join(c["choices"][0]["text"] for c in chunks)
                == expected["choices"][0]["text"]
            )
            assert chunks[-1]["usage"] == expected["usage"]
            assert chunks[-1]["choices"][0]["finish_reason"] == "length"
            report["stream"] = "pass"
            for cancel_after_token in (False, True, True, True):
                async with client.stream(
                    "POST",
                    "/v1/completions",
                    json={**body, "stream": True, "max_tokens": 1024},
                ) as response:
                    response.raise_for_status()
                    if cancel_after_token:
                        async for line in response.aiter_lines():
                            if line.startswith("data: "):
                                break
                for _ in range(100):
                    health = (await client.get("/health")).json()
                    if health["active_requests"] == 0:
                        break
                    await asyncio.sleep(0.05)
                assert health["active_requests"] == health["cached_requests"] == 0
                assert health["free_pages"] == health["total_pages"]
            report.update(disconnect="pass", reclaimed=health)
            for extra in (
                {"temperature": 1},
                {"enable_prefix_caching": True},
                {"n": 2},
            ):
                response = await client.post("/v1/completions", json={**body, **extra})
                assert response.status_code == 422
            report["status"] = "pass"
    except BaseException as exc:
        report.update(status="fail", error=repr(exc))
        raise
    finally:
        (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    asyncio.run(run(parser.parse_args()))
