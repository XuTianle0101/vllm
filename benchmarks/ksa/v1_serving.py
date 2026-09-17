# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise standard vllm serve: completions, SSE, sampling and cancellation."""

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx


async def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"url": args.url, "status": "running", "concurrency": []}
    try:
        async with httpx.AsyncClient(base_url=args.url, timeout=300) as client:
            (await client.get("/health")).raise_for_status()
            models = (await client.get("/v1/models")).json()
            assert args.model in [m["id"] for m in models["data"]]
            body = dict(
                model=args.model,
                prompt=[42] * 65,
                max_tokens=16,
                temperature=0,
                ignore_eos=True,
                logprobs=3,
            )

            async def complete(payload):
                response = await client.post("/v1/completions", json=payload)
                response.raise_for_status()
                return response.json()

            expected = await complete(body)
            assert [
                expected["usage"][k]
                for k in ("prompt_tokens", "completion_tokens", "total_tokens")
            ] == [65, 16, 81]
            assert expected["choices"][0]["finish_reason"] == "length"
            for concurrency in (1, 4, 8):
                start = time.perf_counter()
                responses = await asyncio.gather(
                    *[complete(body) for _ in range(concurrency)]
                )
                divergences = []
                for data in responses:
                    assert data["usage"] == expected["usage"]
                    ref = expected["choices"][0]["logprobs"]
                    actual = data["choices"][0]["logprobs"]
                    for i, (a, b) in enumerate(zip(ref["tokens"], actual["tokens"])):
                        if a != b:
                            top = ref["top_logprobs"][i]
                            margin = top[a] - top.get(b, -float("inf"))
                            # Frozen T00 near-tie allowance, at the first
                            # divergence where both histories still agree.
                            assert margin <= 0.5
                            divergences.append(dict(position=i, margin=margin))
                            break
                report["concurrency"].append(
                    dict(
                        requests=concurrency,
                        elapsed_s=time.perf_counter() - start,
                        near_tie_divergences=divergences,
                    )
                )

            stream_body = dict(
                body, stream=True, stream_options={"include_usage": True}
            )
            pieces, usage, finished, done = [], None, False, False
            async with client.stream(
                "POST", "/v1/completions", json=stream_body
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    if line == "data: [DONE]":
                        done = True
                        break
                    data = json.loads(line[6:])
                    if data.get("usage"):
                        usage = data["usage"]
                    for choice in data["choices"]:
                        pieces.append(choice["text"])
                        finished |= choice["finish_reason"] == "length"
            assert done and finished and usage is not None
            assert all(
                usage[k] == expected["usage"][k]
                for k in ("prompt_tokens", "completion_tokens", "total_tokens")
            )
            assert "".join(pieces) == expected["choices"][0]["text"]
            report["stream"] = "pass"

            params = dict(
                body,
                temperature=0.8,
                top_p=0.9,
                seed=17,
                n=2,
                presence_penalty=0.2,
                repetition_penalty=1.05,
            )
            sampled = await complete(params)
            repeated = await complete(params)
            assert [c["text"] for c in sampled["choices"]] == [
                c["text"] for c in repeated["choices"]
            ]
            assert sampled["usage"]["completion_tokens"] == 32
            report["seeded_sampling_n2_penalties"] = "pass"

            echoed = await complete(
                dict(body, max_tokens=1, echo=True, prompt_logprobs=2)
            )
            assert len(echoed["choices"][0]["logprobs"]["tokens"]) == 66
            stop = expected["choices"][0]["text"][:1]
            stopped = await complete(dict(body, stop=[stop]))
            assert stopped["choices"][0]["finish_reason"] == "stop"
            assert stop not in stopped["choices"][0]["text"]
            report["echo_prompt_logprobs_and_stop"] = "pass"

            for ids in ([151936], [-1]):
                response = await client.post(
                    "/v1/completions", json=dict(body, prompt=ids)
                )
                assert response.status_code == 400
            for _ in range(3):
                async with client.stream(
                    "POST",
                    "/v1/completions",
                    json=dict(
                        body,
                        stream=True,
                        max_tokens=1024,
                    ),
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            break
            for _ in range(100):
                metrics = (await client.get("/metrics")).text
                active = sum(
                    float(line.rsplit(" ", 1)[-1])
                    for line in metrics.splitlines()
                    if line.startswith(
                        (
                            "vllm:num_requests_running{",
                            "vllm:num_requests_waiting{",
                            "vllm:kv_cache_usage_perc{",
                        )
                    )
                )
                if active == 0:
                    break
                await asyncio.sleep(0.1)
            assert active == 0, "cancelled requests retained scheduler pages"
            report["disconnects_and_page_reclamation"] = "pass"
            recovered = await complete(body)
            assert recovered["choices"][0]["text"] == expected["choices"][0]["text"]
            report["status"] = "pass"
    finally:
        (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18005")
    parser.add_argument("--model", default="ksa")
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(run(parser.parse_args()))
