# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded OpenAI completions service for the eager KSA batch runner.

Run with ``python -m vllm.entrypoints.ksa --model /path/to/KSA``. Unsupported
sampling and cache options are rejected rather than silently ignored.
"""

import argparse
import asyncio
import json
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, contextmanager, suppress
from functools import partial
from typing import TYPE_CHECKING, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, StrictInt

if TYPE_CHECKING:
    from vllm.v1.worker.ksa_model_runner import KSAOutput


class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str
    prompt: str | list[StrictInt]
    max_tokens: int = Field(default=16, ge=0, strict=True)
    temperature: Literal[0] = 0
    n: Literal[1] = 1
    stream: bool = False
    ignore_eos: bool = False
    prompt_logprobs: bool = False


@contextmanager
def load_model(model_path):
    from vllm.config import CompilationConfig, CompilationMode, set_current_vllm_config
    from vllm.distributed import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.engine.arg_utils import EngineArgs
    from vllm.model_executor.model_loader import get_model

    config = EngineArgs(
        model=model_path,
        enforce_eager=True,
        max_model_len=8192,
        dtype="bfloat16",
        enable_prefix_caching=False,
        compilation_config=CompilationConfig(
            mode=CompilationMode.NONE, custom_ops=["none"]
        ),
    ).create_engine_config()
    with tempfile.TemporaryDirectory() as temp, set_current_vllm_config(config):
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            backend="nccl",
            distributed_init_method=f"file://{temp}/store",
        )
        initialize_model_parallel(1, 1)
        try:
            model = get_model(vllm_config=config)
            from vllm.model_executor.models.ksa import KSAForCausalLM

            if not isinstance(model, KSAForCausalLM):
                raise ValueError("this entrypoint requires a KSA model")
            yield model
        finally:
            cleanup_dist_env_and_memory()


def create_app(runner, tokenizer, *, served_model_name="ksa", max_pending_requests=128):
    queues: dict[str, asyncio.Queue[KSAOutput | Exception]] = {}
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ksa")

    async def execute(function, *args, **kwargs):
        # A disconnected ASGI task may cancel its await while a previous GPU
        # step still runs. Keep queued cleanup alive on the execution thread.
        return await asyncio.shield(
            asyncio.get_running_loop().run_in_executor(
                executor, partial(function, *args, **kwargs)
            )
        )

    async def drive():
        while True:
            if queues:
                try:
                    outputs = await execute(runner.step)
                except Exception as exc:
                    # Notify every client rather than leave streams hanging.
                    for failed_queue in list(queues.values()):
                        if failed_queue.full():
                            failed_queue.get_nowait()
                        failed_queue.put_nowait(exc)
                    await execute(runner.close)
                else:
                    for output in outputs:
                        queue = queues.get(output.request_id)
                        if queue is not None:
                            if queue.full():
                                queue.get_nowait()
                            queue.put_nowait(output)
            await asyncio.sleep(0.001)

    @asynccontextmanager
    async def lifespan(app):
        task = asyncio.create_task(drive())
        try:
            yield
        finally:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
            await execute(runner.close)
            executor.shutdown(wait=True)

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health():
        def snapshot():
            return dict(
                status="ok",
                active_requests=len(runner.requests),
                cached_requests=len(runner.pool.read_slots),
                free_pages=runner.pool.manager.block_pool.get_num_free_blocks(),
                total_pages=runner.pool.config.num_blocks - 1,
            )

        return await execute(snapshot)

    @app.get("/v1/models")
    async def models():
        return {
            "object": "list",
            "data": [
                {
                    "id": served_model_name,
                    "object": "model",
                    "created": 0,
                    "owned_by": "vllm",
                }
            ],
        }

    @app.post("/v1/completions")
    async def completions(body: CompletionRequest, http_request: Request):
        if body.model != served_model_name:
            raise HTTPException(404, "unknown model")
        if len(queues) >= max_pending_requests:
            raise HTTPException(429, "KSA request queue is full")
        ids = (
            tokenizer.encode(body.prompt, add_special_tokens=False)
            if isinstance(body.prompt, str)
            else body.prompt
        )
        request_id = "cmpl-" + uuid.uuid4().hex
        queue: asyncio.Queue[KSAOutput | Exception] = asyncio.Queue(maxsize=1)
        queues[request_id] = queue
        try:
            await execute(
                runner.add_request,
                request_id,
                ids,
                max_tokens=body.max_tokens,
                eos_token_ids=(
                    () if tokenizer.eos_token_id is None else (tokenizer.eos_token_id,)
                ),
                ignore_eos=body.ignore_eos,
                prompt_logprobs=body.prompt_logprobs,
            )
        except ValueError as exc:
            queues.pop(request_id, None)
            raise HTTPException(400, str(exc)) from exc
        except BaseException:
            queues.pop(request_id, None)
            await execute(runner.abort_request, request_id)
            raise
        created = int(time.time())

        async def results():
            try:
                while True:
                    if await http_request.is_disconnected():
                        break
                    try:
                        output = await asyncio.wait_for(queue.get(), timeout=0.1)
                    except TimeoutError:
                        continue
                    if isinstance(output, Exception):
                        raise RuntimeError(str(output)) from output
                    if output.error:
                        raise RuntimeError(output.error)
                    yield output
                    if output.finish_reason is not None:
                        break
            finally:
                queues.pop(request_id, None)
                await execute(runner.abort_request, request_id)

        def response(output, text):
            result = dict(
                id=request_id,
                object="text_completion",
                created=created,
                model=served_model_name,
                choices=[
                    dict(
                        index=0,
                        text=text,
                        logprobs=None,
                        finish_reason=output.finish_reason,
                    )
                ],
            )
            if output.finish_reason is not None:
                result["usage"] = output.usage
                if body.prompt_logprobs:
                    result["prompt_logprobs"] = output.prompt_logprobs
            return result

        async def stream():
            emitted = ""
            iterator = results()
            try:
                async for output in iterator:
                    text = tokenizer.decode(
                        output.token_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    if output.finish_reason is None and text.endswith("\ufffd"):
                        continue
                    delta = text[len(emitted) :]
                    emitted = text
                    yield "data: " + json.dumps(response(output, delta)) + "\n\n"
            except Exception as exc:
                yield (
                    "data: "
                    + json.dumps(
                        {"error": {"message": str(exc), "type": "server_error"}}
                    )
                    + "\n\n"
                )
            finally:
                try:
                    await iterator.aclose()
                finally:
                    queues.pop(request_id, None)
                    await execute(runner.abort_request, request_id)
            yield "data: [DONE]\n\n"

        if body.stream:
            return StreamingResponse(stream(), media_type="text/event-stream")
        iterator = results()
        try:
            async for output in iterator:
                if output.finish_reason is not None:
                    text = tokenizer.decode(
                        output.token_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    return response(output, text)
        except RuntimeError as exc:
            raise HTTPException(503, str(exc)) from exc
        finally:
            await iterator.aclose()
        raise HTTPException(499, "client disconnected")

    return app


def main():
    import uvicorn

    from vllm.tokenizers import get_tokenizer
    from vllm.v1.worker.ksa_model_runner import KSABatchedRunner

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--served-model-name", default="ksa")
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--kv-cache-num-blocks", type=int)
    args = parser.parse_args()
    with load_model(args.model) as model:
        runner = KSABatchedRunner(
            model,
            max_num_seqs=args.max_num_seqs,
            max_num_batched_tokens=args.max_num_batched_tokens,
            chunk_size=args.chunk_size,
            kv_cache_num_blocks=args.kv_cache_num_blocks,
        )
        app = create_app(
            runner, get_tokenizer(args.model), served_model_name=args.served_model_name
        )
        uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
