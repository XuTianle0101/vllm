# T01

Status: failed

Traceback (most recent call last):
  File "/workspace/volume/h20-data/xutianle/KSA/vllm/benchmarks/ksa/prefill.py", line 442, in main
    run(args)
  File "/workspace/volume/h20-data/xutianle/KSA/vllm/benchmarks/ksa/prefill.py", line 261, in run
    result = compare(
             ^^^^^^^^
  File "/workspace/volume/h20-data/xutianle/KSA/vllm/benchmarks/ksa/prefill.py", line 91, in compare
    result = metrics(torch, expected, actual)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/volume/h20-data/xutianle/KSA/vllm/benchmarks/ksa/baseline.py", line 209, in metrics
    delta = a - b
            ~~^~~
RuntimeError: The size of tensor a (151936) must match the size of tensor b (151937) at non-singleton dimension 1
