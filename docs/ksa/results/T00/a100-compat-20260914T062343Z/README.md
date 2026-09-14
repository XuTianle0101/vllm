# T00 A100 兼容运行结果

- 运行目录：`/tmp/ksa-t00-a100-compat-run`
- 提交：`df7cfd4ae24debabb7b84594566d69e9fbc34e7d`
- GPU：NVIDIA A100-SXM4-80GB
- Transformers：4.57.1，使用 baseline 内置兼容层
- 状态：BLOCKED
- 说明：模型加载、summary 语义探针及 4K/16K/32K/64K 性能重复已完成；130944 token 场景发生 CUDA illegal memory access。该结果不构成完整 T00 精度/性能验收。
- 原始大文件：本次未生成 `.pt` logits 文件；结果目录保留结构化 JSON、CSV 和日志。
