# Hardware

All TFLOPS are **dense** tensor-core figures. Vendor datasheets headline the 2:1
structured-sparsity number (2x dense); a unit test asserts no entry here exceeds its dense
ceiling, because quoting the sparse figure silently doubles every throughput estimate.

`vramBytes` is nominal capacity; the driver keeps a slice of it — see
`driver_reserved_vram_fraction` in [ASSUMPTIONS.md](./ASSUMPTIONS.md).

| id | VRAM | Memory | Bandwidth | bf16 | fp8 | Interconnect | Source |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `h100-sxm-80` | 80 GiB | HBM3 | 3.35 TB/s | 989.4 | 1978.9 | nvlink4 @ 900 GB/s | [datasheet](https://resources.nvidia.com/en-us-tensor-core/nvidia-tensor-core-gpu-datasheet) |
| `h100-pcie-80` | 80 GiB | HBM2e | 2.00 TB/s | 756.5 | 1513 | nvlink-bridge @ 600 GB/s | [datasheet](https://resources.nvidia.com/en-us-tensor-core/nvidia-tensor-core-gpu-datasheet) |
| `h200-sxm-141` | 141 GiB | HBM3e | 4.80 TB/s | 989.4 | 1978.9 | nvlink4 @ 900 GB/s | [datasheet](https://resources.nvidia.com/en-us-data-center-overview/hpc-datasheet-sc23-h200) |
| `b200-sxm-192` | 192 GiB | HBM3e | 8.00 TB/s | 2250 | 4500 | nvlink5 @ 1800 GB/s | [datasheet](https://resources.nvidia.com/en-us-blackwell-architecture/datasheet) |
| `a100-sxm-80` | 80 GiB | HBM2e | 2.04 TB/s | 312 | — | nvlink3 @ 600 GB/s | [datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf) |
| `a100-pcie-80` | 80 GiB | HBM2e | 1.94 TB/s | 312 | — | nvlink-bridge @ 600 GB/s | [datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf) |
| `a100-sxm-40` | 40 GiB | HBM2 | 1.55 TB/s | 312 | — | nvlink3 @ 600 GB/s | [datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf) |
| `l40s-48` | 48 GiB | GDDR6 | 0.86 TB/s | 362.05 | 733 | pcie5 @ 64 GB/s | [datasheet](https://resources.nvidia.com/en-us-l40s/l40s-datasheet-28413) |
| `l4-24` | 24 GiB | GDDR6 | 0.30 TB/s | 121 | 242 | pcie4 @ 32 GB/s | [datasheet](https://resources.nvidia.com/en-us-l4/l4-datasheet) |
| `a10g-24` | 24 GiB | GDDR6 | 0.60 TB/s | 125 | — | pcie4 @ 32 GB/s | [datasheet](https://d1.awsstatic.com/product-marketing/ec2/NVIDIA_AWS_A10G_DataSheet_FINAL_02_17_2022.pdf) |
| `rtx4090-24` | 24 GiB | GDDR6X | 1.01 TB/s | 165.2 | 660.6 | pcie4 @ 32 GB/s | [datasheet](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf) |
| `mi300x-192` | 192 GiB | HBM3 | 5.30 TB/s | 1307.4 | 2614.9 | infinity-fabric @ 896 GB/s | [datasheet](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-data-sheet.pdf) |
| `m5-ultra-512` | 512 GiB | unified | 1.20 TB/s | 23.2 | — | none @ 0 GB/s | [datasheet](https://www.apple.com/newsroom/2026/08/apple-introduces-new-mac-studio-with-m5-max-and-m5-ultra/) |
| `m5-max-128` | 128 GiB | unified | 0.61 TB/s | 11.6 | — | none @ 0 GB/s | [datasheet](https://www.apple.com/newsroom/2026/08/apple-introduces-new-mac-studio-with-m5-max-and-m5-ultra/) |
| `m4-max-128` | 128 GiB | unified | 0.55 TB/s | 11.6 | — | none @ 0 GB/s | [datasheet](https://www.apple.com/newsroom/2024/10/apple-introduces-m4-pro-and-m4-max/) |
| `m4-pro-64` | 64 GiB | unified | 0.27 TB/s | 5.8 | — | none @ 0 GB/s | [datasheet](https://www.apple.com/newsroom/2024/10/apple-introduces-m4-pro-and-m4-max/) |
| `m3-ultra-512` | 512 GiB | unified | 0.82 TB/s | 19.76 | — | none @ 0 GB/s | [datasheet](https://www.apple.com/newsroom/2025/03/apple-reveals-m3-ultra-taking-apple-silicon-to-a-new-extreme/) |
| `m2-ultra-192` | 192 GiB | unified | 0.80 TB/s | 17.02 | — | none @ 0 GB/s | [datasheet](https://www.apple.com/newsroom/2023/06/apple-introduces-m2-ultra/) |

**rtx4090-24:** fp16/bf16 figure is with FP32 accumulate, which is what inference kernels use. FP16-accumulate peaks at 330.3 dense but is not the serving path.

**m5-ultra-512:** 80-core GPU. arXiv:2502.05317 measured peak FP32 via Metal Performance Shaders SGEMM on the base chip (M2 2.24, M3 2.47, M4 2.90 TFLOPS at 10 GPU cores), scaled by Apple's published core count for this configuration. https://arxiv.org/abs/2502.05317 fp16/bf16 are set equal to fp32 rather than assuming Apple's half-rate multiplier, so prefill here is a floor, not a peak. Decode is bandwidth-bound and uses Apple's own published figure. Apple GPUs address unified memory, but Metal caps a process at recommendedMaxWorkingSetSize, ~75% of RAM, which llama.cpp logs and treats as a hard ceiling; raise it with sysctl iogpu.wired_limit_mb. https://developer.apple.com/forums/thread/732035 Tensor parallelism across Apple devices is not a thing; the link is zeroed so any tp>1 shows as unserviceable rather than plausible. M5 is outside the paper's range, so the M4 per-core figure is used; M5 adds per-core neural accelerators, which makes this a conservative floor.

**m5-max-128:** 40-core GPU. arXiv:2502.05317 measured peak FP32 via Metal Performance Shaders SGEMM on the base chip (M2 2.24, M3 2.47, M4 2.90 TFLOPS at 10 GPU cores), scaled by Apple's published core count for this configuration. https://arxiv.org/abs/2502.05317 fp16/bf16 are set equal to fp32 rather than assuming Apple's half-rate multiplier, so prefill here is a floor, not a peak. Decode is bandwidth-bound and uses Apple's own published figure. Apple GPUs address unified memory, but Metal caps a process at recommendedMaxWorkingSetSize, ~75% of RAM, which llama.cpp logs and treats as a hard ceiling; raise it with sysctl iogpu.wired_limit_mb. https://developer.apple.com/forums/thread/732035 Tensor parallelism across Apple devices is not a thing; the link is zeroed so any tp>1 shows as unserviceable rather than plausible. M5 is outside the paper's range, so the M4 per-core figure is used; M5 adds per-core neural accelerators, which makes this a conservative floor.

**m4-max-128:** 40-core GPU. arXiv:2502.05317 measured peak FP32 via Metal Performance Shaders SGEMM on the base chip (M2 2.24, M3 2.47, M4 2.90 TFLOPS at 10 GPU cores), scaled by Apple's published core count for this configuration. https://arxiv.org/abs/2502.05317 fp16/bf16 are set equal to fp32 rather than assuming Apple's half-rate multiplier, so prefill here is a floor, not a peak. Decode is bandwidth-bound and uses Apple's own published figure. Apple GPUs address unified memory, but Metal caps a process at recommendedMaxWorkingSetSize, ~75% of RAM, which llama.cpp logs and treats as a hard ceiling; raise it with sysctl iogpu.wired_limit_mb. https://developer.apple.com/forums/thread/732035 Tensor parallelism across Apple devices is not a thing; the link is zeroed so any tp>1 shows as unserviceable rather than plausible.

**m4-pro-64:** 20-core GPU. arXiv:2502.05317 measured peak FP32 via Metal Performance Shaders SGEMM on the base chip (M2 2.24, M3 2.47, M4 2.90 TFLOPS at 10 GPU cores), scaled by Apple's published core count for this configuration. https://arxiv.org/abs/2502.05317 fp16/bf16 are set equal to fp32 rather than assuming Apple's half-rate multiplier, so prefill here is a floor, not a peak. Decode is bandwidth-bound and uses Apple's own published figure. Apple GPUs address unified memory, but Metal caps a process at recommendedMaxWorkingSetSize, ~75% of RAM, which llama.cpp logs and treats as a hard ceiling; raise it with sysctl iogpu.wired_limit_mb. https://developer.apple.com/forums/thread/732035 Tensor parallelism across Apple devices is not a thing; the link is zeroed so any tp>1 shows as unserviceable rather than plausible.

**m3-ultra-512:** 80-core GPU. arXiv:2502.05317 measured peak FP32 via Metal Performance Shaders SGEMM on the base chip (M2 2.24, M3 2.47, M4 2.90 TFLOPS at 10 GPU cores), scaled by Apple's published core count for this configuration. https://arxiv.org/abs/2502.05317 fp16/bf16 are set equal to fp32 rather than assuming Apple's half-rate multiplier, so prefill here is a floor, not a peak. Decode is bandwidth-bound and uses Apple's own published figure. Apple GPUs address unified memory, but Metal caps a process at recommendedMaxWorkingSetSize, ~75% of RAM, which llama.cpp logs and treats as a hard ceiling; raise it with sysctl iogpu.wired_limit_mb. https://developer.apple.com/forums/thread/732035 Tensor parallelism across Apple devices is not a thing; the link is zeroed so any tp>1 shows as unserviceable rather than plausible.

**m2-ultra-192:** 76-core GPU. arXiv:2502.05317 measured peak FP32 via Metal Performance Shaders SGEMM on the base chip (M2 2.24, M3 2.47, M4 2.90 TFLOPS at 10 GPU cores), scaled by Apple's published core count for this configuration. https://arxiv.org/abs/2502.05317 fp16/bf16 are set equal to fp32 rather than assuming Apple's half-rate multiplier, so prefill here is a floor, not a peak. Decode is bandwidth-bound and uses Apple's own published figure. Apple GPUs address unified memory, but Metal caps a process at recommendedMaxWorkingSetSize, ~75% of RAM, which llama.cpp logs and treats as a hard ceiling; raise it with sysctl iogpu.wired_limit_mb. https://developer.apple.com/forums/thread/732035 Tensor parallelism across Apple devices is not a thing; the link is zeroed so any tp>1 shows as unserviceable rather than plausible.

Adding a GPU: append to `data/gpus.json` with a `source_url`, add its known **sparse** bf16
and fp8 figures to the table in `packages/core/test/data.test.ts`, and run `pnpm gen`.

Generated by `pnpm gen` from `data/gpus.json`. Do not edit by hand.
