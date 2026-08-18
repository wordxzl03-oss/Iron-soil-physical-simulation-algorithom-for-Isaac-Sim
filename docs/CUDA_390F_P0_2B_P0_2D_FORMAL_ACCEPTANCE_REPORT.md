# CUDA 390F P0-2B + P0-2D formal acceptance

## Result

```text
EXECUTION_ENVIRONMENT_CLASS = SANDBOX_GPU_NOT_PASSTHROUGH
P0_2B_FORMAL_STATUS = ENVIRONMENT_BLOCKED
P0_2D_FORMAL_STATUS = ENVIRONMENT_BLOCKED
```

No production physics run occurred. These statuses are not physics failures
and are not formal acceptance passes.

## Environment evidence

- Hostname visible to the process: `eric-Stealth-GS77-12UGS`.
- Kernel: Ubuntu 22.04 series `6.8.0-136-generic`.
- Process scope: `app-code-22547.scope`.
- `nvidia-smi` cannot communicate with an NVIDIA driver from this process.
- No `/dev/nvidia*` device nodes are exposed.
- `CUDA_VISIBLE_DEVICES` is unset, so there is no explicit filtering mask.
- Warp 1.5.0 reports `CUDA devices not available` and enumerates only `cpu`.
- `vulkaninfo` is not installed in this execution environment.
- Isaac Sim installation version is `4.5.0-rc.36+release.19112.f59b3005.gl`.

The current process runs under the managed Codex application sandbox, so the
absence of GPU device nodes here does not establish a driver failure on the
real Ubuntu host. The classification is therefore
`SANDBOX_GPU_NOT_PASSTHROUGH`, not `HOST_DRIVER_FAILURE`.

Per the acceptance stop rule, Isaac startup and the formal production replay
were not attempted after this diagnosis. No driver, CUDA, Vulkan, system or
physics setting was changed.

## Command to run on the real GPU-enabled Isaac host

From the repository root:

```bash
cd /home/eric/Desktop/mesh

./run_390f_interactive.sh \
  --headless \
  --acceptance-cycles 1 \
  --realistic-cut-scoop \
  --mobile-v2-pre-dump-acceptance outputs/mobile_v2_production/p0_2b_p0_2d_pre_dump.npz \
  --cut-fill-payload-audit outputs/mobile_v2_production/p0_2b_p0_2d_production_timeseries.json \
  --mobile-v2-dual-cv-audit outputs/mobile_v2_production/p0_2b_cuda_formal_acceptance_raw.json
```

The command retains the current `GPU_RUNTIME`, `DEVICE`, 701 x 701,
`dx=0.05 m`, material, Mobile V2, FailureSurface V3, Tool-Mobile coupling and
accepted curl-scoop configuration.

## Artifact boundary

Created from measured environment evidence:

- `outputs/mobile_v2_production/cuda_environment_probe.json`
- `outputs/mobile_v2_production/p0_2b_cuda_formal_acceptance.json`
- `outputs/mobile_v2_production/p0_2d_cuda_formal_acceptance.json`

The following were deliberately not created because no physics run occurred:

- `p0_2b_p0_2d_production_timeseries.json`
- `p0_2b_p0_2d_pre_dump.npz`
- `p0_2b_p0_2d_pre_dump_morphology.png`

Null numerical fields in the two acceptance records mean `NOT_MEASURED`; they
are not synthetic replacements.
