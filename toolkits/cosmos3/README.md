# Cosmos3 Tools

Use [`eval_cosmos3_closed_loop.py`](../../examples/embodiment/eval_cosmos3_closed_loop.py) for SGLang inference and [`train_embodied_agent.py`](../../examples/embodiment/train_embodied_agent.py) for native WoVR training. Both entries use `cosmos.share_weights` to select shared or independent policy and simulator weights.

`SGLangEmbodiedWorker` generates native normalized 10D actions. `EnvWorker` runs `Cosmos3Env`, which sends those exact actions to the simulator's SGLang server for forward dynamics. `EmbodiedEvalRunner` feeds the final generated image back to the policy. This inference-only evaluation follows the existing rollout model configuration. The native training workflow is described below.

Both evaluation roles use `InferenceHTTPClient` for HTTP transport. `Cosmos3Client` requests decoded tensors from `/rollout/generate`; feedback uses raw frames without MP4 conversion. Each worker owns its client instance. Generation has no automatic retries.

The default uses one environment, one rollout stage, and one GPU. It starts from LIBERO-Spatial episode 1261, frame 0, then runs three chunks of 16 actions. Only the initial external/wrist images and instruction come from the dataset. Useful behavior depends on the checkpoint; the base Nano checkpoint is used first to exercise the pipeline.

From the RLinf checkout, prepare a virtual environment within the current workspace:

```bash
uv venv ../.venv --python 3.11
SGLANG_BUILD_RUST_EXTS=none uv --no-config pip install --python ../.venv/bin/python \
  -r examples/embodiment/requirements-cosmos3.txt
```

The requirements pin SGLang 0.5.19 at revision `0bcd822377da7b5718e674eaf9c870d349424dd1` and Torch 2.13.0. `--no-config` prevents RLinf's general dependency overrides from changing this runtime. Reuse the workspace environment if already created.

For a CUDA 13 build on an NVIDIA data center GPU with driver 570, use [`run_with_cuda_compat.sh`](../../examples/embodiment/run_with_cuda_compat.sh) to load project-local CUDA compatibility libraries. On Ubuntu 22.04 x86-64, prepare the verified package from [NVIDIA's repository](https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/) from the RLinf checkout:

```bash
mkdir -p ../.cuda-compat/13.0
curl -fL \
  https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-compat-13-0_580.196.07-1ubuntu1_amd64.deb \
  -o ../.cuda-compat/13.0/cuda-compat-13-0.deb
printf '%s  %s\n' \
  b0ee4fb1f439e919ca5c5fd7c5cff3e8b8afdbdb0b5ee38f7c09c047ea41a035 \
  ../.cuda-compat/13.0/cuda-compat-13-0.deb | sha256sum -c -
dpkg-deb -x ../.cuda-compat/13.0/cuda-compat-13-0.deb ../.cuda-compat/13.0
```

This checks the package checksum and extracts the libraries into the workspace. Reuse this directory if already prepared. The wrapper defaults to `../.cuda-compat/13.0/usr/local/cuda-13.0/compat`, resolved relative to the checkout. Set `COSMOS3_CUDA_COMPAT_DIR` to the exact library directory for another layout. It prepends that directory to `LD_LIBRARY_PATH` for its command and child processes on driver 570; on drivers >=580 it preserves the existing library path. Other driver branches are rejected. See NVIDIA's [forward compatibility requirements](https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html) for supported hardware and package combinations.

Configure checkpoint, dataset, and output paths in the YAML. Select an available physical GPU through `cluster.component_placement`; the default `env,rollout: 0` uses GPU 0. For GPU 5, set `env,rollout: 5`. RLinf discovers physical devices through NVML, so `CUDA_VISIBLE_DEVICES` alone does not select worker placement. Run a local single-node smoke test with prepared local assets:

```bash
env -u http_proxy -u https_proxy -u all_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  RAY_ADDRESS=local PYTHONPATH="$PWD" HF_HUB_OFFLINE=1 \
  XDG_CACHE_HOME="$PWD/../.cache" \
  CUDA_CACHE_PATH="$PWD/../.cache/cuda" TRITON_CACHE_DIR="$PWD/../.cache/triton" \
  bash examples/embodiment/run_with_cuda_compat.sh ../.venv/bin/python \
  examples/embodiment/eval_cosmos3_closed_loop.py \
  runner.logger.log_path=/shared/cosmos3/run-001
```

`RAY_ADDRESS=local` creates an independent Ray session for this single-node test. Removing proxy variables lets the server health check connect directly. These settings belong to the launch command; the CUDA wrapper only configures library loading. For Ray temporary data on shared storage, set `RAY_TMPDIR` to a short path or short symlink to that storage, because Unix socket paths have a length limit.

By default, `cosmos.share_weights=true` launches one model instance for both roles from `rollout.model.model_path`. A caller-owned server can be used with `rollout.sglang.launch_server=false` and `rollout.sglang.server_urls='[http://host:30000]'`; it is left running. The entry checks the advertised checkpoint path. Use ordinary inference mode and leave action normalization statistics unset.

For independent weights, set `cosmos.share_weights=false`, choose `cosmos.simulator.model_path`, and add a `simulator` component placement for its server, for example GPU `1` while `env,rollout` uses GPU `0`. Other simulator serving settings inherit `rollout.sglang` and can be overridden under `cosmos.simulator.sglang`; checkpoint, launch mode, URLs, and worker names are independently configured. An external simulator uses `cosmos.simulator.sglang.launch_server=false` and its own `cosmos.simulator.sglang.server_urls`. Each role's checkpoint is checked separately. Cleanup stops only servers launched by the entry, including after failed warmup. Shared mode requires `cosmos.simulator.model_path=null` and empty simulator server URLs. The LIBERO comparison entry currently requires shared mode.

Set `env.eval.debug.save_artifacts=true` to record a run, using a new artifact directory. Under `<log_path>/artifacts`, inspect `config.json`, `server.json`, and `episode_000/`. `server.json` identifies sharing mode and both role servers. Each `cycle_000/`-style directory retains the input canvas, `actions_native.npy`, dynamics request/response, predicted video, next image, and status. The episode's `closed_loop.mp4` omits repeated conditioning frames. Rewards are interface placeholders; metrics report generated length/chunk count, not task success. Artifact recording is disabled by default.

For one cycle, override both `env.eval.max_episode_steps=16` and `env.eval.max_steps_per_rollout_epoch=16`. For submitted compute tasks, use the same entry, shared paths, and a fixed code snapshot. In a multi-node Ray allocation, use its cluster address instead of `RAY_ADDRESS=local`. Choose compatibility libraries on each execution node before starting Ray there; do not propagate a driver-570 override blindly to nodes with newer drivers. On a shared host, inspect and stop only your run's processes before reusing its GPU.

CPU tests exercise real worker inference/step methods against a fake HTTP service, including exact action handoff and three-cycle image feedback. A real Nano run on an A800 with driver `570.172.08`, compatibility libraries `580.196.07`, Torch `2.13.0+cu130`, and SGLang `0.5.19` also completed three cycles: 48 action steps and a 49-frame stitched video. Artifact checks confirmed finite `(16, 10)` actions per cycle, exact action transfer, and exact image feedback between cycles. This validates inference and feedback wiring; task quality requires a separate evaluation.

## Native WoVR training

Run the standard embodied trainer with `cosmos3_wovr` after setting the Cosmos checkpoint, LeRobot keyframes and reward checkpoint in the [training configuration](../../examples/embodiment/config/cosmos3_wovr.yaml).

```bash
bash examples/embodiment/run_embodiment.sh cosmos3_wovr
```

The default configuration places the FSDP actor, rollout worker and world-model environment on GPUs 0, 1 and 2 respectively. Shared mode publishes actor weights to the rollout and simulator replicas. For a frozen independent simulator, set `cosmos.share_weights=false` and `cosmos.simulator.model_path`. Use `runner.resume_dir` to restore a standard embodied checkpoint.

The [English guide](../../docs/source-en/rst_source/evaluations/guides/cosmos3_sglang.rst) and [Chinese guide](../../docs/source-zh/rst_source/evaluations/guides/cosmos3_sglang.rst) include the native SDK revision, asset requirements and commands for both weight modes.

## GRPO in LIBERO-Spatial

To use LIBERO physics and task-success rewards, set `actor.model.model_path` and `env.train.action_stats_path` in the [LIBERO training configuration](../../examples/embodiment/config/libero_spatial_grpo_cosmos3.yaml), then run:

```bash
bash examples/embodiment/run_embodiment.sh libero_spatial_grpo_cosmos3
```

The default placement uses GPU 0 for the actor and GPU 1 for rollout and LIBERO rendering. Install the native Cosmos SDK and the LIBERO environment with `requirements/install.sh embodied --model cosmos3 --env libero` using the SDK revision in the guides above.
