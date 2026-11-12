# Cosmos3 Tools

The policy/world-model loop uses RLinf's existing evaluation workers. Start with [`examples/embodiment/eval_cosmos3_closed_loop.py`](../../examples/embodiment/eval_cosmos3_closed_loop.py) and [`config/cosmos3_closed_loop.yaml`](../../examples/embodiment/config/cosmos3_closed_loop.yaml).

`SGLangEmbodiedWorker` generates native normalized 10D actions. `EnvWorker` runs `Cosmos3Env`, which sends those exact actions to the same SGLang server for forward dynamics. `EmbodiedEvalRunner` feeds the final generated image back to the policy. This inference-only evaluation follows the existing rollout model configuration and does not launch an actor or perform training updates.

Both roles use `InferenceHTTPClient` for HTTP transport. `Cosmos3Client` retains only video-job submission, polling, and download orchestration. Each worker owns its client instance; they connect to the same model server. Video submission has no automatic retries.

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

This launches one shared server, coordinates both workers, saves artifacts, and stops its server. A caller-owned server can be used with `rollout.sglang.launch_server=false` and `rollout.sglang.server_urls='[http://host:30000]'`; it is left running. External servers must use the same checkpoint, ordinary inference mode, and no action normalization statistics. The entry checks the advertised checkpoint path.

Each run requires a new artifact directory. Under `<log_path>/artifacts`, inspect `config.json`, `server.json`, and `episode_000/`. Each `cycle_000/`-style directory retains the input canvas, `actions_native.npy`, dynamics request/response, predicted video, next image, and status. The episode's `closed_loop.mp4` omits repeated conditioning frames. Rewards are interface placeholders; metrics report generated length/chunk count, not task success.

For one cycle, override both `env.eval.max_episode_steps=16` and `env.eval.max_steps_per_rollout_epoch=16`. For submitted compute tasks, use the same entry, shared paths, and a fixed code snapshot. In a multi-node Ray allocation, use its cluster address instead of `RAY_ADDRESS=local`. Choose compatibility libraries on each execution node before starting Ray there; do not propagate a driver-570 override blindly to nodes with newer drivers. On a shared host, inspect and stop only your run's processes before reusing its GPU.

CPU tests exercise real worker inference/step methods against a fake HTTP service, including exact action handoff and three-cycle image feedback. A real Nano run on an A800 with driver `570.172.08`, compatibility libraries `580.196.07`, Torch `2.13.0+cu130`, and SGLang `0.5.19` also completed three cycles: 48 action steps and a 49-frame stitched video. Artifact checks confirmed finite `(16, 10)` actions per cycle, exact action transfer, and exact image feedback between cycles. This validates inference and feedback wiring; task quality requires a separate evaluation.

See the [English guide](../../docs/source-en/rst_source/evaluations/guides/cosmos3_sglang.rst) and [Chinese guide](../../docs/source-zh/rst_source/evaluations/guides/cosmos3_sglang.rst). Checkpoint conversion utilities remain in this toolkit directory.
