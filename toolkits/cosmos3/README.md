# Cosmos3 Tools

The policy/world-model loop uses RLinf's existing evaluation workers. Start with [`examples/embodiment/eval_cosmos3_closed_loop.py`](../../examples/embodiment/eval_cosmos3_closed_loop.py) and [`config/cosmos3_closed_loop.yaml`](../../examples/embodiment/config/cosmos3_closed_loop.yaml).

`SGLangEmbodiedWorker` generates native normalized 10D actions. `EnvWorker` runs `Cosmos3Env`, which sends those exact actions to the same SGLang server for forward dynamics. `EmbodiedEvalRunner` feeds the final generated image back to the policy. This inference-only evaluation follows the existing rollout model configuration and does not launch an actor or perform training updates.

The default uses one environment, one rollout stage, and one GPU. It starts from LIBERO-Spatial episode 1261, frame 0, then runs three chunks of 16 actions. Only the initial external/wrist images and instruction come from the dataset. Useful behavior depends on the checkpoint; the base Nano checkpoint is used first to exercise the pipeline.

From the RLinf checkout, prepare a virtual environment within the current workspace:

```bash
uv venv ../.venv --python 3.11
SGLANG_BUILD_RUST_EXTS=none uv --no-config pip install --python ../.venv/bin/python \
  -r examples/embodiment/requirements-cosmos3.txt
```

The requirements pin SGLang 0.5.19 at revision `0bcd822377da7b5718e674eaf9c870d349424dd1` and Torch 2.13.0. `--no-config` prevents RLinf's general dependency overrides from changing this runtime. The GPU driver must support its CUDA build. Reuse the workspace environment if already created.

Configure checkpoint, dataset, and output paths in the YAML, then run:

```bash
PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python \
  examples/embodiment/eval_cosmos3_closed_loop.py \
  runner.logger.log_path=/shared/cosmos3/run-001
```

This launches one shared server, coordinates both workers, saves artifacts, and stops its server. A caller-owned server can be used with `rollout.sglang.launch_server=false` and `rollout.sglang.server_urls='[http://host:30000]'`; it is left running. External servers must use the same checkpoint, ordinary inference mode, and no action normalization statistics. The entry checks the advertised checkpoint path.

Each run requires a new artifact directory. Under `<log_path>/artifacts`, inspect `config.json`, `server.json`, and `episode_000/`. Each `cycle_000/`-style directory retains the input canvas, `actions_native.npy`, dynamics request/response, predicted video, next image, and status. The episode's `closed_loop.mp4` omits repeated conditioning frames. Rewards are interface placeholders; metrics report generated length/chunk count, not task success.

For one cycle, override both `env.eval.max_episode_steps=16` and `env.eval.max_steps_per_rollout_epoch=16`. For submitted compute tasks, use the same entry, shared paths, and a fixed code snapshot.

CPU tests exercise real worker inference/step methods against a fake HTTP service, including exact action handoff and three-cycle image feedback. They do not start Ray or load model weights. GPU inference remains unverified: the available driver 570 / CUDA 12.8 is incompatible with this Torch CUDA 13 runtime.

See the [English guide](../../docs/source-en/rst_source/evaluations/guides/cosmos3_sglang.rst) and [Chinese guide](../../docs/source-zh/rst_source/evaluations/guides/cosmos3_sglang.rst). Checkpoint conversion utilities remain in this toolkit directory.
