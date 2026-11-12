# Cosmos3 Policy and World Model Loop

Run Cosmos3 as both an action policy and an action-conditioned world model, starting from one recorded LIBERO-Spatial observation. The runner loads one checkpoint in a SGLang server, requests an action chunk, and supplies those exact actions to a separate forward-dynamics request. The final generated frame becomes the next observation.

This first version checks that the two model calls and observation feedback work. A base Nano checkpoint can exercise the pipeline; useful manipulation and physically accurate video depend on the checkpoint. Full RLinf worker integration and task-success evaluation are separate workflows.

## Prepare the Inputs

Edit [config/closed_loop_nano.yaml](config/closed_loop_nano.yaml) to select your checkpoint, dataset, server, and generation settings. The provided defaults use these shared-storage assets:

| Input | Default |
| --- | --- |
| Checkpoint | `/mnt/public2/hongzhi/models/Cosmos3-Nano` |
| LeRobot dataset | `/mnt/public2/hongzhi/dataset/physical-intelligence/libero` |
| Initial observation | Episode `1261`, frame `0`, dataset task index `30` |
| Instruction | Pick up the black bowl next to the cookie box and place it on the plate |
| Camera layout | External image on the left, wrist image on the right |
| Model actions | 16 normalized 10D actions: translation, rot6d, gripper |
| Generated video | 17 frames at 20 FPS, including the conditioning frame |

Dataset task indices differ from LIBERO suite-local task indices. The selected episode belongs to LIBERO-Spatial; its first recorded observation need not be an untouched simulator reset state.

The dataset supplies images and task text only. The runner preserves normalized model actions between the policy and world model, without passing through LIBERO's 7D controller representation. It does not use recorded actions or future images to continue the loop.

The source dataset reports 10 FPS. The model's configured rate is 20 FPS for both calls; a single initial still image does not require replay at the source rate. Record both rates when comparing future runs with demonstrations.

## Run One Cycle

Use an isolated Python 3.11 environment with SGLang v0.5.19 pinned to commit `0bcd822377da7b5718e674eaf9c870d349424dd1` in [requirements-closed-loop.txt](requirements-closed-loop.txt), plus the client dependencies listed there. The pin fixes the policy envelope, forward-dynamics request fields, and video polling API. The launcher verifies the installed SGLang commit before loading the model.

Install the requirements on a compute machine with a CUDA 13-capable driver:

```bash
SGLANG_BUILD_RUST_EXTS=none uv pip install \
  --python /path/to/venv/bin/python \
  -r toolkits/cosmos3/requirements-closed-loop.txt
```

This installs the pinned server and the image/action client dependencies into the selected environment. The build flag disables optional Rust extensions. This release requires PyTorch 2.13.0; its default wheel uses CUDA 13, and the official CUDA 12.8 index has no matching wheel. Confirm CUDA availability in the selected environment before loading the checkpoint.

Select an available GPU and run from the RLinf checkout:

```bash
CUDA_VISIBLE_DEVICES=0 COSMOS_PYTHON=/path/to/venv/bin/python \
  bash toolkits/cosmos3/run_closed_loop_task.sh \
  --output-dir /shared/cosmos3/runs/one-cycle \
  --cycles 1
```

This starts one model server, decodes the selected dataset observation, generates actions, and conditions a video on them. The runner stops its server when it exits. Choose a new output directory for each run so traces remain attributable to one invocation.

The launcher forwards arguments to `python -m toolkits.cosmos3.run_closed_loop`. Set `COSMOS_PYTHON` to the runtime's interpreter, or activate that environment and leave the variable unset. Supply `--config /path/to/config.yaml` to use a different configuration.

## Run Generated Feedback

After the one-cycle check succeeds, use a fresh output directory for three consecutive cycles:

```bash
CUDA_VISIBLE_DEVICES=0 COSMOS_PYTHON=/path/to/venv/bin/python \
  bash toolkits/cosmos3/run_closed_loop_task.sh \
  --output-dir /shared/cosmos3/runs/three-cycles \
  --cycles 3
```

The second cycle receives the first cycle's final generated canvas, and the third receives the second's. Both requests within a cycle use the same current image, domain, action width, timing, and unchanged checkpoint. Generated canvases retain the external/wrist layout without concatenating or padding them again.

Use `--server-url http://127.0.0.1:30000` to connect to an existing compatible server instead of launching one. The runner checks the server's model path and advertised identity and leaves externally managed servers running. An external server must use the pinned API and leave `action_stats_path` unset so both endpoints exchange normalized actions. Its source revision cannot be verified through `/server_info`; the manifest records the advertised runtime version. Keep ordinary inference enabled; SGLang's diffusion RL rollout/log-probability mode is a different API.

On failure, the runner removes its pending video job and stops a server it launched. In this SGLang release, deleting a job on an external server removes job metadata but cannot abort GPU generation; the external server owner manages that work.

## Inspect and Reproduce a Run

Inspect the resolved configuration, run manifest, request/response records, native and conditioning action arrays, per-cycle input/output images, and generated video. Verify that the action arrays agree and each next input is the previous output. A run passes this milestone when the requested cycles complete with finite actions and valid frames; visual plausibility is not a pass condition.

Each `cycle_000/` directory contains `input.png`, `actions_native.npy`, `actions_conditioning.npy`, `policy_request.json`, `policy_response.json`, `dynamics_request.json`, `dynamics_response.json`, `prediction.mp4`, and `next.png`. The run directory also contains `manifest.json`, `resolved_config.yaml`, `metrics.json`, `server.json`, initial camera images, and the stitched `rollout.mp4`. Locally launched servers write `runtime.json` and `logs/server.log`. A failed run preserves its available artifacts and records the error in `manifest.json`.

Run the same launcher inside a submitted compute task for longer experiments. Use an immutable code snapshot and shared paths for the checkpoint, dataset, and output directory; pass the platform's job identifier through `--task-id`. The launcher executes inside an allocation and does not submit or allocate jobs itself.

When a tuned checkpoint becomes available, update the configuration and verify that its action width, domain, normalization, camera layout, and state requirements match. Changing the checkpoint path alone is sufficient only when those interfaces remain compatible.
