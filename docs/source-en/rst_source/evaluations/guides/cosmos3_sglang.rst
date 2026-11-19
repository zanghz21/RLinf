Cosmos3 SGLang Evaluation
==========================

Evaluate Cosmos3 on the LIBERO simulator using the SGLang backend: the model runs in a standalone SGLang server process, and rollout workers act as clients that send observations, receive action commands, and feed them into the simulation. Suitable for inference-only evaluation scenarios.

To check a pipeline in which Cosmos3 also predicts the next observation, use :ref:`cosmos3-learned-loop`. That workflow starts from a recorded LIBERO-Spatial image and tests generated feedback; simulator task-success evaluation uses the workflow below. For matched initial states and action replay videos, use :ref:`cosmos3-simulator-comparison`.

To update a Cosmos policy inside its learned simulator, use :ref:`cosmos3-wovr-training`. That training entry uses the native Cosmos SDK and a trained reward model. Both closed-loop entries select shared or separate policy/simulator weights with ``cosmos.share_weights``.

How It Works
----------------------------------------

Each GPU runs one SGLang server (``server_type: embodied``) executing ``Cosmos3OmniDiffusersPipeline``, exposing the action policy as the HTTP endpoint ``POST /v1/actions/generations``. The eval driver hands each server URL to a rollout worker, which turns one step of N environments into one or more requests.

The server only batches prompts of equal token length, so ``Cosmos3SGLangAdapter.request_groups`` first groups the environments by their augmented prompt, then splits each group into chunks of at most ``rollout.sglang.server.batching_max_size`` environments. The example config sets it to ``8``; left unset, the adapter follows SGLang's own default of ``1``, which sends every environment in its own request. The worker posts each group separately and reassembles the results in the original environment order.

Each response carries one record per environment: ``data[i].action.values`` holds that environment's ``[horizon, raw_action_dim]`` normalized rot6d chunk, and the adapter sorts the records by ``input_index`` before de-normalizing and converting them to 7-D axis-angle for LIBERO.

.. code:: text

   EnvWorker(libero) --obs(images+task)--> Cosmos3SGLangAdapter.request_groups
     groups the N envs by prompt, each group at most batching_max_size envs
        --one POST /v1/actions/generations per group-->
   SGLang server (Cosmos3OmniDiffusersPipeline, diffusion num_inference_steps steps)
        --response data[i].action.values = [horizon, 10] per env (normalized rot6d)-->
   Cosmos3SGLangAdapter parses each record, ordered by input_index:
     take first 10 channels → quantile de-normalize → rot6d(6) to axis-angle(3)
   the worker reassembles the groups into [N, 16, 7] in env order
        --[N, 16, 7]-->
   EnvWorker.chunk_step advances the simulation

Installation
----------------------------------------

Install the LIBERO environment and SGLang 0.5.19 with its ``diffusion`` extra. The ``sglang`` model pins torch 2.13.0 and picks the CUDA 12 or CUDA 13 build that the driver supports:

.. code-block:: bash

   bash requirements/install.sh embodied --model sglang --env libero
   source .venv/bin/activate

Prepare Checkpoint
----------------------------------------

The eval input is a diffusers component directory ``model_diffusers``, produced by converting the SFT checkpoint via cosmos-framework. The full four-step conversion is described in the "Checkpoint Conversion" section of :doc:`Cosmos3 SFT <../../examples/embodied/sft_cosmos3>`.

.. note::

   Evaluation **does not** require network access to HuggingFace or the Qwen3-VL cache: the tokenizer is copied into ``model_diffusers/text_tokenizer/`` during conversion, and the server reads it directly from there.

Run LIBERO-Spatial
----------------------------------------

The default config is ``evaluations/libero/libero_spatial_cosmos3_eval_sglang.yaml``. Before running, point the YAML at your local ``model_diffusers``:

.. code-block:: yaml

   rollout:
     model:
       model_path: /path/to/model_diffusers          # eval input diffusers directory
       action_stats_path: /path/to/cosmos-framework/cosmos_framework/data/generator/action/normalizer_stats/libero_native_frame_wise_relative_rot6d.json  # rot6d stats file from cosmos-framework

   env:
     eval:
       total_num_envs: 128   # adjust by GPU count / memory (example: 128 for 8 GPUs)

Then run:

.. code-block:: bash

   bash evaluations/run_eval.sh libero libero_spatial_cosmos3_eval_sglang

**What this command does:** launches one Cosmos3 SGLang server per GPU, starts the LIBERO environment for evaluation, prints per-episode success/failure, and summarizes the success rate at the end. Logs are written to ``logs/<timestamp>-<config>/eval_embodiment.log``.

Key Configuration
----------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Field
     - Description
   * - ``rollout.model.model_path``
     - diffusers checkpoint directory (eval input).
   * - ``rollout.model.action_stats_path``
     - Quantile stats file for action de-normalization; must be from the same source as SFT (``*_rot6d.json``).
   * - ``rollout.model.action_normalization``
     - ``quantile_rot``; must match training.
   * - ``rollout.model.raw_action_dim`` / ``action_dim``
     - Model side 10 (rot6d) / env side 7 (axis-angle); **both required** (see FAQ).
   * - ``rollout.model.num_action_chunks``
     - Number of action steps returned per request (example 16); ``env.eval.max_steps_per_rollout_epoch`` must be divisible by this.
   * - ``rollout.model.num_inference_steps`` / ``num_frames`` / ``size``
     - Diffusion steps and input video specs; must match training.
   * - ``rollout.sglang.server.num_gpus`` / ``tp_size``
     - GPUs per server and TP; both 1 for single-GPU deployment (one server per GPU).
   * - ``rollout.sglang.server.batching_max_size``
     - Largest number of environments the server batches in one request (example 8). The adapter splits each prompt group into chunks of this size; left unset it follows SGLang's default of 1, one request per environment.
   * - ``rollout.sglang.http_timeout_s``
     - HTTP timeout; diffusion inference is slow, recommend ``600``.
   * - ``env.eval.total_num_envs``
     - Number of parallel environments; adjust by GPU / memory.

Configure Policy Guidance
----------------------------------------

To set the strength of text conditioning during policy sampling, configure
``rollout.model.guidance_scale``. Use a finite value greater than or equal to
``1.0``; ``1.0`` uses the conditional prediction without CFG amplification.
For Cosmos3 Edge, select the RLinf extension so the action endpoint preserves
the requested value:

.. code-block:: yaml

   rollout:
     model:
       guidance_scale: 1.5
     sglang:
       server:
         external_model_package: rlinf.models.embodiment.cosmos3.sglang_extension
         pipeline_class_name: RLinfCosmos3Pipeline

The server worker loads this module through SGLang's official
``SGLANG_EXTERNAL_MODEL_PACKAGE`` interface in each server process.
The extension reuses the Cosmos3 model and pipeline stages while keeping the
installed SGLang source unchanged. Policy requests that omit guidance use
``1.0`` with this extension; requests for ``forward_dynamics`` retain SGLang's
video sampling defaults and explicit overrides. Distilled checkpoints retain
their fixed guidance of ``1.0``.

The learned-loop example selects this extension and policy guidance ``1.5``.
When connecting to an externally managed server, launch it with the same
external module and pipeline selection.

Verification
----------------------------------------

Check ``eval_embodiment.log`` and confirm these milestones appear in order:

1. Server launch: ``Launching sglang server (server_type=embodied) ...``
2. Weight loading complete: ``[RunAI Streamer] Overall time to stream 28.3 GiB ... to cpu: <seconds>`` (usually within tens of seconds on local disk; **missing this line** means loading is stuck — see FAQ)
3. Server ready: ``sglang server assigned: rank=i -> http://...``
4. Per-episode results: ``[libero eval] task_id=.., trial_id=.., success=..``
5. Summary: ``success_once`` / ``success_at_end`` / ``num_trajectories``

LIBERO trajectory counting rules: see :ref:`libero-eval-config`.

.. _cosmos3-learned-loop:

Run a Learned World Model Loop
----------------------------------------

Use the existing evaluation workers to run Cosmos3 as both policy and world model. ``SGLangEmbodiedWorker`` requests actions; ``EnvWorker`` runs ``Cosmos3Env`` and sends those unchanged normalized 10D actions to the simulator server for forward prediction. ``EmbodiedEvalRunner`` feeds the final generated frame into the next policy call. This inference-only flow uses the rollout model configuration; training uses the separate entry below.

The policy uses ``InferenceHTTPClient`` for action requests. For forward dynamics, ``Cosmos3Client`` sends an inference-only request (``rollout=false``) to the simulator's official ``POST /rollout/generate`` endpoint. The server returns decoded tensors in a MessagePack response with safetensors data. RLinf converts floating-point pixels to ``uint8`` RGB in memory using SGLang's clamp-and-truncate rule, then passes the final frame directly to the next policy call. Conditioning images use lossless PNG data URIs in memory. MP4 encoding and decoding are outside the feedback path.

Configure ``examples/embodiment/config/cosmos3_closed_loop.yaml`` with your checkpoint and LeRobot dataset paths. The default selects LIBERO-Spatial episode ``1261``, frame ``0``, dataset task index ``30``. It concatenates external and wrist views, predicts 16 actions, and requests 17 video frames at 20 FPS. The dataset's 10 FPS metadata is recorded separately because initialization uses one still observation. Subsequent observations come from generated video.

Prepare the environment inside your workspace, following the :download:`runtime instructions <../../../../../toolkits/cosmos3/README.md>`. The example requirements pin SGLang 0.5.19 and Torch 2.13.0; ``uv --no-config pip install`` avoids the repository's general Torch override. Reuse your workspace's ``../.venv`` when it is already prepared.

For a CUDA 13 build on an NVIDIA data center GPU with driver 570, extract ``cuda-compat-13-0`` into ``../.cuda-compat/13.0`` using the download, checksum, and ``dpkg-deb -x`` commands in the runtime instructions. Reuse that directory if it is already prepared. ``examples/embodiment/run_with_cuda_compat.sh`` locates the libraries relative to the checkout's parent directory and prepends them to ``LD_LIBRARY_PATH`` for its command and child processes. Set ``COSMOS3_CUDA_COMPAT_DIR`` to the exact library directory if you use another location. The wrapper enables the override on driver 570, leaves the library path unchanged on drivers >=580, and rejects other branches.

Select an available physical GPU in ``cluster.component_placement``. For example, edit the configuration to select GPU 5:

.. code-block:: yaml

   cluster:
     component_placement:
       env,rollout: 5

The default configuration selects GPU 0. RLinf discovers physical devices through NVML, so ``CUDA_VISIBLE_DEVICES`` alone does not select worker placement. From the RLinf checkout, run three cycles with local assets:

.. code-block:: bash

   env -u http_proxy -u https_proxy -u all_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
     RAY_ADDRESS=local PYTHONPATH="$PWD" HF_HUB_OFFLINE=1 \
     XDG_CACHE_HOME="$PWD/../.cache" \
     CUDA_CACHE_PATH="$PWD/../.cache/cuda" TRITON_CACHE_DIR="$PWD/../.cache/triton" \
     bash examples/embodiment/run_with_cuda_compat.sh ../.venv/bin/python \
     examples/embodiment/eval_cosmos3_closed_loop.py \
     runner.logger.log_path=/shared/cosmos3/run-001 \
     env.eval.max_episode_steps=48 env.eval.max_steps_per_rollout_epoch=48

This command creates an independent Ray session for a single-node smoke test and removes proxy settings from that process so the server health check connects directly. The CUDA wrapper only configures library loading. For Ray temporary data on shared storage, set ``RAY_TMPDIR`` to a short path or a short symlink to that storage; Ray's Unix socket paths have a length limit.

The default configuration starts one server, one rollout worker, and one environment worker on the same node. The command's 48 environment steps are three chunks of 16 actions. Use a new output path for every debug run. For one cycle, override both ``env.eval.max_episode_steps=16`` and ``env.eval.max_steps_per_rollout_epoch=16``. To reuse a compatible server, set ``rollout.sglang.launch_server=false`` and ``rollout.sglang.server_urls='[http://model-server:30000]'``. HTTP and HTTPS server URLs are supported; the server can run on another node. The entry checks its checkpoint path and leaves externally owned servers running. External servers must support the rollout endpoint, use ordinary inference, and leave action normalization statistics unset.

The default ``cosmos.share_weights=true`` loads one model instance for both roles, using ``rollout.model.model_path``. Keep ``cosmos.simulator.model_path=null`` and its server URLs empty in this mode. To use independent weights, edit the same YAML:

.. code-block:: yaml

   cosmos:
     share_weights: false
     simulator:
       model_path: /path/to/simulator/model_diffusers
   rollout:
     model:
       model_path: /path/to/policy/model_diffusers
   cluster:
     component_placement:
       env,rollout: 0
       simulator: 1

This starts a policy server on GPU 0 and a simulator server on GPU 1. Simulator serving options inherit ``rollout.sglang`` except for the independently configured checkpoint, launch mode, URLs, and worker names. Override other settings under ``cosmos.simulator.sglang``, for example ``+cosmos.simulator.sglang.server.batching_max_size=2``. For external simulation, set ``cosmos.simulator.sglang.launch_server=false`` and ``cosmos.simulator.sglang.server_urls='[http://simulator:30000]'``; simulator placement is then unused. Each role may independently launch or reuse a server. Separate external roles require distinct URLs. Startup checks each role's advertised checkpoint and cleanup stops only servers owned by this run, including after failed model initialization. Debug ``server.json`` records ``share_weights`` and the ``policy`` and ``simulator`` server metadata.

Set ``env.eval.total_num_envs`` to run multiple learned environments in one environment worker. Actions have shape ``[B,T,10]`` and observations have shape ``[B,H,W,3]``. Each chunk dispatches B independent HTTP requests concurrently, then keeps the responses in environment order. Observations and episode counters advance only after every response succeeds. A failed batch waits for the other client requests to finish and raises without retrying generation.

To select distinct starting images, set ``env.eval.dataset.episode_indices`` and ``env.eval.dataset.frame_indices`` to lists of length B. When either list is ``null``, the corresponding scalar ``episode_index`` or ``frame_index`` is repeated for all environments. For example, append these overrides to the learned-loop command:

.. code-block:: bash

   env.eval.total_num_envs=2 \
     'env.eval.dataset.episode_indices=[1261,1262]' \
     'env.eval.dataset.frame_indices=[0,0]'

Each simulator request still contains one sample. ``rollout.sglang.server.batching_max_size`` controls policy batch capacity and defaults to ``1`` in the learned-loop configuration. Concurrent simulator HTTP submission does not imply a batched GPU forward pass or a throughput gain; scheduling inside the unchanged SGLang server determines dynamics execution.

By default, ``env.eval.debug.save_artifacts=false``: the environment keeps the current observation in memory and does not accumulate trajectory frames or write images, videos, or request records. Metrics report generated length and chunk count; placeholder reward tensors do not measure task success.

To inspect a run, enable debug artifacts:

.. code-block:: yaml

   env:
     eval:
       debug:
         save_artifacts: true
       video_cfg:
         save_video: false

Debug mode records the input and next-observation PNGs, raw predicted frames, native actions, request metadata, and per-cycle status under ``<log_path>/artifacts/``. Multiple environments use separate ``episode_000/env_000/``, ``episode_000/env_001/`` directories; single-environment runs retain the ``episode_000/`` layout. It also saves preview MP4s and ``closed_loop.mp4``, joining the trajectory without repeated conditioning frames. The next observation always comes from the raw frames, including in debug mode; preview compression cannot change feedback pixels. Keep ``video_cfg.save_video=false`` because this environment owns its debug recording.

Check that each dynamics request receives the policy's unchanged native actions and that the next cycle's observation exactly matches the preceding raw prediction's final frame. Debug mode retains trajectory frames for the preview; leave it disabled for runs that only need metrics.

A base Nano checkpoint can exercise this connection before a tuned checkpoint is available. Completing the loop verifies inference and feedback wiring; task quality requires a separate evaluation.

Use the same entry inside submitted compute tasks with shared asset/output paths and an immutable code snapshot. The default is a single-node example; HTTP transport also permits a separately hosted compatible server. Select compatibility libraries for each execution node before starting Ray; a driver-570 override must not be applied blindly to newer-driver nodes.

.. _cosmos3-wovr-training:

Train Cosmos in Its Learned Simulator
-------------------------------------

The ``cosmos3_wovr`` recipe uses the standard embodied training entry. The rollout
worker samples native 10D actions. ``WorldModelEnv`` runs Cosmos forward dynamics,
feeds its last generated frame back to the policy and scores each action's frame
with a frozen reward model. The FSDP actor uses RLinf's GRPO advantages and
clipped actor loss. The default placement assigns one GPU each to the actor,
rollout and environment workers.

Install the native SDK revision used by the adapter, then supply an unquantized
Cosmos export with trained LIBERO action heads, a LeRobot v2 dataset and a
trained RLinf ResNet reward checkpoint. The native export's ``config.json`` can
also be supplied with ``actor.model.native_config_path``.

.. code-block:: bash

   COSMOS3_GIT_REF=4ed6244506e00bb270ed17d4532897f2aaa7bc16 \
     bash requirements/install.sh embodied --model cosmos3 --env libero
   source .venv/bin/activate
   export PYTHONPATH="$PWD" EMBODIED_PATH="$PWD/examples/embodiment"
   python examples/embodiment/train_embodied_agent.py --config-name cosmos3_wovr \
     actor.model.model_path=/shared/cosmos-policy \
     env.train.dataset.root=/shared/lerobot/libero \
     'env.train.dataset.keyframes=[{episode_index:1261,frame_index:0}]' \
     env.train.reward_model.model_path=/shared/rewards/libero.pth

Choose unsolved keyframes from the target task. Each GRPO group starts from the
same frame and instruction. The default classifier sees the external view
cropped to 160 by 160 from the dual-camera canvas. ``success_reward_threshold``
in ``env.train`` sets the first success. The actor's loss mask includes that action
and excludes subsequent actions. For each chunk, the rollout samples a denoising
transition index and records its input, output and behavior log probabilities.
The actor replays that transition and sums log probabilities across the ten
action channels. The optimized ratio is the conditional transition density.

With ``cosmos.share_weights=true``, the actor publishes updated weights to both the
rollout and simulator workers before the next scheduled rollout. The simulator
holds its own inference replica. To train against a fixed simulator, append the
following overrides to the same command. For an independent native model config,
also set ``cosmos.simulator.native_config_path``.

.. code-block:: bash

   cosmos.share_weights=false cosmos.simulator.model_path=/shared/cosmos-simulator

The usual embodied checkpoint settings apply. Set ``runner.save_interval`` to the
desired number of iterations and resume with
``runner.resume_dir=/shared/run/checkpoints/global_step_5``. Training metrics
are available through ``runner.logger``. The reward measures predicted success in
the learned simulator. Use the LIBERO evaluation for physical-simulator success.

.. _cosmos3-libero-grpo:

Train Cosmos3 with GRPO in LIBERO-Spatial
------------------------------------------

Use ``libero_spatial_grpo_cosmos3`` to train the native Cosmos3 policy with
LIBERO's physics and task-success rewards. Supply the unquantized policy export
and the action statistics used to train its LIBERO action head.

.. code-block:: bash

   COSMOS3_GIT_REF=4ed6244506e00bb270ed17d4532897f2aaa7bc16 \
     bash requirements/install.sh embodied --model cosmos3 --env libero
   source .venv/bin/activate
   export PYTHONPATH="$PWD" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
   python examples/embodiment/train_embodied_agent.py --config-name libero_spatial_grpo_cosmos3 \
     actor.model.model_path=/shared/cosmos-policy \
     env.train.action_stats_path=/shared/libero_native_frame_wise_relative_rot6d.json

Each GRPO group starts from the same LIBERO task and initial state. The policy
observes the external and wrist cameras together. The environment worker
denormalizes each 10D action and converts its 6D rotation to axis-angle for
LIBERO's 7D controller. The actor uses the saved native diffusion transitions
to update the policy and syncs its weights to rollout. The default placement uses
GPU 0 for the actor and GPU 1 for rollout and LIBERO rendering. Set
``runner.val_check_interval`` to a positive number to evaluate during training.
To change the suite for both training and evaluation, override
``env.train.task_suite_name``, for example with ``libero_goal``.

.. _cosmos3-simulator-comparison:

Compare Cosmos3 and LIBERO Videos
----------------------------------------

The comparison entry requires ``cosmos.share_weights=true`` so each comparison worker uses one model server for both roles. Independent role weights are supported by the closed-loop evaluation and training entries above.

Compare the generated dynamics with LIBERO using ``examples/embodiment/compare_cosmos3_libero.py``. Within each task/reset pair, all branches start from the same LIBERO state after settling. The script gives Cosmos3 the exact camera canvas rendered by that reset, so the initial pixels match. A recorded LeRobot image cannot reconstruct the complete physical state; this comparison uses a live LIBERO reset.

Run the three comparisons with the checkpoint from ``cosmos3_closed_loop.yaml`` and the action statistics used during its SFT:

.. code-block:: bash

   MUJOCO_GL=egl PYOPENGL_PLATFORM=egl RAY_ADDRESS=local PYTHONPATH="$PWD" \
     bash examples/embodiment/run_with_cuda_compat.sh ../.venv/bin/python \
     examples/embodiment/compare_cosmos3_libero.py \
     comparison.output_dir=/shared/cosmos3/comparison-001 \
     comparison.action_stats_path=/path/to/cosmos-framework/cosmos_framework/data/generator/action/normalizer_stats/libero_native_frame_wise_relative_rot6d.json \
     comparison.suite=libero_spatial 'comparison.task_ids=[0,6]' 'comparison.reset_ids=[0,1]' \
     '~cluster.component_placement' '+cluster.component_placement={env:"0-1",rollout:"0-1"}' \
     comparison.steps=240 comparison.num_envs=2 rollout.model.guidance_scale=1.5

The command loads ``examples/embodiment/config/cosmos3_libero_compare.yaml`` and enumerates four task/reset combinations in task-major order: task 0 with resets 0 and 1, then task 6 with resets 0 and 1. RLinf launches two comparison workers using the environment placement and two model servers using the rollout placement, on GPUs 0 and 1. Each comparison worker keeps a fixed server for policy and dynamics. Cases are divided into groups of at most ``comparison.num_envs`` and assigned to comparison workers in round-robin order. Here each worker runs two environments; an incomplete last group retains its actual size. Select the suite and lists with ``comparison.suite``, ``comparison.task_ids``, and ``comparison.reset_ids``. Each list must be nonempty and contain distinct nonnegative integers. The configuration defaults select tasks 0–9 and resets 0–9, with up to 10 environments per worker. Task 6's instruction is “pick up the black bowl next to the cookie box and place it on the plate.” Keep ``comparison.steps`` a positive multiple of the action chunk length. Use ``comparison.settle_steps``, ``comparison.reset_seed``, and ``comparison.camera_size`` to configure settling, reset randomness, and camera resolution. Select the checkpoint with ``rollout.model.model_path``, policy guidance with ``rollout.model.guidance_scale``, and simulator guidance with ``env.eval.generation.guidance_scale``. To change denoising steps, set both ``rollout.model.num_inference_steps`` and ``env.eval.generation.num_inference_steps`` to the same value. Policy and dynamics must also agree on frame count, canvas size, frame rates, and seed. The environment and rollout placements must select the same number of workers. For a single GPU, set both placements to ``0``; the separate learned-loop evaluation entry remains limited to one worker pair.

For each task/reset pair, the script records four trajectories and pairs them into three videos:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Video
     - Cosmos3 (Left)
     - LIBERO (Right)
   * - ``01_independent_rollouts.mp4``
     - Policy rollout
     - Policy rollout
   * - ``02_cosmos_actions.mp4``
     - Original policy rollout
     - Replay Cosmos3 actions
   * - ``03_libero_actions.mp4``
     - Replay LIBERO actions
     - Original policy rollout

Within each group, policy inference combines environments with the same prompt into one multi-image HTTP request, up to ``rollout.sglang.server.batching_max_size``. The comparison configuration defaults this capacity to ``comparison.num_envs``; override it independently to use smaller policy batches. Mixed prompts form separate requests, and a final partial batch keeps its actual size. Both independent rollouts use the same environment membership, order, prompts, and seed for corresponding policy calls.

Cosmos3 dynamics still submit one HTTP request per environment concurrently. LIBERO uses RLinf's vectorized ``LiberoEnv`` with one subprocess per environment and advances each action across the vector before the next action, including during replay. ``comparison.num_envs`` sets the per-worker environment limit, while the policy capacity sets how many compatible observations share one policy request. Simulator HTTP concurrency does not imply batched dynamics execution inside SGLang. Replay branches use the source rollout's saved actions throughout. Each policy response is retained as normalized native 10-D actions for Cosmos3 and as converted 7-D commands for LIBERO, including LIBERO's gripper convention. The Cosmos3 replay uses the retained native actions directly, avoiding a reconstruction from 7-D commands. LIBERO's controller clipping and gripper binarization make its executable commands a lossy projection of the native actions.

Camera canvases show the external view on the left and wrist view on the right. The default 240-step run contains the shared initial frame plus one frame per action, at 20 FPS. Chunk boundaries omit repeated conditioning frames. Feedback always uses raw frames in memory; the saved MP4s are for viewing.

Use a new output directory for each run. The root ``summary.json`` lists the task/reset cases, assigned workers, output paths, and completion status; ``config.yaml`` records the resolved configuration. Each case has its own directory, such as ``task_006/reset_000/``, containing the three videos, initial pixels, suite and settled physical states, and a ``manifest.json`` with checks and timing summaries. Its four trajectory subdirectories contain ``video.mp4``, ``actions_native.npy``, ``actions_libero.npy``, and ``metrics.json``. Set ``comparison.save_raw_frames=true`` to retain uncompressed frame arrays. Timing records separate policy calls from simulator steps and include rendering or HTTP overhead. For each simulator chunk, ``concurrent_batch_s`` measures the whole group; overlapping per-environment ``simulator_s`` values must not be summed as group wall time. ``policy_batch`` records the request membership, position, batch size, and seed, including partial batches. Shared policy-request latency is repeated in each member's ``policy_s`` and must also be counted only once when aggregating across environments. LIBERO records its vector-wide duration for each member with ``timing_scope=vector_chunk``; count this shared duration only once as well. For throughput comparisons, use them alongside the videos rather than as a controlled GPU throughput benchmark.

To reuse compatible servers, set ``rollout.sglang.launch_server=false`` and provide one URL per comparison worker, in rank order, such as ``rollout.sglang.server_urls='[http://model-a:30000,http://model-b:30000]'``. The script checks ``/v1/actions/metadata`` against the configured policy capacity and saves each server's response and worker assignment in ``workers.json``. It leaves externally managed servers running.

FAQ
----------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Symptom
     - Fix
   * - SGLang cannot find model components
     - Confirm both ``model_path`` entries point to the ``model_diffusers`` directory containing ``model_index.json``.
   * - Actions are wrong / nothing succeeds
     - Verify ``action_normalization`` / ``action_stats_path`` / ``num_inference_steps`` / ``num_frames`` / ``size`` match SFT.
   * - First batch HTTP timeout
     - Increase ``rollout.sglang.http_timeout_s`` and ``http_max_retries``.
   * - Local requests blocked by proxy
     - Add the server's advertised IP to ``NO_PROXY``. For the local smoke command above, unset HTTP/HTTPS/ALL proxy variables in both cases.
   * - LIBERO rendering errors
     - ``run_eval.sh`` renders with EGL by default; on hosts without EGL, set ``MUJOCO_GL=osmesa`` and ``PYOPENGL_PLATFORM=osmesa``.
   * - GPU not released before re-run
     - Check ``nvidia-smi`` and stop only processes owned by your run. Check for residual ``ray::SGLangServerGroup`` processes before reusing that GPU.
