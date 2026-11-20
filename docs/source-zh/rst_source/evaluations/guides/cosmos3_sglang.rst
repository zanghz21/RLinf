Cosmos3 SGLang 评测
=====================

用 SGLang 后端在 LIBERO 仿真器上评测 Cosmos3：模型放进独立的 SGLang server 进程，rollout worker 作为客户端发观测、收动作、送入仿真。适用于只做推理评测的场景。

如果还希望由 Cosmos3 预测下一次观测，使用 :ref:`cosmos3-learned-loop`。该流程从 LIBERO-Spatial 的记录图像开始，验证模型生成的观测能否持续反馈；下面的仿真评测流程则用于统计任务成功率。需要从相同初始状态对比两种 simulator，并交叉 replay 动作时，使用 :ref:`cosmos3-simulator-comparison`。

需要在模型生成的环境中训练 Cosmos policy 时，使用 :ref:`cosmos3-wovr-training`。该入口使用原生 Cosmos SDK 和已训练的奖励模型。推理闭环与训练闭环均通过 ``cosmos.share_weights`` 选择共用权重或独立权重。

工作原理
----------------------------------------

每张 GPU 起一个 SGLang server（``server_type: embodied``），运行 ``Cosmos3OmniDiffusersPipeline``，把动作策略暴露成 HTTP 端点 ``POST /v1/actions/generations``。eval driver 把 server URL 下发给 rollout worker，worker 把 N 个环境的一步观测拆成一个或多个请求发出。

server 只会把 token 长度相同的 prompt 放在同一批，因此 ``Cosmos3SGLangAdapter.request_groups`` 先按补全后的 prompt 给环境分组，再把每组按 ``rollout.sglang.server.batching_max_size`` 切成若干份，每份最多这么多个环境。示例配置中该值为 ``8``；不设置时 adapter 采用 SGLang 自身的默认值 ``1``，即每个环境单独发一次请求。worker 逐组发送，再把结果按原来的环境顺序拼回去。

响应中每个环境各占一条记录：``data[i].action.values`` 是该环境的 ``[horizon, raw_action_dim]`` 归一化 rot6d 动作块。adapter 先按 ``input_index`` 排序，再去归一化并转成 7 维 axis-angle 送入 LIBERO。

.. code:: text

   EnvWorker(libero) --观测(图像+指令)--> Cosmos3SGLangAdapter.request_groups
     按 prompt 给 N 个环境分组，每组最多 batching_max_size 个环境
        --每组发一次 POST /v1/actions/generations-->
   SGLang server（Cosmos3OmniDiffusersPipeline，扩散 num_inference_steps 步）
        --响应中每个环境一条 data[i].action.values = [horizon, 10]（归一化 rot6d）-->
   Cosmos3SGLangAdapter 按 input_index 顺序逐条解析：
     裁前 10 通道 → quantile 去归一化 → rot6d(6) 转 axis-angle(3)
   worker 把各组结果按环境顺序拼成 [N, 16, 7]
        --[N, 16, 7]-->
   EnvWorker.chunk_step 推进仿真

安装
----------------------------------------

安装 LIBERO 环境和带 ``diffusion`` extra 的 SGLang 0.5.19。``sglang`` model 会固定 torch 2.13.0，并按驱动支持的版本选择 CUDA 12 或 CUDA 13 的 wheel：

.. code-block:: bash

   bash requirements/install.sh embodied --model sglang --env libero
   source .venv/bin/activate

准备 Checkpoint
----------------------------------------

评测输入是 diffusers 组件目录 ``model_diffusers``，由 SFT checkpoint 经 cosmos-framework 转换得到，完整四步见 :doc:`Cosmos3 SFT <../../examples/embodied/sft_cosmos3>` 的「转换 Checkpoint」一节。

.. note::

   评测\ **不需要**\ 联网拉 HuggingFace 或 Qwen3-VL 缓存：转换时 tokenizer 已拷进 ``model_diffusers/text_tokenizer/``，server 直接从该目录读。

运行 LIBERO-Spatial
----------------------------------------

默认配置 ``evaluations/libero/libero_spatial_cosmos3_eval_sglang.yaml``。开跑前改 YAML 指向本地 ``model_diffusers``：

.. code-block:: yaml

   rollout:
     model:
       model_path: /path/to/model_diffusers          # 评测输入的 diffusers 目录
       action_stats_path: /path/to/cosmos-framework/cosmos_framework/data/generator/action/normalizer_stats/libero_native_frame_wise_relative_rot6d.json  # cosmos-framework 的 rot6d 统计文件

   env:
     eval:
       total_num_envs: 128   # 按 GPU 数 / 显存调整（示例 8 卡用 128）

改好后运行：

.. code-block:: bash

   bash evaluations/run_eval.sh libero libero_spatial_cosmos3_eval_sglang

每张 GPU 起一个 Cosmos3 SGLang server，逐 episode 打印成功与否，最后汇总成功率；日志写到 ``logs/<时间戳>-<config>/eval_embodiment.log``。

关键配置
----------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - 字段
     - 说明
   * - ``rollout.model.model_path``
     - diffusers checkpoint 目录（评测输入）。
   * - ``rollout.model.action_stats_path``
     - 动作去归一化的 quantile 统计文件，须与 SFT 同源（``*_rot6d.json``）。
   * - ``rollout.model.action_normalization``
     - ``quantile_rot``，须与训练一致。
   * - ``rollout.model.raw_action_dim`` / ``action_dim``
     - 模型侧 10（rot6d）/ 环境侧 7（axis-angle）；**两者都必填** （见常见问题）。
   * - ``rollout.model.num_action_chunks``
     - 每次请求返回的动作步数（示例 16）；``env.eval.max_steps_per_rollout_epoch`` 须能被其整除。
   * - ``rollout.model.num_inference_steps`` / ``num_frames`` / ``size``
     - 扩散步数与输入视频规格，须与训练一致。
   * - ``rollout.sglang.server.num_gpus`` / ``tp_size``
     - 每 server 占用 GPU 与 TP；单卡部署均为 1（每卡一个 server）。
   * - ``rollout.sglang.server.batching_max_size``
     - 单次请求中 server 最多批处理的环境数（示例为 8）。adapter 按该值切分每个 prompt 分组；不设置时采用 SGLang 默认值 1，即每个环境单独一次请求。
   * - ``rollout.sglang.http_timeout_s``
     - HTTP 超时；扩散推理慢，建议 ``600``。
   * - ``env.eval.total_num_envs``
     - 并行环境数，按 GPU / 显存调整。

配置 Policy Guidance
----------------------------------------

通过 ``rollout.model.guidance_scale`` 设置 policy 采样时对任务文本的强调程度。该值必须有限且大于等于 ``1.0``；``1.0`` 直接使用条件预测，不进行 CFG 放大。对于 Cosmos3 Edge，选择 RLinf 扩展，让 action 端点保留请求中的 guidance 值：

.. code-block:: yaml

   rollout:
     model:
       guidance_scale: 1.5
     sglang:
       server:
         external_model_package: rlinf.models.embodiment.cosmos3.sglang_extension
         pipeline_class_name: RLinfCosmos3Pipeline

server worker 通过 SGLang 官方的 ``SGLANG_EXTERNAL_MODEL_PACKAGE`` 接口，在每个 server 进程中加载此模块。扩展复用 Cosmos3 模型和 pipeline stages，保持已安装的 SGLang 源码不变。使用该扩展时，policy 请求省略 guidance 会采用 ``1.0``；``forward_dynamics`` 请求沿用 SGLang 的视频采样默认值及显式覆盖规则。蒸馏 checkpoint 仍采用固定的 guidance ``1.0``。

模型观测闭环示例已选择此扩展，并将 policy guidance 设为 ``1.5``。连接外部管理的 server 时，也应在服务启动时选择相同的外部模块和 pipeline。

验证
----------------------------------------

看 ``eval_embodiment.log``，确认这几个节点依次出现：

1. 起 server：``Launching sglang server (server_type=embodied) ...``
2. 权重加载完成：``[RunAI Streamer] Overall time to stream 28.3 GiB ... to cpu: <秒数>`` （本地盘通常数十秒内；**缺这行** 说明加载卡住，见常见问题）
3. server 就绪：``sglang server assigned: rank=i -> http://...``
4. 逐 episode 结果：``[libero eval] task_id=.., trial_id=.., success=..``
5. 汇总：``success_once`` / ``success_at_end`` / ``num_trajectories``

LIBERO 轨迹计数规则见 :ref:`libero-eval-config`。

.. _cosmos3-learned-loop:

运行模型生成的观测闭环
----------------------------------------

通过现有评测 worker，让 Cosmos3 同时生成动作和后续观测。``SGLangEmbodiedWorker`` 请求动作，``EnvWorker`` 运行 ``Cosmos3Env``，将归一化 10 维动作原样传给 simulator server 进行 forward prediction。``EmbodiedEvalRunner`` 将生成视频的最后一帧反馈给下一轮 policy。本流程使用 rollout 模型配置，只执行推理；训练使用下文的独立入口。

policy 通过 ``InferenceHTTPClient`` 请求动作。forward dynamics 由 ``Cosmos3Client`` 向 simulator 的官方 ``POST /rollout/generate`` 接口发送仅推理请求（``rollout=false``）。server 通过 MessagePack 响应返回 safetensors 编码的已解码张量。RLinf 在内存中按 SGLang 的裁剪与截断规则，将浮点像素转换为 ``uint8`` RGB，再把最后一帧直接传给下一轮 policy。条件图像使用内存中的无损 PNG data URI，反馈链路不经过 MP4 编码和解码。

在 ``examples/embodiment/config/cosmos3_closed_loop.yaml`` 中配置 checkpoint 和 LeRobot 数据集路径。默认选择 LIBERO-Spatial episode ``1261``、frame ``0``、数据集 task index ``30``。输入拼接的外部相机与腕部相机图像，预测 16 个动作，再以 20 FPS 生成 17 帧视频。初始化只取一帧观测，因此单独记录数据集的 10 FPS 元数据。后续观测全部来自生成视频。

按照 :download:`运行环境说明 <../../../../../toolkits/cosmos3/README.md>`，在当前工作区内准备环境。示例依赖固定 SGLang 0.5.19 和 Torch 2.13.0；``uv --no-config pip install`` 避免仓库通用的 Torch 覆盖规则影响此环境。若工作区已有 ``../.venv``，可直接复用。

在驱动为 570 的 NVIDIA 数据中心 GPU 上运行 CUDA 13 构建时，按照运行环境说明中的下载、校验和 ``dpkg-deb -x`` 命令，将 ``cuda-compat-13-0`` 解压到 ``../.cuda-compat/13.0``。已有兼容库目录可直接复用。``examples/embodiment/run_with_cuda_compat.sh`` 相对于仓库的父目录定位这些库，并为启动命令及其子进程将库路径加入 ``LD_LIBRARY_PATH`` 最前面。使用其他位置时，将 ``COSMOS3_CUDA_COMPAT_DIR`` 设为实际存放库文件的目录。脚本在驱动 570 上启用兼容库，在驱动 >=580 上保留原库搜索路径，遇到其他驱动分支则报错退出。

通过 ``cluster.component_placement`` 选择空闲的物理 GPU。例如，使用 GPU 5 时，将配置改为：

.. code-block:: yaml

   cluster:
     component_placement:
       env,rollout: 5

默认配置选择 GPU 0。RLinf 通过 NVML 发现物理设备，仅设置 ``CUDA_VISIBLE_DEVICES`` 不会改变 worker 的 placement。在 RLinf 根目录使用本地模型和数据运行三轮闭环：

.. code-block:: bash

   env -u http_proxy -u https_proxy -u all_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
     RAY_ADDRESS=local PYTHONPATH="$PWD" HF_HUB_OFFLINE=1 \
     XDG_CACHE_HOME="$PWD/../.cache" \
     CUDA_CACHE_PATH="$PWD/../.cache/cuda" TRITON_CACHE_DIR="$PWD/../.cache/triton" \
     bash examples/embodiment/run_with_cuda_compat.sh ../.venv/bin/python \
     examples/embodiment/eval_cosmos3_closed_loop.py \
     runner.logger.log_path=/shared/cosmos3/run-001 \
     env.eval.max_episode_steps=48 env.eval.max_steps_per_rollout_epoch=48

该命令为单节点 smoke test 创建独立的 Ray 会话，并在本次进程中取消代理设置，让 server 健康检查直接连接。CUDA wrapper 只配置动态库加载。将 Ray 临时数据放在共享存储时，应把 ``RAY_TMPDIR`` 设为较短的路径，或指向共享存储的短路径符号链接；Ray 使用的 Unix socket 路径有长度限制。

默认配置在同一节点启动一个 server、一个 rollout worker 和一个 env worker。命令中的 48 个环境步对应三个包含 16 个动作的 chunk。每次 debug 运行需要新的输出路径。单轮测试同时覆盖 ``env.eval.max_episode_steps=16`` 和 ``env.eval.max_steps_per_rollout_epoch=16``。复用兼容 server 时，设置 ``rollout.sglang.launch_server=false`` 和 ``rollout.sglang.server_urls='[http://model-server:30000]'``。支持 HTTP 和 HTTPS 地址，server 可以部署在其他节点。入口检查 checkpoint 路径，退出时保留外部管理的 server。外部 server 必须支持 rollout 接口，使用普通推理模式，并保持动作归一化统计参数未设置。

默认 ``cosmos.share_weights=true``，两个角色共用 ``rollout.model.model_path`` 对应的同一个模型实例。此时保持 ``cosmos.simulator.model_path=null``，simulator 的 server URL 列表也保持为空。需要独立权重时，在同一 YAML 中配置：

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

该配置在 GPU 0 启动 policy server，在 GPU 1 启动 simulator server。simulator 的 checkpoint、启动方式、URL 和 worker 名称独立配置，其余服务参数继承 ``rollout.sglang``。可在 ``cosmos.simulator.sglang`` 下覆盖其他参数，例如 ``+cosmos.simulator.sglang.server.batching_max_size=2``。连接外部 simulator 时，设置 ``cosmos.simulator.sglang.launch_server=false`` 和 ``cosmos.simulator.sglang.server_urls='[http://simulator:30000]'``，此时不使用 simulator placement。两个角色分别选择本次启动或复用外部服务；独立模式下，外部服务的 URL 必须不同。入口逐一检查服务公布的 checkpoint，退出时只停止本次启动的服务，模型初始化失败时也会清理。debug 产物 ``server.json`` 记录 ``share_weights`` 及 ``policy``、``simulator`` 两个角色的服务信息。

通过 ``env.eval.total_num_envs``，在一个 env worker 内运行多个模型环境。动作 shape 为 ``[B,T,10]``，观测 shape 为 ``[B,H,W,3]``。每个 chunk 并发提交 B 个独立 HTTP 请求，再按环境顺序接收结果；所有响应成功后，才一起更新观测与 episode 计数。如果一批中有请求失败，先等待其他客户端请求结束，再抛出异常，不自动重试生成。

需要不同初始图像时，将 ``env.eval.dataset.episode_indices`` 和 ``env.eval.dataset.frame_indices`` 分别设为长度 B 的列表。某一列表为 ``null`` 时，对应的单数配置 ``episode_index`` 或 ``frame_index`` 会广播到全部环境。例如，在模型观测闭环命令后追加：

.. code-block:: bash

   env.eval.total_num_envs=2 \
     'env.eval.dataset.episode_indices=[1261,1262]' \
     'env.eval.dataset.frame_indices=[0,0]'

每个 simulator 请求仍只包含一个样本。``rollout.sglang.server.batching_max_size`` 控制 policy 的 batch 容量，在模型观测闭环配置中默认为 ``1``。simulator 的 HTTP 并发提交不等于 GPU batch forward，也不直接保证吞吐提升；dynamics 如何执行由未改动的 SGLang server 内部调度决定。

默认 ``env.eval.debug.save_artifacts=false``，环境只在内存中保留当前观测，不累积轨迹帧，也不写入图像、视频或请求记录。指标只报告生成步数和 chunk 数；接口所需的占位 reward 张量不表示任务成功。

需要检查某次运行时，开启 debug 产物：

.. code-block:: yaml

   env:
     eval:
       debug:
         save_artifacts: true
       video_cfg:
         save_video: false

debug 模式在 ``<log_path>/artifacts/`` 下记录输入与下一次观测的 PNG、预测原始帧、native 动作、请求元数据和每轮状态。多环境分别使用 ``episode_000/env_000/``、``episode_000/env_001/`` 等目录，单环境保留 ``episode_000/`` 布局，并保存预览 MP4 与去除重复条件帧后的 ``closed_loop.mp4``。即使开启 debug，下一次观测也始终来自原始帧，预览视频的压缩不会改变反馈像素。保持 ``video_cfg.save_video=false``，由环境自身管理 debug 录制。

检查 dynamics 请求是否接收到 policy 原样输出的 native 动作，以及下一轮观测是否逐像素等于前一轮原始预测的最后一帧。debug 模式会为预览保留轨迹帧；只需要指标的运行应保持关闭。

获得微调 checkpoint 之前，可用 Nano 基础模型验证连接。闭环跑通说明推理和反馈链路正常，任务表现仍需单独评测。

较长实验可在提交的计算任务中调用同一入口，使用共享数据与输出路径，并固定代码快照。默认示例使用单节点，HTTP 传输也支持独立部署的兼容 server。在各执行节点启动 Ray 之前，选择该节点适用的兼容库；不要将驱动 570 所用的库路径直接应用于使用更新驱动的节点。

.. _cosmos3-wovr-training:

在 Cosmos 模拟环境中训练 Cosmos 策略
--------------------------------------

``cosmos3_wovr`` 配置沿用 RLinf 的 embodied 训练入口。rollout worker 采样原生
10 维动作，``WorldModelEnv`` 调用 Cosmos 预测后续图像，把末帧反馈给策略，并用冻结的奖励模型逐帧打分。
FSDP actor 使用 RLinf 的 GRPO 优势估计和带裁剪的策略损失更新权重。默认部署为 actor、rollout
和环境各分配一张 GPU。

安装适配器使用的 Cosmos SDK 版本，准备带 LIBERO 动作头的未量化 Cosmos 权重、
LeRobot v2 数据集和训练好的 RLinf ResNet 奖励权重。原生模型的 ``config.json``
也可通过 ``actor.model.native_config_path`` 单独指定。

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

从目标任务中选取尚未成功的关键帧。每个 GRPO 组从同一张图像和相同指令出发。
奖励模型默认从双视角画面中裁出 160×160 的外部相机图像。``env.train.success_reward_threshold``
决定首次成功的动作，actor 的损失掩码保留该动作并屏蔽后续动作。每个动作块随机选择一个去噪步骤，
保存该步骤的输入、输出和采样时的对数概率。actor 重算同一步转移的对数概率，按十个动作通道求和，
得到用于更新策略的条件转移概率比。

.. _cosmos3-grpo-trainable-parameters:

两套 Cosmos3 GRPO 配置默认只训练扩散 Transformer（DiT）内部的生成专家，包括生成分支
的最终归一化层，以及 Edge 模型中生成侧的 key 归一化层。如需同时训练 DiT 外的模块，
可按需在启动命令末尾添加以下独立选项。

* ``actor.model.train_action_input_projection=true`` 训练动作输入投影
  （``action2llm``），将带噪动作映射到 Transformer 的隐藏空间。
* ``actor.model.train_action_output_projection=true`` 训练动作输出投影
  （``llm2action``），将隐藏状态映射为动作的去噪速度。
* ``actor.model.train_action_modality_embedding=true`` 训练动作模态嵌入
  （``action_modality_embed``）。
* ``actor.model.train_time_embedder=true`` 训练扩散时间嵌入（``time_embedder``）。

例如，要只训练动作输出投影，可设置
``actor.model.train_dit_backbone=false actor.model.train_action_output_projection=true``。
两套配置都将以上四个模块的开关显式设为 ``false``。即使名称匹配
``actor.model.keys_to_select``，也需要打开对应开关才会训练该模块。

默认 ``cosmos.share_weights=true``。每到权重同步点，actor 会在下一次采样前把权重发送给 rollout
和 simulator。各 worker 持有自己的推理副本。要固定 simulator 权重，请在同一启动命令中追加以下覆盖项。
若独立权重另有原生配置文件，可指定 ``cosmos.simulator.native_config_path``。

.. code-block:: bash

   cosmos.share_weights=false cosmos.simulator.model_path=/shared/cosmos-simulator

``runner.save_interval`` 控制标准 embodied checkpoint 的保存间隔，可通过
``runner.resume_dir=/shared/run/checkpoints/global_step_5`` 恢复训练。
恢复之前只训练动作头的优化器 checkpoint 时，需打开三个动作模块开关，并设置
``actor.model.train_dit_backbone=false``。
指标由 ``runner.logger`` 指定的后端记录。训练奖励表示模型预测的成功程度，LIBERO
仿真中的任务成功率可通过评测流程测量。

.. _cosmos3-libero-grpo:

在 LIBERO-Spatial 中用 GRPO 训练 Cosmos3
-----------------------------------------

用 ``libero_spatial_grpo_cosmos3`` 运行原生 Cosmos3 策略，让 LIBERO 物理仿真器执行动作，
并根据任务是否成功计算奖励。准备带 LIBERO 动作头的未量化 Cosmos 权重，以及训练该动作头时使用的归一化统计文件。
该配置同样默认只训练 DiT 生成专家，可用相同的
:ref:`可训练参数选项 <cosmos3-grpo-trainable-parameters>` 调整训练范围。

.. code-block:: bash

   COSMOS3_GIT_REF=4ed6244506e00bb270ed17d4532897f2aaa7bc16 \
     bash requirements/install.sh embodied --model cosmos3 --env libero
   source .venv/bin/activate
   export PYTHONPATH="$PWD" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
   python examples/embodiment/train_embodied_agent.py --config-name libero_spatial_grpo_cosmos3 \
     actor.model.model_path=/shared/cosmos-policy \
     env.train.action_stats_path=/shared/libero_native_frame_wise_relative_rot6d.json

每个 GRPO 组从同一个任务和初始状态出发。策略读取主相机和腕相机的拼图。
环境 worker 对 10 维动作做反归一化，将 6 维旋转表示转成轴角，交给 LIBERO 的 7 维控制器执行。
actor 用采样时保存的原生去噪转移更新策略，并将权重同步给 rollout。默认 actor 使用 GPU 0，
rollout 和 LIBERO 渲染共用 GPU 1。将 ``runner.val_check_interval`` 设为正整数可在训练中定期评估。
要同时切换训练和评估任务集，可设置 ``env.train.task_suite_name=libero_goal``。

.. _cosmos3-simulator-comparison:

对比 Cosmos3 与 LIBERO 视频
----------------------------------------

对比入口要求 ``cosmos.share_weights=true``，每个 comparison worker 使用同一个模型 server 执行两个角色。独立权重请使用上文的闭环评测或训练入口。

通过 ``examples/embodiment/compare_cosmos3_libero.py``，对比模型预测与 LIBERO 物理仿真的轨迹。每个 task/reset 组合中的所有分支使用同一 LIBERO 状态，并等待场景稳定。脚本把 LIBERO 此时渲染的相机拼图直接作为 Cosmos3 的初始观测，保证起点像素一致。LeRobot 记录图像无法还原完整的物理状态，因此这里使用实际 LIBERO reset。

沿用 ``cosmos3_closed_loop.yaml`` 中的 checkpoint，并指定其 SFT 使用的动作统计文件，运行三组对比：

.. code-block:: bash

   MUJOCO_GL=egl PYOPENGL_PLATFORM=egl RAY_ADDRESS=local PYTHONPATH="$PWD" \
     bash examples/embodiment/run_with_cuda_compat.sh ../.venv/bin/python \
     examples/embodiment/compare_cosmos3_libero.py \
     comparison.output_dir=/shared/cosmos3/comparison-001 \
     comparison.action_stats_path=/path/to/cosmos-framework/cosmos_framework/data/generator/action/normalizer_stats/libero_native_frame_wise_relative_rot6d.json \
     comparison.suite=libero_spatial 'comparison.task_ids=[0,6]' 'comparison.reset_ids=[0,1]' \
     '~cluster.component_placement' '+cluster.component_placement={env:"0-1",rollout:"0-1"}' \
     comparison.steps=240 comparison.num_envs=2 rollout.model.guidance_scale=1.5

命令加载 ``examples/embodiment/config/cosmos3_libero_compare.yaml``，按 task 优先顺序列出四个组合：task 0 的 reset 0、1，以及 task 6 的 reset 0、1。RLinf 在 GPU 0、1 上按 environment placement 启动两个 comparison worker，按 rollout placement 启动两个模型 server；每个 comparison worker 固定使用同一个 server 执行 policy 和 dynamics。组合按每组最多 ``comparison.num_envs`` 个划分，再轮流分配给各个 comparison worker。此示例每个 worker 运行两个环境，尾组保留实际样本数。通过 ``comparison.suite``、``comparison.task_ids`` 和 ``comparison.reset_ids`` 选择 suite 及编号列表。列表须非空，包含不重复的非负整数。配置默认选择 task 0–9 和 reset 0–9，每个 worker 最多运行 10 个环境。task 6 的任务为“拾起饼干盒旁的黑碗，放到盘子上”。``comparison.steps`` 必须是 action chunk 长度的正整数倍。通过 ``comparison.settle_steps``、``comparison.reset_seed`` 和 ``comparison.camera_size`` 调整场景稳定步数、reset 随机种子与相机分辨率。通过 ``rollout.model.model_path`` 指定 checkpoint，通过 ``rollout.model.guidance_scale`` 和 ``env.eval.generation.guidance_scale`` 分别设置 policy 与 simulator guidance。修改去噪步数时，同时把 ``rollout.model.num_inference_steps`` 和 ``env.eval.generation.num_inference_steps`` 设为相同值。两边的帧数、画布大小、帧率与种子也须一致。environment 和 rollout 的 placement 必须指定相同的 worker 数量；单 GPU 时都设为 ``0``。单独的模型观测闭环评估入口仍限制为一对 worker。

每个 task/reset 组合分别记录四条轨迹，生成三组对比视频：

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - 视频
     - Cosmos3（左侧）
     - LIBERO（右侧）
   * - ``01_independent_rollouts.mp4``
     - Policy rollout
     - Policy rollout
   * - ``02_cosmos_actions.mp4``
     - 原 policy rollout
     - Replay Cosmos3 动作
   * - ``03_libero_actions.mp4``
     - Replay LIBERO 动作
     - 原 policy rollout

每组内的 policy 把相同 prompt 的观测合并到一个多图 HTTP 请求，最多包含 ``rollout.sglang.server.batching_max_size`` 个环境。对比配置默认将此容量设为 ``comparison.num_envs``，也可以单独覆盖以减小 policy batch。不同 prompt 分开请求，尾批保留实际样本数。两条独立 rollout 对应的 policy 请求使用相同的环境成员、顺序、prompt 与种子。

Cosmos3 dynamics 仍按环境各发一个 HTTP 请求，并发等待结果；LIBERO 复用 RLinf 的向量化 ``LiberoEnv``，每个环境使用一个子进程，每个动作在所有环境执行后再推进下一个动作，replay 也按此方式推进。``comparison.num_envs`` 决定每个 worker 的环境数量上限，policy capacity 决定一个 policy 请求最多合并多少个兼容观测。simulator 的 HTTP 并发不代表 SGLang 内部对 dynamics 做了 GPU batching。replay 则全程使用来源轨迹保存的动作。每次 policy 响应同时保留归一化的原生 10 维动作，以及转换后可直接交给 LIBERO 的 7 维命令，其中包含 LIBERO 的夹爪约定。Cosmos3 replay 直接读取保留的原生动作，避免从 7 维命令反向重建。LIBERO controller 的裁剪与夹爪二值化会丢失部分信息，因此可执行命令只是原生动作的有损映射。

每个相机拼图左侧为外部视角，右侧为腕部视角。默认 240 步视频包含共同的初始帧和每个动作之后的一帧，播放帧率为 20 FPS；chunk 连接处去除重复条件帧。闭环反馈始终使用内存中的原始帧，MP4 仅用于观看。

每次运行选择新的输出目录。根目录的 ``summary.json`` 列出各个 task/reset 组合、分配的 worker、输出路径和完成状态，``config.yaml`` 保存解析后的配置。每个组合独占一个目录，例如 ``task_006/reset_000/``，其中保存三组视频、初始图像、suite 初始物理状态及稳定后的物理状态，以及记录检查结果和耗时的 ``manifest.json``。其下四个轨迹子目录分别包含 ``video.mp4``、``actions_native.npy``、``actions_libero.npy`` 和 ``metrics.json``。设置 ``comparison.save_raw_frames=true`` 可额外保存未压缩帧数组。耗时记录区分 policy 请求与 simulator 步进，其中包含渲染或 HTTP 开销。每个 simulator chunk 的 ``concurrent_batch_s`` 表示整组墙钟耗时，各环境存在重叠的 ``simulator_s`` 不能相加后当作整组耗时。``policy_batch`` 记录请求成员、样本位置、batch size 与 seed，尾批也按实际大小记录。同一个 policy 请求的延迟会重复写入各成员的 ``policy_s``，跨环境汇总时同样只应计算一次。LIBERO 在每个成员中记录整组向量环境的耗时，并标记 ``timing_scope=vector_chunk``，这项共享耗时也只能计算一次。这些记录适合结合视频观察，不应直接视为受控 GPU 吞吐量基准。

复用兼容 server 时，设置 ``rollout.sglang.launch_server=false``，并按 rank 顺序为每个 comparison worker 提供一个 URL，例如 ``rollout.sglang.server_urls='[http://model-a:30000,http://model-b:30000]'``。脚本通过 ``/v1/actions/metadata`` 检查 server 是否支持配置的 policy 容量，将各 server 的响应与 worker 分配信息保存到 ``workers.json``。退出时保留外部管理的 server。

常见问题
----------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - 现象
     - 处理
   * - SGLang 找不到 model components
     - 确认两处 ``model_path`` 都指向含 ``model_index.json`` 的 ``model_diffusers`` 目录。
   * - 动作乱 / 完全不成功
     - 核对 ``action_normalization`` / ``action_stats_path`` / ``num_inference_steps`` / ``num_frames`` / ``size`` 与 SFT 一致。
   * - 首批请求 HTTP 超时
     - 调大 ``rollout.sglang.http_timeout_s`` 与 ``http_max_retries``。
   * - 本地请求被 proxy 拦截
     - 将 server 对外公布的 IP 加入 ``NO_PROXY``。上面的本地 smoke 命令会取消大小写两组 HTTP/HTTPS/ALL 代理变量。
   * - LIBERO 渲染报错
     - ``run_eval.sh`` 默认使用 EGL 渲染；若机器不支持 EGL，设 ``MUJOCO_GL=osmesa``、``PYOPENGL_PLATFORM=osmesa``。
   * - 重跑前 GPU 未释放
     - 查看 ``nvidia-smi``，只停止属于本次运行的进程。复用该 GPU 前，检查是否残留 ``ray::SGLangServerGroup`` 进程。
