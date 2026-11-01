# Pose-Aligned Generative Object Reconstruction

面向 AR 场景的位姿对齐物体重建：结合多视图图像和已知相机位姿，通过三维查询投影、多视图特征聚合及 SS／SLat 两阶段条件生成，重建完整物体 Mesh。

## 目录

```text
pose_aligned_reconstruction/   模型、缓存准备、训练、推理和评测
ar_ss_flow/                    位姿投影与条件数据加载组件
reconvggt_ar_adapter_a/         特征与骨干模型适配组件
trellis_point_prior_mv/        坐标和多视图公共工具
pixal3d_multiview/              数据集预处理依赖
ReconViaGen/                   项目使用的 TRELLIS／ReconViaGen 源码及 VGGT 组件
scripts/                       运行环境配置
```

共享模块按当前模型的依赖关系一起提供，可从仓库根目录直接使用 Python 模块入口。

## 环境

参考环境为 Linux、Python 3.10、PyTorch 2.4 和 CUDA 12.1。安装 CUDA Toolkit 及与其匹配的 PyTorch 后：

```bash
git clone https://github.com/yeruoxiangchen/yxctracking.git
cd yxctracking
python -m pip install -r ReconViaGen/requirements.txt
python -m pip install -r requirements.txt
source scripts/env.sh /data/pose-aligned-reconstruction
```

`ReconViaGen/requirements.txt` 包含 CUDA 扩展轮子，适用于 Python 3.10／CUDA 12／PyTorch 2.4。其他组合需使用对应版本的 FlashAttention、nvdiffrast、spconv 和 Gaussian rasterizer。VGGT 源码已随仓库提供，入口会设置其导入路径。

基础预训练模型使用 [`Stable-X/trellis-vggt-v0-2`](https://huggingface.co/Stable-X/trellis-vggt-v0-2)。数据集、特征缓存、训练权重和评测产物单独存放，通过参数传入。

## 输入与数据准备

真实采集数据包含 RGB 图像、前景掩码、相机内参及外参。以下入口依次准备视频缓存、运行时输入和 DINO 特征输入，具体必填参数见各入口帮助：

```bash
python -m pose_aligned_reconstruction.dataset_tools.prepare_omni_real_video_cache --help
python -m pose_aligned_reconstruction.dataset_tools.prepare_omni_real_runtime_inputs --help
python -m pose_aligned_reconstruction.dataset_tools.prepare_omni_real_dino_only_model_inputs --help
```

训练数据使用包含物体标识、相机与图像特征、SS／SLat 监督目标的 manifest 和缓存。对应准备入口：

```bash
python -m pose_aligned_reconstruction.prepare_proobjaverse_official_slat_compact_cache --help
python -m pose_aligned_reconstruction.materialize_proobjaverse_official_ss_targets --help
python -m pose_aligned_reconstruction.prepare_proobjaverse_official_slat_dino_cache --help
```

`compact` 缓存、目标解码审计和训练 manifest 具有配套的格式及身份校验，使用同一批数据生成的文件。

## 训练

以下路径代表已经准备好的缓存、监督目标和评测绑定文件。模型的两个阶段分别训练；单卡调试可将 `--nproc_per_node` 改为 `1`。

SS 阶段：

```bash
torchrun --standalone --nproc_per_node=8 \
  -m pose_aligned_reconstruction.train_proobjaverse_official_native_ss_no_vggt \
  --cache_manifest /data/pose-aligned-reconstruction/cache/ss_training_manifest.json \
  --output_dir /data/pose-aligned-reconstruction/outputs/ss \
  --max_steps 30000 --grad_accum 1 --warmup_steps 600 \
  --min_condition_views 1 --max_condition_views 8 \
  --amp_dtype bf16 --gradient_checkpointing --seed 42
```

SLat 阶段：

```bash
torchrun --standalone --nproc_per_node=8 \
  -m pose_aligned_reconstruction.train_proobjaverse_official_native_slat_no_vggt \
  --architecture v2 \
  --cache_manifest /data/pose-aligned-reconstruction/cache/slat_manifest.json \
  --lifting_cache_manifest /data/pose-aligned-reconstruction/cache/lifting_manifest.json \
  --target_decoder_audit /data/pose-aligned-reconstruction/cache/decoder_audit.json \
  --native_ss_report /data/pose-aligned-reconstruction/reports/native_ss_report.json \
  --stock_slat_freeze /data/pose-aligned-reconstruction/reports/stock_slat_freeze.json \
  --output_dir /data/pose-aligned-reconstruction/outputs/slat \
  --max_steps 30000 --grad_accum 1 --warmup_steps 3000 \
  --min_condition_views 1 --max_condition_views 8 \
  --amp_dtype bf16 --gradient_checkpointing --seed 42
```

使用 `--resume /path/to/checkpoint.pt` 恢复训练；使用 `--init_checkpoint` 在新的数据身份下初始化可训练参数。前者保留优化器、步数和缓存身份校验。

## 推理与评测

显式指定推理输入和 SS／SLat 产物：

```bash
python -m pose_aligned_reconstruction.infer_real_proobjaverse_official_ss_slat \
  --model_input_manifest /data/pose-aligned-reconstruction/inputs/model_input_manifest.json \
  --native_ss_report /data/pose-aligned-reconstruction/reports/native_ss_report.json \
  --native_slat_checkpoint /data/pose-aligned-reconstruction/checkpoints/slat/step_030000.pt \
  --cross_deployment_bridge_report /data/pose-aligned-reconstruction/reports/cross_deployment_bridge.json \
  --stock_slat_freeze /data/pose-aligned-reconstruction/reports/stock_slat_freeze.json \
  --expected_slat_step 30000 --weights ema --seeds 42 --device cuda \
  --output_dir /data/pose-aligned-reconstruction/outputs/inference
```

SS 部署报告绑定相应检查点，跨阶段报告绑定同一组 SS／SLat 权重及评测文件。上述文件需来自配套训练与评测流程。`--object CATEGORY:OBJECT_ID` 可限制推理对象。

几何评测入口包括 `evaluate_native_ss_stock_slat_mesh.py`、`evaluate_proobjaverse_official_native_ss_stock_slat.py` 与 `mesh_benchmark_metrics.py`，提供网格与参考几何的距离和 F-score 等计算。

`infer_current` 和 `current_30k.py` 保留论文冻结实验的路径、模型身份及校验规则，供已有实验产物复现；新的训练或数据部署使用上面的显式参数入口。

## 检查代码包

```bash
python -m pose_aligned_reconstruction.verify_package --imports
python -m unittest discover -s pose_aligned_reconstruction -p 'test_*.py' -v
```

代码中 `pose_point_depth_mv.*` 字符串是旧权重和缓存的格式标识，继续保留以兼容已有产物；Python 源码使用 `pose_aligned_reconstruction` 包。

## 致谢与许可

项目基于 [TRELLIS](https://github.com/microsoft/TRELLIS)、[ReconViaGen](https://github.com/GAP-LAB-CUHK-SZ/ReconViaGen)、[VGGT](https://github.com/facebookresearch/vggt) 和 DINOv2 构建。随附第三方源码保留原版权说明；ReconViaGen 许可见 [ReconViaGen/LICENSE](ReconViaGen/LICENSE)。
