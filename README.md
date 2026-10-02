# ShotSense

本地 DNG 调色参数研究原型，输入须包含有效相机白平衡信息。EfficientNet-B0 语义特征与 132 维线性 Lab 统计融合，预测 FiveK 五位专家参数的算术均值。建议使用旧版 Camera Raw **PV2003** 绝对值；`HighlightRecovery` 不等于现代 `Highlights`。正式验证仅通过曝光与高光恢复；对比度、饱和度及白平衡输出保持实验状态。

## 运行

在项目根目录执行：

```sh
venv/bin/python -m streamlit run app/streamlit_app.py
```

打开 http://127.0.0.1:8501，上传未调色 DNG 或点击“使用项目样例”，查看参数并下载 JSON。输入上限 128 MB、4000 万像素，单次处理超时 60 秒。本地应用每次只运行一个 RAW 任务。页面并列展示保持原比例的 RAW 基准图与近似应用效果，最长边 1600 像素，可下载 PNG。默认开启高光保护，可调节 0–100% 应用强度；实际预览参数与剪裁统计显示在页面，并写入 JSON。只模拟已验证的曝光和高光压缩，实验参数不参与。近似显影不等同于 Lightroom，也不保证恢复原始剪裁细节。

## 环境和复现

已验证 macOS arm64、Python 3.9.8。开发与训练环境：

```sh
python3 -m venv venv
venv/bin/python -m pip install -r requirements.lock.txt
venv/bin/python -m pip check
venv/bin/python -m pytest -q
```

原始数据保留在 `data/raw/dngs/` 和 `data/raw/fivek_dataset/raw_photos/fivek.lrcat`。Catalog 只读，异常标签隔离，旧标签保存不可变快照。预训练权重必须预先位于 `artifacts/torch/checkpoints/efficientnet_b0_rwightman-7f5810bc.pth`；训练首次准备权重需要网络，ONNX 推理完全离线。

```sh
venv/bin/python -m src.extract_labels
venv/bin/python -m src.preprocess --limit 20 --workers 1
venv/bin/python -m src.train --pilot --metadata data/processed/pilot/metadata.npz --output-dir artifacts/pilot_model --epochs 30
venv/bin/python -m src.export_onnx --checkpoint artifacts/pilot_model/best.pt --metadata data/processed/pilot/metadata.npz --output-dir artifacts/pilot_model
venv/bin/python -m scripts.prepare_data --workers 6
venv/bin/python -m src.train
venv/bin/python -m src.export_onnx
```

缓存按处理代码、依赖、标签与配置哈希识别；重跑自动续传、重建损坏文件。配置变化时需使用新目录，或明确 `--rebuild`。Pilot 和正式数据、划分、模型独立保存。正式划分按 photo ID、seed=42、80/10/10 固定，X 标准化只使用训练集。尚无连拍／近重复分组，不能将 photo ID 划分视为已排除所有相似场景泄漏。

## 命令行推理

```sh
venv/bin/python -m src.inference path/to/input.dng --output result.json --preview input.jpg
```

JSON 区分 `recommended_absolute` 和 `experimental_absolute`，附单位、旧流程语义、模型与数据版本、耗时。没有同版本基线时不输出 Delta。ONNX 推理不导入 torch/torchvision；仍需 NumPy、ONNX Runtime、rawpy、colour-science、OpenCV、Pillow。

## 验收依据

- `data/intermediate/label_audit.json`：字段来源、省略默认值证据、异常 ID、43 个修复字段、流程版本与标签哈希。
- `data/intermediate/color_audit.json`：线性／D50 配置、共享与独立 RAW 上下文逐像素比较。
- `data/intermediate/preprocessing_benchmark.json`：相同 Pilot 的 1/2/6 worker 吞吐和峰值内存。
- `data/processed/manifest.json`、`splits.json`：正式数据核算、版本与互斥划分；5000 张中有效 4946 张、标签异常 3 张、缺少有效相机 WB 的不支持输入 51 张。
- `data/processed/input_support_audit.json`：逐文件重查 RAW header 的支持范围审计及保留的原始失败报告。未知处理错误仍会失败；不静默启用自动 WB。
- `artifacts/model/evaluation.json`：基线、物理线性回归、语义／双分支对比、一次最终测试、专家分歧、相机／Catalog 标签／长尾及 As-Shot 白平衡分层误差。
- `artifacts/model/deployment_report.json`：PyTorch／ORT 等价误差与 CPU forward p50/p95。模型耗时和 RAW 端到端耗时分别报告。

参数通过固定验证集基线只是研究有效性证据；业务可接受误差尚未标定。六项参数不能完整复现专家显影，现代 Lightroom 映射、忠实高清预览、额外 RAW 格式和已调色 JPEG 不在已验证范围内。正式测试宏平均误差比常量基线降低 18.7%，曝光 MAE 0.281 EV、高光恢复 MAE 7.428；FP32 CPU forward p95 19.39 ms，26 项测试通过。完整验收汇总见 `artifacts/model/ACCEPTANCE.md`，详细进度以 `SHOTSENSE_MASTER_PLAN.md` 为准。

预览改进的详细计划见 `PREVIEW_IMPROVEMENT_PLAN.md`。三张真实图像的旧／新效果和剪裁统计见 `artifacts/preview_v2/acceptance.json`；`venv/bin/python -m scripts.benchmark_previews` 可重现这些结果。新版真实浏览器验收受已保存权限限制，未执行。

当前预览版本为 `linear-raw-shoulder-v3`：正常中间调保持线性曝光，高光采用连续肩部；页面与 JSON 显示基准预览通道上限诊断。诊断不能证明传感器过曝或恢复纹理。16 张 v2/v3 对比及记录见 `artifacts/preview_v3/REVIEW.md`，复现命令为 `venv/bin/python -m scripts.compare_preview_v3`。

## GitHub 仓库内容

仓库包含代码、依赖锁、主计划与开发日志、正式 ONNX 模型及训练 checkpoint、验收 JSON、预览对比 JPG 和历史源码快照。原始 FiveK DNG/Catalog、提取标签、预处理缓存、语义特征缓存、预训练骨干缓存、虚拟环境及重复预览 PNG 留在本地，不纳入 Git。

克隆后先按“环境和复现”创建环境。正式模型已包含，可直接上传自己的有效 DNG 推理，无需重新训练。“使用项目样例”需要本地 `data/raw/dngs/` 中存在 DNG；全套数据测试、训练及照片对比复现需要另外准备原始数据和对应缓存。文档中的 `data/` 审计路径及逐图 PNG 指向开发环境本地保留的资料。
