# ShotSense V3.1：AI 调色参数推荐系统 — 修订实施计划

## 当前状态与计划审查（2026-10-02）

- [x] 完成本次计划审查：核对全部现有源码、5000 张 DNG、专家 JSON、只读 Catalog 查询和关键库官方文档；修订数据契约、任务顺序与阶段验收。
- **实现已扩展至训练、ONNX 和本地应用**：隔离环境、标签审计、颜色管道、缓存、Dataset、模型和 20 张 Pilot 已验收；全量支持范围审计与预处理已完成，N=4946，正式模型、FP32 ONNX、Torch-free 推理与页面样例／上传／下载已完成验收。
- **标签修复**：50000 行 Catalog 联表记录中选择每图／专家唯一非空记录。43 个字段修复，3 张缺少绝对白平衡的图像隔离，4997 张候选。Saturation／HighlightRecovery 省略零值经命名历史记录证明；全部专家记录确认属于 PV2003。
- **可复现环境**：项目独立 venv、锁定依赖、官方 EfficientNet-B0 权重本地缓存；pip check 通过。运行代码与模型使用哈希标识版本。
- **验证状态**：26 项检查通过；正式 PyTorch／ORT 最大归一化误差 3.21e-7，离线／线上同 DNG 特征与 JPEG 完全相同。正式测试宏平均误差比常量降低 18.7%；曝光／高光恢复为验证建议，其余四项为实验。
- **交付范围**：可运行本地参数建议 MVP（http://127.0.0.1:8501）与 JSON 导出；Contrast／Saturation／Temperature／Tint 保持实验，忠实 Lightroom 渲染、现代参数映射和业务误差标定未完成。验收汇总见 artifacts/model/ACCEPTANCE.md。

### 审查发现与处理优先级

| 优先级 | 漏洞／逻辑错误 | 修订后的处理 |
|---|---|---|
| P0 | 绝对参数、Delta 与归一化符号混用 | 固定绝对目标；明确逐图基线后才能算 Delta（KI-004） |
| P0 | Contrast、Temperature 编码区间覆盖不了标签 | 扩展区间，禁止静默裁剪有效标签（KI-003） |
| P0 | rawpy 默认 BT.709 编码被按 ProPhoto CCTF 解码 | 物理输出改为显式线性，验证 D50 矩阵；锁定 sRGB 编码（KI-002） |
| P0 | 空设置覆盖、字段误匹配、缺失直接补 0 | 修复提取器，确认去重、键边界与默认值来源（KI-001） |
| P0 | 高光恢复混写为现代 Highlights | 保留旧流程字段，跨版本转换独立验证（KI-005） |
| P1 | 相机 WB 已应用，却要求恢复绝对 Kelvin/Tint | 增加白平衡可行性 Gate 与上下文／降级规则（KI-006） |
| P1 | 物理特征尺度差异大，划分与标准化未保存 | 固定图像级划分；只用训练集拟合并保存标准化 |
| P1 | 冻结梯度被视为完全冻结骨干 | 保持骨干 eval，验证 BatchNorm buffer 不漂移 |
| P1 | 续传只检查 JPEG，没有特征／版本一致性 | 每个 ID 的图片、特征、配置共同验收，原子写入 |
| P1 | CNN 默认动态 INT8，整条 RAW 流程承诺 <30ms | 先 FP32，按测量选量化，分开模型与端到端计时 |
| P1 | CSS/Canvas 被当作 Lightroom 忠实显影器 | 首版参数建议与导出，预览独立验收（KI-007） |
| P2 | 依赖与本机不一致，计划额外引入 timm | 复用 torchvision；隔离环境验证依赖组合 |

### 执行顺序（20 张 Pilot 已完成）

执行顺序：**依赖与标签契约 → 色彩正确性 → 串行 Pilot → 恢复／失败测试 → Dataset 契约 → 小批并行比较 → 全量预处理 → 训练基线**。

1. 修复标签提取器并输出审计；保留旧 JSON 快照与原 Catalog，确认字段默认值与参数版本。
2. 修正物理输出 gamma、线性缩小与 D50 转换；迁移 `RGB_to_XYZ` API 并分别验证等价性和色彩正确性。
3. 实现单次读取 RAW、双次显影、224×224 JPEG、逐图特征缓存、`--limit`、`--workers` 和失败清单。
4. 固定排序的前 20 个候选 ID 运行串行 Pilot；另用 3 个已知异常 ID 与一个模拟失败验证过滤／重试，不计入这 20 个候选 ID。
5. 完成归一化、Dataset 和小规模数据测试；重跑同一 Pilot 验证续传无重复、损坏补建、版本不匹配拒绝。
6. 对相同样本比较串行／并行结果、吞吐和峰值内存，通过后全量运行并固定正式划分。

---

## 🧠 核心架构设计 (Core Architecture)

目标是从**未应用专家调色的 DNG**推荐官方 Catalog 中的 6 个参数。保留 V3.1 双分支监督回归方案；标签是专家编辑记录，不是唯一审美真值，也不是完整 Lightroom 显影配方。

- **语义分支**：使用已有 torchvision 的 `EfficientNet-B0` 与固定 `IMAGENET1K_V1` 权重，移除分类层、全局池化取得 **1280D**。冻结梯度并持续保持骨干 `eval()`，MLP 单独训练；权重提前准备并记录版本，推理不联网下载。
- **物理分支**：经验证的线性 ProPhoto 输出→XYZ D50→Lab，提取 **132D** 描述统计；不宣称其直接测量传感器曝光或场景照度。
  - 96D：L、a、b 各 32-bin 的归一化全局直方图。
  - 18D：上／中／下三区的 L、a、b 均值与标准差。
  - 18D：L≤33.3、33.3<L≤66.7、L>66.7 的 L、a、b 均值与标准差。现有实现没有分区像素占比，称“亮度分区统计”，不称完整 Zone System。
- **融合回归头**：训练集标准化后的物理特征与语义特征拼接，**1280+132=1412D → 512 → 128 → 6**，隐藏层 ReLU，末层 Tanh。KI-006 若证明需要 WB 上下文，须显式修改输入契约和维数，不能塞入已有 132D 字段。
- **目标定义**：每张图先在原始单位中对 A–E 的 6 个参数分别取算术均值，再归一化；`Y` 保存原始单位均值，保存专家原值与 `Y_std` 供分歧审计。参数均值不保证等于专家图像的平均渲染或任一专家风格。
- **固定字段顺序**：`Exposure, Contrast, Saturation, Temperature, Tint, HighlightRecovery`。界面名称为曝光、对比度、饱和度、色温、色调、高光恢复；模型、NPZ、导出和 API 同序。
- **首版输入范围**：与训练显影流程一致的 DNG；已调色 JPEG／PNG、其他 RAW 格式和现代 Lightroom 参数映射另行验证。

### 6D 参数归一化映射 (Normalization Spec)

下表是本模型采用的**旧流程参数编码区间**，不是所有 Lightroom 版本通用的范围。区间覆盖现有有效专家标签；分布已经修复后的 4997 张有效标签重算，与本表三位小数显示一致；精确值见 label_audit.json。

| 参数 | 原始单位编码区间 | 归一化区间 | 映射中心（→0） | 专家均值 p1～p99 | 专家均值 min～max |
|---|---|---|---|---|---|
| Exposure | `[-4,4]` EV | `[-1,1]` | 0 EV | -0.344～2.254 | -1.518～3.274 |
| Contrast | `[-50,100]` | `[-1,1]` | 25 | 0～34.008 | -6.8～68 |
| Saturation | `[-100,100]` | `[-1,1]` | 0 | -4.8～16.408 | -15.6～30.8 |
| Temperature | `[2000,50000]` K | `[-1,1]` | 26000 K | 2379.938～8202.301 | 2000～35558.054 |
| Tint | `[-150,150]` | `[-1,1]` | 0 | -21.6～37.808 | -96.4～97 |
| HighlightRecovery | `[0,100]` | `[-1,1]` | 50 | 3.8～68.208 | 0～98.4 |

映射：`norm = 2 * (raw - min) / (max - min) - 1`；逆映射：`raw = (norm + 1) / 2 * (max - min) + min`。

- 验证单个专家值和聚合后的 Y；缺失、NaN/Inf、越界明确报错／隔离并记录原因，不裁剪后冒充有效真值。合法高色温必须保留。
- 现有有效样本中，Contrast 均值有 **20 张<0**，Temperature 均值有 **13 张>10000K**，单个专家有 **101 条>10000K**。原映射会产生无法表示的目标或标签损失。
- `norm=0` 只是区间中心，不是“无需调整”。温度区间扩大改变损失尺度，必须单独报告 Kelvin 与 mired 误差；验证显示必要时再考虑非线性温度编码。
- 模型输出 `recommended_absolute`。只有使用方提供同参数版本的 `current_absolute`，才计算 `delta = recommended_absolute - current_absolute`；没有基线不返回 Delta、不计算方向准确率。相机 WB 乘数不是 Lightroom Kelvin/Tint。
- 本表已修订，`ParameterNormalizer` 已实现并验证 NumPy／Tensor 往返；后续映射改动必须同步代码、缓存／模型版本和导出元数据。

---

## 数据、色彩与复现契约

### Catalog 标签

1. 只读 SQLite；验证 A–E collection 名称与 ID。按 `(源文件ID, 专家)` 关联，剔除空设置后要求恰好一条有效记录；零条／多条非空必须失败，不能随意覆盖。
2. 匹配完整键名，不能让 Temperature/Tint 匹配 Incremental 或其他前缀字段。明确 `WhiteBalance=Custom` 时 Custom 字段优先规则，其他模式读取有效绝对值。
3. **仅允许已确认的省略默认值**：非空专家记录中，Saturation 缺失 17390 条、HighlightRecovery 缺失 8597 条；先确认序列化默认语义再补 0，并记录来源。禁止通用“缺失→0”。Temperature/Tint 缺失各 15 条，对应 3 张已知异常图像，不能以相对量代替。
4. 审计 ProcessVersion、CameraProfile、WhiteBalance、来源记录 ID 与默认规则。25000 条中仅 **283 条显式 `ProcessVersion="5.0"`**，其他 24717 条省略；不得据此宣称全部 PV2010，先核实省略版本语义。
5. 原 Catalog 不修改；旧 JSON 保留提取版本快照，修复后输出新标签与差异报告。“保留溯源”不构成永久保留解析 bug 的理由。
6. 输入不使用专家 TIFF、专家 WB 或 Catalog 中 `(default) Input with ExpertC WhiteBalance minus1.5` 的设置，避免目标泄漏。训练标签始终来自官方 Catalog。

### RAW 与颜色

- 单次 `rawpy.imread()`、双次 `postprocess()`；两次使用一致的 WB、去马赛克、方向、裁剪与曝光基准。显式锁定 `use_camera_wb=True`、`use_auto_wb=False`、`no_auto_bright=True`、`bright=1.0`、`half_size=False`、高光模式及其他影响输出的选项，记录 rawpy／LibRaw 版本。
- **物理输出**：`output_color=ProPhoto, output_bps=16, gamma=(1,1)`；归一化为线性 RGB，用面积插值缩至 224×224 后转换 XYZ D50→Lab，不再调用 ProPhoto CCTF 解码。验证 LibRaw 输出矩阵与 colour-science 矩阵相容，不能仅凭枚举名称认定严格 D50。
- **语义输出**：`output_color=sRGB, output_bps=8, gamma=(2.4,12.92)`；缩至 224×224、JPEG quality=95，显式处理 RGB/BGR。训练与线上共用 JPEG 编解码、缩放和 ImageNet mean/std 归一化，避免一边用 JPEG、一边用未压缩数组。
- rawpy 默认 gamma 是 `(2.222,4.5)`，不会随 ProPhoto 枚举自动变成 ROMM 编码；现代码对默认输出使用 ProPhoto CCTF 解码，颜色正确性未通过。[rawpy 参数文档](https://letmaik.github.io/rawpy/api/rawpy.Params.html)、[LibRaw 参数文档](https://www.libraw.org/docs/API-datastruct-eng.html)。
- LibRaw 0.22.1 源码将该输出称为 `ProPhoto D65`，自动 ICC 使用矩阵推导 D50 值。本次源码矩阵推导值与标准 ProPhoto 矩阵的最大系数差约 `1.22e-4`；实现时明确采用并验证的矩阵、白点与容差，避免重复色适应。[LibRaw 色彩常量](https://github.com/LibRaw/LibRaw/blob/0.22.1/src/tables/colorconst.cpp)、[转换实现](https://github.com/LibRaw/LibRaw/blob/0.22.1/src/postprocessing/postprocessing_utils_dcrdefs.cpp)。
- 迁移至 `RGB_to_XYZ(..., colourspace=..., illuminant=..., apply_cctf_decoding=False)`。本次小型数值检查新旧签名 diff=0，仅证明迁移等价，不能证明旧 gamma 正确。
- 保留直接输出 sRGB 的双次显影方案。历史 PSNR 23.5dB 缺少统一编码／白点基准，是两种实现的观察，不能解释为 colour-science 转换必然造成损失。
- 首版全画面直接缩为 224×224，记录纵横比变形限制；它不完全采用预训练权重 resize-256／中心裁切流程。若效果不足再比较保持比例方案，同时更新训练与推理。[torchvision 权重与预处理说明](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b0.html)。
- 广色域 Lab a/b 可能越出直方图区间；计数前把越界值并入端点 bin 并记录比例，不能静默丢掉像素；均值／标准差保留未裁剪 Lab。空亮度区返回零，空间三区必须非空。
- `HighlightMode.Clip` 与低分辨率统计不保留全部可恢复 RAW 高光信息；须通过高光参数基线／消融验证，不能承诺恢复 JPEG 已剪裁的细节。

### 缓存、划分与特征标准化

- `metadata.npz`：`X float32 (N,132)`、`Y float32 (N,6)`、`Y_std float32 (N,6)`、`ids Unicode (N,)`；X/Y 保留原始特征／单位，Dataset 归一化标签。使用 `allow_pickle=False`。
- ID 为小写完整文件名，实际路径映射至原始大小写；检查规范化冲突。任务固定排序，汇总按 ID 排序，不受 worker 完成顺序影响。
- 每个 ID 保存小型特征缓存与 JPEG，主进程汇总 NPZ／报告。写临时文件、验证后原子替换；图片、特征、标签来源及处理版本都有效才跳过。只有 JPEG 存在不算完成。
- 配置清单记录标签哈希、字段／特征顺序、显影／缩放／JPEG 设置、代码与依赖版本；不匹配时明确要求重建，禁止混用。不引入数据库或复杂任务队列。
- `--limit N` 为固定排序后的前 N 个候选 ID，包含已缓存项；重跑处理同一集合。Pilot 缓存版本一致可用于全量，临时划分不覆盖正式划分。
- 输入按成功（含有效续传）／过滤／失败互斥分类，总数等于输入数。零成功不生成可训练 NPZ；未解决失败以非零退出状态和清单报告。
- 正式有效 ID 固定后，seed=42 按**原始照片**划分 80/10/10，保存 `splits.json` 与数据版本；A–E 不跨集合。可识别的连拍／近重复合组后分配，比例允许近似；未知分组限制写入报告。
- X 标准化仅用训练集拟合，零方差列 scale=1；均值／尺度作为模型 buffer 保存并导出 ONNX，Dataset 不重复标准化。标签区间固定，不能用验证／测试集拟合新的映射。
- 首版不做单独改变色彩、曝光、白平衡或空间统计的增强；增加增强时同时更新两个分支及必要标签。

---

## 📂 目录架构 (Directory Architecture)

以下结构已实现；预览采用独立 preview.py，不改变训练与 RAW 推理输入。

```text
ShotSense/
├── data/
│   ├── raw/dngs/                      # 原始 DNG，不修改
│   ├── raw/fivek_dataset/raw_photos/   # 原 Catalog，不修改
│   ├── intermediate/
│   │   ├── expert_labels.json         # 经版本化审计的专家标签
│   │   └── label_audit.json           # 字段／默认／版本／差异报告
│   └── processed/
│       ├── images/                    # 224×224 sRGB JPEG
│       ├── features/                  # 逐图特征与版本缓存
│       ├── metadata.npz               # X、Y、Y_std、ids
│       ├── manifest.json              # 配置与处理报告
│       └── splits.json                # 正式图像级划分
├── src/
│   ├── color_pipeline.py              # 训练／推理共用显影与色彩流程
│   ├── extract_labels.py              # 完整键解析、去重与审计
│   ├── preprocess.py                  # 共用特征提取与离线缓存
│   ├── dataset.py                     # Dataset、ParameterNormalizer
│   ├── model.py                       # 冻结骨干、X 标准化、MLP
│   ├── train.py                       # 基线、训练、验证与 checkpoint
│   ├── export_onnx.py                 # FP32 导出与可选量化
│   └── inference.py                   # 共用预处理 + ONNX Runtime
├── artifacts/                         # 模型、配置、评估与计时报告
├── app/
│   └── streamlit_app.py               # 首版本地应用；远程 API 按需增加
├── tests/
│   ├── test_labels.py                 # 解析边界、空记录与默认语义
│   ├── test_color_pipeline.py         # 编码／矩阵／双次显影
│   ├── test_dataset.py                # 数据与续传契约
│   ├── test_model.py                  # 形状、冻结状态、小批过拟合
│   └── test_inference.py              # 预处理与 PyTorch／ONNX 等价
├── ANTIGRAVITY.md
├── SHOTSENSE_MASTER_PLAN.md
└── requirements.txt
```

---

## 🎯 实施阶段里程碑 (Phases & Milestones)

### Phase 1：可信标签、色彩正确性与可恢复数据通路

- [x] 原始数据齐备：5000 张 DNG 与 Catalog 已存在，ID 对齐。
- [x] GATE 3 原型：5000×5×6 数值记录已提取；正式语义审计已完成（label_audit.json）。
- [x] GATE 1/2 原型：rawpy／colour-science 和 132D 代码已存在；颜色数值与 RAW 上下文等价验收已完成（color_audit.json）。
- [x] **P1-A 环境**：整理 requirements，校正 Torch／OpenCV 声明，复用 torchvision，删除没有调用需求的依赖，只保留一种 OpenCV 包；隔离环境验证依赖冲突、RAW 双次显影与模型导入，记录可复现版本。
- [x] **P1-B 标签**：实现去重与完整键解析；确认省略默认值和 ProcessVersion；输出审计、旧／新差异、隔离 ID／原因，重算归一化表统计。
- [x] **P1-C 色彩**：显式线性物理输出与 sRGB 编码；黑／白／中性灰／彩色样本验证转换与 gamma，灰色 a/b 接近 0；至少一张 DNG 比较共享上下文双次显影与两个独立上下文结果，记录矩阵／容差／版本。API 等价和颜色正确性分开验收。
- [x] **P1-D 串行 Pilot**：实现逐图缓存、原子写入、过滤、ID 对齐与 `--limit 20 --workers 1`。错误处理覆盖显影、转换、特征和写盘全过程，不只 rawpy。
- [x] **P1-E 数据测试**：归一化边界／往返、合法负 Contrast／高色温、缺失拒绝、Tensor／批形状、finite、JPEG 尺寸与 RGB 顺序；验证续传幂等、损坏重建、版本不匹配拒绝、无重复 ID。
- [x] **P1-F 并行 Pilot**：macOS spawn、模块级 worker、`__main__` 保护与 `--workers`；比较 1/2/6 workers，控制 OpenCV／OpenMP 嵌套线程，以吞吐及峰值内存决定默认值，不能只按 CPU 核数决定。
- [x] **P1-G 全量与划分**：全部 5000 输入分类核算；没有未解决处理失败，每个成功 ID 对应完整有效缓存；筛选规则和版本固定后记录最终 N，保存正式 splits 并验证互斥。

**进入 Phase 2**：标签默认／版本语义确认、颜色数值检查通过、Pilot／恢复测试通过、全量失败解决、正式划分固定。只生成文件或形状正确不足以验收。

### Phase 2：基线、双分支训练与有效性评估

- [x] **训练集基线**：逐参数训练集中位数常量预测；仅物理特征的简单线性回归（含常数列，优先 NumPy）；语义／双分支同划分比较。保存专家分歧，不将其视为校准后的模型置信度。
- [x] **白平衡 Gate（KI-006）**：检查相机／场景分层 Temp/Tint 误差，与常量及可得的 As-Shot 基线比较；不足则增加在线可获得的 WB 上下文并更新输入契约，或只交付优于对应基线的非白平衡参数。当前只验证曝光／高光恢复，另外四项保持实验；不能将六项全标为已验证。
- [x] **模型**：1412D MLP，训练集 X 标准化 buffer；验证 `(B,6)`、finite、Tanh 区间以及训练后冻结参数与 BatchNorm buffer 不变。
- [x] **训练检查**：固定 Python／NumPy／Torch seed，记录设备；先小批过拟合检查梯度与标签配对，再全量训练。小批过拟合不替代泛化指标。
- [x] **训练循环**：归一化目标逐参数 Smooth L1／Huber 后等权平均；初始只训练 MLP，AdamW。学习率、batch、epoch 上限与早停耐心保存到配置，通过验证集选择。
- [x] **模型选择**：按验证集六项归一化 MAE 宏平均保存 best checkpoint；同步保存字段顺序、映射、X 统计、显影配置、数据／划分版本、权重版本与种子。选定后一次正式测试；测试失败后再调参需新的最终评估安排，不能循环窥看测试集。
- [x] **评估**：逐参数原始单位 MAE／RMSE、温度 mired MAE、长尾／相机／场景误差及样本数。可靠逐图基线存在时才增加 Delta 三分类方向指标，记录零附近容差、类占比与覆盖率。

**进入 Phase 3**：验证集宏平均误差优于训练集中位数基线；每项声称有效的参数也要优于对应常量基线，并报告与简单回归比较。失败先查标签、尺度、WB 上下文和信息损失，不直接增大网络。业务可接受的绝对误差尚待标定，标定前称研究原型。

### Phase 3：FP32 ONNX、等价性与实测部署

- [x] 导出双输入 FP32：`image float32 (B,3,224,224)`、`physical_raw float32 (B,132)`→同序归一化绝对参数。X 标准化包含在图内；记录 opset、输入名、动态 batch 支持与目标 provider。
- [x] ONNX checker、PyTorch／ORT 小批与边界样本等价；初始 FP32 归一化输出最大误差容差 `1e-4`，超限先排查，再确定有依据的容差。检查反归一化和单张／批量结果。
- [x] 推理共用显影、特征、JPEG 编解码和映射；同一 DNG 比较离线与线上 X／image／预测。推理验收环境不联网、不导入 torch／torchvision；RAW 处理仍需 rawpy、colour-science、NumPy 与图像库。
- [x] FP32 基准：指定硬件、provider、线程、batch=1，预热 10 次、计时至少 100 次，报告 p50／p95、体积。**<30ms 是模型 forward 的待验证目标**；另外测 RAW 读取／双次显影、特征、JPEG 与端到端时间，不以缓存耗时替代。
- [x] **按需量化**：正式 FP32 18.30 MiB，CPU 4 threads p50=18.66ms／p95=19.39ms，满足当前需求，不实施 INT8。以后 FP32 体积／延迟不满足实测需求时再尝试。CNN 优先训练集校准的静态 QDQ INT8；只量化 MLP 时可评估动态 INT8。验证 provider 支持、逐参数 MAE 相对退化≤2%（基准误差为 0 用数值精度容差），且符合原目标或实测改善延迟／体积；不达标保留 FP32。[ONNX Runtime 量化建议](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)。

**进入 Phase 4**：框架／预处理等价、模型包和配置完整、环境可复现、质量与耗时有实测报告。INT8 和 <30ms 均非无条件承诺。

### Phase 4：本地参数建议 MVP 与独立预览验收

- [x] Streamlit 直接调用本地 inference：DNG 上传→参数建议→原图显示→结构化参数下载；远程多客户端需求出现后再增加 FastAPI。
- [x] 检查文件类型、大小和解码像素数；大 RAW 在受限 worker 内处理、清理临时文件，显示明确加载／失败状态。上限按实际内存和耗时写入配置，不能只检查扩展名。
- [x] 返回字段、单位、参数版本语义、模型／数据版本及功能验证状态；无当前参数时显示绝对建议。“应用”使用设值语义，重复点击不叠加 Delta。
- [x] 先导出 JSON；XMP／Lightroom 导入在字段与旧流程映射验证后提供，禁止直接改名 HighlightRecovery 为 Highlights2012／现代 Highlights。
- [x] **近似预览验收**：独立线性 RAW 16-bit→最长边 1600 等比预览；默认高光保护、强度调节、实际参数与 PNG／JSON 对齐；数值、应用及 3 张真实 DNG 离线视觉检查通过。模型输入与已训练模型未改变。详见 PREVIEW_IMPROVEMENT_PLAN.md。
- [ ] **忠实预览 Gate**：自定义曲线仍非 Lightroom/PV2003 等价；完整显影器、同版本配置和原始高光重建须单独验证。真实浏览器验收受保存权限限制，未完成。
- [ ] 代表性图像检查单参数／组合变化与幂等应用。专家还调整 Brightness、Shadows、FillLight、ToneCurve、CameraProfile 等未建模项；六项全预测正确也不能仅用完整专家 TIFF 像素差验收。

**MVP 完成**：Phase 1–3 验收通过，上传／建议／下载／错误处理可用，支持范围及旧流程语义清楚；未验收的忠实高清预览与现代 Lightroom 映射保持未完成。

---

## ⚠️ 已知问题与技术决策 (Known Issues & Decisions)

### KI-001：标签解析边界、空设置与缺失默认

- 异常 ID：`a3131-ke_.dng`、`a3214-ke_-8375.dng`、`a3741-ke_-8337.dng`；实际路径保留原大小写。原因是绝对 WB 字段缺失，不未经证据归因于 PV2003。
- 撤销“只过滤不修提取器”：修解析根因和记录选择，保留旧快照，绝对 WB 仍不可得的图像才隔离。Temperature<1000K 只是已知错误防线，不是完整质量校验。
- JSON 键齐全不证明默认值／版本语义正确；修复与审计已完成，43 个字段变化、3 张异常隔离；保留旧快照。

### KI-002：双次显影保留，颜色验收重做

- 保留单次读取、双次显影；物理线性输出、sRGB 编码显式锁定，API 等价与颜色正确性分别验收。
- 不将 PSNR 23.5dB 解释为转换必然损失。原 GATE 1/2 的“严格管道完成”下调为原型完成。

### KI-003：归一化覆盖有效标签

- 撤销 Contrast `[0,100]` 和 Temperature `[2000,10000]`，使用本表范围。不按 p1～p99 裁剪训练标签，长尾独立报告。
- 温度中心 26000K 是编码结果，不是中性 WB；是否用 mired 等编码由验证误差决定。

### KI-004：绝对参数与 Delta

- 目标是原始单位专家算术均值的绝对建议；Delta 来自使用方提供的同版本基线。没有基线不报告方向准确率。
- 调整方向不能由 Tanh 符号或映射中心推断。

### KI-005：高光恢复与现代 Highlights

- 标签是 HighlightRecovery；旧／新流程控制项和算法不同，不能只改名／区间。[Adobe 流程版本说明](https://helpx.adobe.com/ie/lightroom-classic/desktop/process-and-develop-photos/develop-module-options.html)、[Adobe 旧版流程差异文档](https://helpx.adobe.com/sk/archive/lightroom/lightroom-5-troubleshooting.pdf)。
- 先核实 Catalog 流程语义，现代兼容与忠实预览分别验收。

### KI-006：白平衡上下文与目标泄漏

- 推论：相机 WB 会消除部分原色偏；当前像素特征不包含 WB 基线，单凭处理后像素不能保证恢复绝对 Kelvin/Tint，能力需要分层验证。
- Catalog `InputAsShotZeroed` collection `943690` 有 5000 条唯一非空记录，其中 **3395 条显式 Temperature/Tint、1605 条缺失**。它是基线调查起点，不能假设全量有绝对基线，也不能用专家 WB 补缺。
- Phase 2 必须通过 WB Gate；新增上下文要在训练／线上同定义可得，并更新缓存和模型输入。不可得时先交付经过验证的四项建议，Temp/Tint 保持实验状态，禁止用线上得不到的专家数据提高指标。

### KI-008：缺失相机白平衡的输入支持边界

- 全量显影发现 51 张 DNG 的相机 WB 为 `[0,1,0,0]`，分属旧式相机输入；LibRaw 的隐式自动 WB 不符合训练契约。保持线上／离线拒绝，不引入无验证的 fallback 或专家 WB。
- 数据准备增加独立支持范围审计：只将这一确证的契约违例分类为 unsupported，逐文件用 LibRaw header 重新确认无有效相机 WB，保存 ID、相机、文件信息、原失败报告及选择代码版本。未知解码／数值／I/O 错误仍必须失败，不能一概过滤。
- 最终候选 4997 中排除 51 张 unsupported，正式有效 N=4946；与 3 张标签异常合计隔离 54 张。保留原 4946 张缓存的显影契约与哈希，不重写原数据、不修改颜色算法。正式划分在支持范围固定后创建。

### KI-007：部署与渲染边界

- 模型与 RAW→结果分开计时，量化按实际收益启用；未验证的 CSS／Canvas 渲染只作近似预览。
- 六项参数不构成完整专家配方，高光剪裁和缩小限制恢复能力。

### 环境与依赖检查点

- 验证环境：独立 Python 3.9.8 venv，NumPy 1.26.4、rawpy 0.27.0／LibRaw 0.22.1、colour-science 0.4.4、opencv-python-headless 4.11.0.86、Torch 2.8.0／torchvision 0.23.0、onnx 1.19.1／onnxruntime 1.19.2、Streamlit 1.50.0。requirements.lock.txt 锁定完整环境，pip check 通过。
- 同一 20 张 Pilot：1／2／6 workers 吞吐 1.072／1.813／3.053 张/秒，进程树峰值 RSS 0.82／1.33／2.75 GB，输出完全一致。Pilot 默认 1，正式运行显式选择 6；限制嵌套线程。
- 当前目录没有 Git 仓库；建立版本管理前以源码／标签／配置哈希和快照标识版本，不虚构 commit ID。

---

## 📖 进度日志 (Progress Log)

历史条目保留当时记录；其中“严格颜色管道已完成”、旧归一化范围、“只过滤不修解析器”和 PSNR 因果解释已被本次审查修正，当前规范以上述契约为准。

- **2026-07-21**: 项目正式建立。完成了 `SHOTSENSE_MASTER_PLAN.md` 核心架构与各阶段任务清单的初始化。定义了 `EfficientNet-B0` (1280D) 与物理统计 (132D) 的双分支融合架构（融合至 1412D）。同步初始化了 `ANTIGRAVITY.md` 自动同步与日志规范。
- **2026-07-25**: 架构升级至 **V3.1 Ground-Truth Parameter Supervision**。废弃了残缺的 Kaggle JPG 数据集，重新下载完整的 50GB MIT-Adobe FiveK 官方数据。成功解析 `fivek.lrcat` 提取真实标签（GATE 3），并建立基于 `rawpy` 和 `colour-science` 的绝对颜色科学管道（GATE 1 & 2）。实现了 132D 物理特征提取流水线。
- **2026-09-12**: **Phase 1 深度审查与方案修订**。对全部代码、数据、环境进行了完整审计。关键发现: (1) `extract_labels.py` 对 3 张 PV2003 图像存在 IncrementalTemperature 误匹配 Bug（影响 0.06%，决策为在预处理阶段过滤）; (2) `colour-science 0.4.4` 的 `RGB_to_XYZ` API 弃用警告需迁移至新签名（已验证输出一致）; (3) rawpy 支持单次 imread 双次 postprocess（ProPhoto+sRGB），避免色空间转换精度损失; (4) multiprocessing 兼容性验证通过，8 核 CPU 实测加速比 1.9x。制定了完整的 6D 参数归一化映射方案（基于 25000 条标签的全量统计）、`src/dataset.py` 设计与 `tests/test_dataset.py` 单元测试矩阵。
- **2026-10-02**: **复工状态复核与下一步整理**。确认 5000 张 DNG、25000 条专家记录及 3 张已知异常图像；JPEG 缓存与 metadata 尚未生成，Dataset、模型、训练及应用尚未实施。补充 20 张 Pilot → 数据测试 → 全量预处理的执行顺序与验收标准，标记绝对参数／Delta 定义及当前环境与依赖声明的差异。本次完成计划整理与只读检查，未实施数据管道改动或运行训练。

- **2026-10-02（计划审查修订）**：核对源码、全量标签与只读 Catalog，确认 25000 空设置、省略默认字段、20 张负 Contrast、13 张均值色温>10000K、15 条无绝对 WB 异常记录和 As-Shot 基线缺失；数值检查 RGB_to_XYZ 新旧 API 等价，并确认默认 gamma 与 ProPhoto 解码不匹配。统一绝对输出、旧流程高光语义，完善缓存／划分／训练／ONNX／应用的依赖和验收，同步 ANTIGRAVITY。完成文档修订；运行代码修复和正式 Gate 尚未实施。

- **2026-10-02（实现与 Pilot 验收）**：完成隔离环境和依赖锁、顶层 Catalog 解析与省略默认证据、线性 LibRaw→D50 Lab、可恢复缓存、Dataset／持久划分、冻结骨干训练、FP32 ONNX、Torch-free 推理和本地页面。18 项测试通过，单／多进程 Pilot 输出一致；6 workers 正式预处理运行中。正式训练、分层评估、部署计时及页面端到端验收尚待完成。

- **2026-10-02（全量数据验收）**：5000 输入完成核算：4946 成功、3 标签异常、51 不支持的缺失相机 WB 输入（DCS460D 28、PowerShot S70 17、EOS D30 6）。逐文件 LibRaw header 复核并保留原始失败报告，不改变颜色算法或启用隐式 AWB；未知错误仍阻断。正式互斥划分固定 3957／495／494，全量训练启动。

- **2026-10-02（正式模型与本地 MVP 验收）**：冻结骨干保持不变，正式双分支验证／测试宏平均归一化 MAE 0.066111／0.067113，常量基线 0.081530／0.082541。曝光测试 MAE 0.281 EV，高光恢复 7.428；只有这两项通过验证建议门槛，另外四项实验。相机／Catalog 标签／长尾、3395 条可得 As-Shot 基线覆盖评估已记录。FP32 ONNX 18.30 MiB，框架最大误差 3.21e-7，CPU p95 19.39ms，未量化。22 项测试、真实页面样例、上传与 JSON 下载通过；样例本地任务约 2.00 秒、RAW→结果约 0.69 秒。忠实渲染、现代流程映射和业务误差标定仍未完成。

- **2026-10-02（真实浏览器闭环）**：完成 DNG 文件上传→参数生成→实际 JSON 下载并核对字段／模型版本；替换输入清除旧结果，损坏 DNG 显示错误且不保留旧参数。应用恢复可用样例并保持本地运行。验收报告、JSON 示例、页面截图与精确源码快照保存于 artifacts/model。

- **2026-10-02（网页近似效果预览）**：新增应用前／近似应用后并排图片及 224×224 PNG 下载，只模拟已验证的曝光和高光软压缩，实验参数不应用；每次从基准图计算，不叠加。灰阶数值检查覆盖零值恒等、曝光方向、高光单调／暗部不变、非法参数拒绝与重复渲染；页面集成测试验证两幅图片和两种下载，2 项针对性测试通过。忠实显影 Gate 未完成。浏览器权限被用户拒绝，未进行新版页面的真实浏览器验收，需要用户自行刷新查看。

- **2026-10-02（预览改进计划与实施）**：按 PREVIEW_IMPROVEMENT_PLAN.md 加入独立线性 RAW 预览源，最长边 1600、等比显示；加入单调高光保护、0–100% 强度、实际参数／剪裁诊断及 PNG／JSON 对齐。25 项测试通过，禁用 Torch／网络的 RAW 预览检查通过；沙漠／暗场样例新增通道满值比例从 1.3167%／1.7179% 降到 0%，前景有所提亮；3 张固定样例已离线检查，无明显光晕，比例与细节改善。模型、特征和数据版本不变。整图语义已经参与参数预测；区域语义调色列为后续单独验证。真实浏览器与忠实 Lightroom 验收保持未完成。

- **2026-10-02（页面更新诊断）**：确认旧服务仍在运行且 runOnSave=false；重启当前项目服务以清除旧进程缓存，启用 runOnSave=true／fileWatcherType=poll。页面顶部新增 linear-raw-protected-v2 版本标识。新服务启动并监听 127.0.0.1:8501，模块导入及配置读取验证通过；用户需重新生成样例或上传结果，刷新本身不会运行 RAW 推理。未绕过浏览器保存的权限限制。

- **2026-10-02（多场景预览抽查）**：完成 12 张真实 DNG：8 张训练集场景抽查＋4 张未参与训练的固定测试集照片，包含人像、暗室、夜景、逆光、雪山、日落、织物和湖景。使用当前模型和保护高光，生成基准／100%／75% 对比及逐图 JSON；12 张零强度 PNG 逐字节恒等，两档均无新增通道满值。离线可视检查发现人像／暗室／夜景可见度改善，逆光主体提升有限，原过曝云层压暗后仍平坦发灰；未把数值剪裁下降当作纹理恢复。报告见 artifacts/preview_variety/REVIEW.md；不改变模型或渲染器，不据小样本宣称整体准确率。

- **2026-10-02（预览 v3：连续肩部与上限诊断）**：先记录计划，保留 v2 源码和原 12 张对比，再实现 linear-raw-shoulder-v3。正常中间调保持线性曝光，高光肩部连续且单调，代理压缩起点移到 0.8；页面／JSON 加入显影后显示源通道上限及全白比例，明确不等于传感器过曝或细节恢复。26 项测试通过；12 张回归＋4 张未查看随机测试集图片全部完成新旧对比，100%／75% 均无新增通道满值。人像／雪地／树干中间调改善、云层灰块减轻，逆光及已丢失纹理问题仍保留。保存 artifacts/preview_v3 的源码、图片、报告及复现脚本；模型与预处理不变。本地服务已重启加载新版，真实浏览器验收未执行。

- **2026-10-02（GitHub 版本管理准备）**：用户指定 git@github.com:Evannn888/ShotSense.git 并授权上传。只读核对目标为空、SSH 可用；初始化版本管理并排除原始数据、虚拟环境、私密配置、训练缓存和重复 PNG。纳入正式模型、源码、验收报告、开发记录、对比 JPG 与历史源码快照；README 记录仓库范围及克隆后的数据要求。推送结果以 Git remote 核对为准。
