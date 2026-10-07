# ShotSense 自动调整的数据集选择 · 2026-10-05

最合适的起点是补齐现有 FiveK 的人工场景属性和专家成品图，再用 MSEC 扩展曝光变化、人像用 PPR10K、偏色用 Rendered WB。当前无需为了连接预训练模型重新训练一个大场景分类器。

这次核对发现，之前“缺数据”的说法应具体化：本地已使用 RAW 和专家绝对参数，但原站还提供人工场景属性及完整专家像素目标。资源足以开始配对监督实验；目标下载、颜色转换、对齐审计和独立效果验证仍待完成。

## 第一优先：复用 FiveK 的完整资源

[FiveK 原始数据页](https://data.csail.mit.edu/graphics/fivek/)提供 5,000 张 DNG、五位修图师的 16 位 ProPhoto RGB TIFF，以及主体、光源、室内外、时段四组人工属性。它同时支持粗场景检查和修图目标研究。

本轮只读取公共元数据：提取 5,000 条完整记录及每条五个专家目标 URL。用完整源文件名、统一大小写及 .dng 后缀与现有清单连接，无模糊或仅数字 ID 匹配；本地 4,946 张全部对应，原有训练／验证／测试的 3,957／495／494 分组未变。元数据与匹配摘要已保存，没有下载新的原图或专家目标像素。

| 本地支持照片的人工属性 | 数量 |
|---|---:|
| 人物主体 | 1,332 |
| 自然主体 | 1,064 |
| 动物主体 | 296 |
| 人造物主体 | 1,857 |
| 混合光 | 826 |
| 室内 | 1,051 |
| 黎明／黄昏 | 203 |
| 夜间 | 129 |
| 时段未知 | 1,358 |

各行不是互斥分类，不能相加当作照片总数。动物标签不等于宠物，人物标签不等于脸部或肤色掩码；人造物也不等于纯建筑。标签没有说明“需要提亮多少”或剪影是否刻意，保留 unknown 与原始词义。

原站 HTTP200；一个专家 C TIFF 的 HEAD 请求也为 HTTP200、Content-Type=image/tiff、31,609,888 字节，未读取像素。旧 [Image-Adaptive-3DLUT 的 480p 数据分享](https://github.com/HuiZeng/Image-Adaptive-3DLUT) Google Drive 文件夹仍为 HTTP404，不能把旧链接失败理解为整个 FiveK 不可获取。专家目标应从原站按需取少量，审计 ICC、裁剪／尺寸／对齐后再转换成与 JPEG 实验一致的 sRGB。

## 配套数据集

| 数据集与作者来源 | 能提供什么 | 在 ShotSense 中的用法与边界 |
|---|---|---|
| [MSEC / Exposure Correction](https://github.com/mahmoudnafifi/Exposure_Correction) | [原论文](https://openaccess.thecvf.com/content/CVPR2021/papers/Afifi_Learning_Multi-Scale_Photo_Exposure_Correction_CVPR_2021_paper.pdf)报告 24,330 个 8 位 sRGB 曝光版本；由 FiveK RAW 经 Camera Raw SDK 渲染相对 -1.5、-1、0、+1、+1.5 EV，专家 C 成品作目标。 | 最贴近“曝光变化后如何控制调整”的开发数据。24,330 是变体数，并非独立原始场景数；0 EV 是源拍摄曝光，也不保证理想。与已有 FiveK 重叠，统一按原图／场景分组，不能当作新的外部测试集。它支持曝光／色调修正研究，不能证明真实低光噪声还原能力。 |
| [PPR10K](https://github.com/csjliang/PPR10K) | 11,161 张 RAW 人像、1,681 组、三位修图师的目标、人体区域掩码和源／目标 XMP。作者列出完整资源约 406GB，360p TIFF 包约 91GB。 | 补人像和同组照片一致性。掩码是人体区域，不能直接当肤色精确分割。先少量组、单一专家风格；不要为了首版连接就下载整个包。XMP 导入旧 RAW 参数模型需要另做版本／字段／完整配方审计。 |
| [Rendered WB](https://github.com/mahmoudnafifi/WB_sRGB)、[Deep White-Balance Editing](https://github.com/mahmoudnafifi/Deep_White_Balance) | 作者当前发布说明列 105,638 个渲染图像，包含错误白平衡输入、正确白平衡目标及元数据；原论文所用约 65K 是较早范围。 | 专门研究 JPEG 偏色校正和色彩稳定性，不能把发布数量视为独立场景数。注意同场景／相机／渲染变体分组及色卡像素；优先作者提供的去色卡版本。全局白平衡目标不能指导保留混合光或蓝橙布光的审美决策。各版本来源及与 FiveK 的重叠需按元数据核对。 |
| [LCDP](https://github.com/onpix/LCDPNet) | 成对的局部曝光问题输入和参考 PNG；针对一张照片同时存在过亮、过暗区域，作者数据采用 RAW／ISP 渲染。 | 适合逆光、亮窗与暗前景的诊断，帮助判断当前全局 LUT／逐像素曲线的能力边界。参考目标不是场景标签或当前控制参数。仓库已有数据列表；归档的场景来源、数量、编码与分组尚未审计，本轮不宣称它是独立新数据。 |
| [LOL](https://daooshee.github.io/BMVC2018website/)、[LOL-v2 作者说明](https://github.com/flyywh/CVPR-2020-Semi-Low-Light) | 低光／正常光配对；v1 是 485 个训练对、15 个保留评估对，v2 有单独的真实与合成部分。 | 检查暗部可见度、噪声放大和色偏。项目已保留 v1 作者归档及 120 对开发数据，先复用；保持 eval15 像素不访问。提升可见度不等于自然艺术调色，不要把所有正常光参考都当成通用审美目标。外部模型可能已训练过这些照片。 |
| [SICE](https://github.com/csjcai/SICE) | 589 个多曝光序列、4,413 张图；参考由多曝光融合／HDR 算法结果经主观筛选产生。 | 后续高动态范围／曝光压力测试。每个场景的全部曝光放在同一组；单图模型只能看到一个输入。融合参考含多次曝光的信息，本项目的单图处理无法保证恢复那些细节；不优先作为整体自然调色风格训练。 |

上表是基于数据目标与本项目接口作出的用途判断，不是我们已经在这些数据上测得更好画质。分别保留曝光、白平衡、人像、审美任务，避免将不同参考目标直接混成一套风格。

## 当前访问证据

仅进行了小体积公开页面请求或 HEAD，没有下载数据归档。HTTP200 代表列表／预览页可访问，不代表归档可完整下载、解压、匹配或通过颜色／画质审计。

| 作者发布入口 | 本轮证据 |
|---|---|
| [FiveK 原站](https://data.csail.mit.edu/graphics/fivek/) | HTTP200；完整 5,000 条属性及五位专家 URL；一张专家 C TIFF HEAD200。 |
| [旧 480p 分享文件夹](https://drive.google.com/drive/folders/1Y1Rv3uGiJkP6CIrNTSKxPn1p-WFAc48a?usp=sharing) | HTTP404。 |
| [MSEC 训练镜像](https://drive.google.com/file/d/1YtsTeUThgD2tzF6RDwQ7Ol9VTSwqFHc_/view?usp=sharing) | HTTP200，标题 training.zip；作者 Sync 入口也返回 training.zip 页面。 |
| [PPR10K 数据入口](https://drive.google.com/drive/folders/1kB2OSAGy8uc0xUXaMKoPB0HMSc-rkrLW?usp=drive_link) | HTTP200，数据文件夹列表可见；未经全部子文件访问／归档验证。 |
| [LCDP 数据入口](https://drive.google.com/drive/folders/10Reaq-N0DiZiFpSrZ8j5g3g0EJes4JiS?usp=sharing) | HTTP200，标题 lcdpnet；未验证大归档。 |
| [Rendered WB JPEG 镜像](https://github.com/mahmoudnafifi/WB_sRGB#dataset) | 作者链接的公开 Drive 页 HTTP200，标题 Set1_input_images_JPG.zip。仍需对应目标及元数据。 |
| [LOL-v2](https://drive.google.com/file/d/1dzuLCk9_gE2bFF222n3-7GVUlSVHpMYC/view?usp=sharing) | HTTP200，标题 LOL-v2.zip；未解压检查具体版本／划分。 |

访问请求、重定向、返回标题和取样响应哈希保存在 access_checks.json。九份作者仓库 README 元数据快照见 source_manifest.json；FiveK 官网 HTML 与标注摘要另存。原论文读取遇到的部分抓取失败，不算数据不存在，也不影响上述独立 HTTP 证据。

## 新近大型数据的处理

[VeraRetouch 的 AetherRetouch-1M+](https://arxiv.org/html/2604.27375v1)在论文中提出百万级自动／风格／参数调色配对，包含从高质量照片反向生成未修图版本、预设风格和参数扰动；专家参考来自 FiveK/PPR10K。它的合成思路值得借鉴，数量不能解释为百万次人工逐张修图。

截至本轮，作者 [仓库](https://github.com/OpenVeraTeam/VeraRetouch)明确链接模型权重和示例，但本轮没有找到可核实的完整作者数据归档；对作者 Gyh68 和 AetherRetouch 名称的公开 Hugging Face 数据集 API 查询返回空。这个限定检查不能证明所有平台均未发布。暂列研究线索，避免把论文规模当作现在已能获取的数据。模型可用性与完整训练数据可用性分开记录。

## 最适合现在的执行顺序

1. **先补现有 FiveK。** 公开属性与本地清单已经能连接；先从现有训练／开发验证组选约 100 个代表场景，单独建立不可覆盖的获取清单，下载专家 C 的少量成品。选 C 是便于对齐 MSEC 的开发选择，不能称为唯一正确审美。保留之前测试组和所有原图／Catalog。
2. **冻结颜色与配对契约。** 对照原图、正常开发图和专家 TIFF 的裁剪／方向／尺寸／ICC，保留原始 16 位目标，另生成有来源记录的 sRGB 训练版本。无法正确对齐的配对隔离；不要以自己的近似参数渲染当专家成品。
3. **用 MSEC 补曝光维度。** 先验证少量完整场景的不同曝光变体；沿用 FiveK 原图／相关场景分组，同场景不能跨集合。因为预训练 LUT 的 FiveK 数据重叠可能存在，结论限定为开发诊断。
4. **让目标图监督我们自己的决策。** 固定当前渲染器，生成少量受约束的调色／基础提亮／暗部提升候选，与专家参考及原图比较。选出的控制量是本工具的候选选择标签，不是旧 CameraRaw 绝对参数。若当前全局工具表达不了专家局部效果，记录能力不足，不为贴近目标无限增加强度。
5. **按失败再扩数据。** 人像色偏优先 PPR10K；全局偏色优先 Rendered WB；亮暗共存优先 LCDP；极暗问题先用已在本地的 LOL 做风险回归。正常曝光、剪影、夜间布光和混合光仍需“保留原有氛围”的真实偏好对照。

这些是建议的开发顺序与起步规模，没有执行新图片获取、训练、参数搜索或网页集成。公开集的已看样本会作为开发数据；最后仍需新的相机／手机照片及人工偏好检查。

## 个人研究范围与状态

FiveK 的两份原始 [数据协议](https://data.csail.mit.edu/graphics/fivek/legal/LicenseAdobe.txt)、[联合协议](https://data.csail.mit.edu/graphics/fivek/legal/LicenseAdobeMIT.txt)允许自身研究用途并要求保留许可与署名；PPR10K 和 Rendered WB 也明确提供非商业研究使用。你的当前个人研究方向可以考虑这些资源。获取时记录各数据包自己的协议，仓库代码许可不自动替代数据协议；未在入口明确区分的数据条款留待归档审计。

- 已完成：作者来源／用途／当前公开入口检查，FiveK 5,000 条公开属性与 4,946 本地支持源的精确匹配，任务／重叠／版本边界和获取优先级建议。
- 未完成：新专家目标或大归档下载、完整源／目标对齐与编码审计、决策层训练、独立识别／调色画质验收。
- 当前程序、模型、数据划分和依赖保持不变；不加入锐化。
