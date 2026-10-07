# 新照片场景识别验证 · 2026-10-05

这轮支持继续采用 MobileCLIP2-S0 作为主体识别模块，但不支持让它直接决定曝光或套用某个 LUT。8 张新选照片的粗粒度主体判断均符合预先记录的画面观察；光线判断在低调人像、蓝调城市和逆光场景中存在误导风险。它能回答“画面主要是什么”，还不能可靠回答“摄影师希望怎么修”。

本轮验证识别与运行一致性，没有调色、锐化、训练或修改当前网页。以下“应该保留／保护”是下一步处理策略建议，尚未证明新的照片调整效果。

![8 张照片与全画面识别结果](</Users/evanchen/CS Projects/ShotSense/data/user_photo_diagnostics/scene-analysis-public-v1/recognition_results.png>)

## 逐张结果

| 照片 | 主体判断，两种取景一致 | 光线判断：中心裁剪 → 全画面 | 分析与处理含义 |
|---|---|---|---|
| 白天海滩人像 | 人像 | 阴天 → 白天自然光 | 主体合理。两种光线描述都符合亮灰天空和柔和自然光；排名差很小，不宜强分阴天／晴天后大幅改色。 |
| 蓝橙布光人像 | 人像 | 白天欠曝 → 白天欠曝 | 关键失败案例。画面有明显蓝橙方向光和黑背景，低调氛围合理；拍摄地点／时刻未知。欠曝标签不能证明曝光错误，也不能成为自动抹去蓝橙光或强行提亮的依据。 |
| 蓝调城市全景 | 城市建筑 | 白天欠曝 → 白天欠曝 | 主体合理。天空仍有蓝色天光，建筑灯也亮着；虽然原文件名写着夜景，画面更接近日暮／蓝调时段。模型欠曝判断无法确认摄影意图。现有类别缺少蓝调时段。 |
| 银河星空 | 自然风景 | 夜景 → 夜景 | 夜景判断贴合星空。适合给自动提亮增加保守约束；不能把天空提成白天。红色前景光也不等于整体白平衡错误。 |
| 晚霞与树剪影 | 自然风景 | 日出／日落 → 日出／日落 | 识别合理。黑色前景是剪影，亮度低不代表需要救回所有暗部。类别没有区分日出／日落或判断剪影意图。 |
| 餐厅室内混合光 | 室内空间 | 室内人工光 → 室内人工光 | 主体合理，但窗外天光和室内灯同时存在。人工光与自然光的相似度差仅 0.0083／0.0123；不能按单一光源全局校正色温。 |
| 雾中逆光树林 | 自然风景 | 白天自然光 → 白天欠曝 | 取景敏感案例。树林、雾气和阳光束都清楚；全画面模型却把欠曝排在首位，差仅 0.0089。应保护光束与高光层次，不能把正常逆光全局拉平。 |
| 户外幼犬 | 宠物 | 白天自然光 → 白天自然光 | 主体与光线判断合理。它并未识别宠物局部明暗或毛发区域，暂不能据此声称局部调整已经可用。 |

本轮 8 张主体标签在两种取景下均符合预先声明的可接受标签；这只是有限样本中的观察，不能称为总体准确率 100%。覆盖了人像、自然风景、室内空间、城市建筑、宠物五种粗类别，尚未覆盖食物、静物、文档或各类别内部的复杂构图。8 张照片中有 2 张的光线首位标签随取景改变：白天人像、逆光树林。前者可接受，后者可能造成不合适的提亮决策。两次重复运行的全部 cosine 分数完全一致，说明这些差异来自输入取景，重复一致不代表判断正确。

## 特意暗化后会不会变成“夜景”

![两组固定暗化对照](</Users/evanchen/CS Projects/ShotSense/data/user_photo_diagnostics/scene-analysis-public-v1/darkening_results.png>)

两张对照都在运行前确定：将白天人像和逆光树林的 sRGB 转为线性显示 RGB，乘以 1/8，再转回 8 位 sRGB。这是约 3 档的人工显示亮度降低，不是实拍相机欠曝，也没有模拟噪声、动态范围或真实细节损失。

| 对照 | 全画面光线判断 | 结果解释 |
|---|---|---|
| 白天人像 → 固定暗化 | 白天自然光 → 阴天 | 仍识别人像，两种取景都没有误判夜景。但它也没有将“白天欠曝”排在首位，无法衡量该补多少曝光。线性亮度中位数由 0.5080 降至 0.06345，约为原来的 1/8。 |
| 逆光树林 → 固定暗化 | 白天欠曝 → 白天欠曝 | 仍识别风景，两种取景都没有误判夜景。原图和暗化图标签相同，无法据此区分原有逆光氛围与实际暗化程度。中位数由 0.2322 降至 0.02905。 |

这两个对照是有用的正面结果，但不能证明其他暗照片不会被误判。更关键的发现是：场景相似度不能代替亮度测量，更不能自动推断摄影师是否希望提亮。

## 原生人脸检测与速度

macOS Vision 人脸请求 revision 3 在两个主要人像及暗化人像中各检测到 1 个脸框，其余输入为 0；联系图绿色框与人物脸部位置相符。餐厅中远处的人没有被检测到。这里只验证了几例定位，未测量一般人脸检测精度，更没有验证肤色或人脸分割。

原生分类 revision 2 的高排名包含帽子、衣服、液体、砖块等标签。餐厅把人排得很高，宠物图把材质／砖块排在动物之前；这些细粒度标签不能直接作为本项目粗粒度的场景主类别。保留原生人脸位置作为补充证据较合适。

当前 Mac 上，CPU 单线程 ONNX 第二轮 20 次前向的中位数为 101.45 ms，95 分位为 105.57 ms；建立会话约 99.98 ms。它们分别是模型前向和会话建立时间，不包括下载、完整分辨率解码、调色、PNG 导出或网页响应。正式功能不一定要长期运行两种取景；取景策略仍需更多样本验证。

## 对实现方案的影响

建议继续采用“主体识别 + 图像量化分析 + 受约束的调整”，不要把光线首位标签接成一个直接选择曝光／LUT 的开关。

1. **主体识别用于保护。** 人像限制明显色偏，夜空保留暗天空，日落保留剪影和晚霞；这些保护规则仍需实际输出对比验证。主体标签与光线信息应允许重叠，例如“人像 + 人工布光 + 低调”。
2. **调整幅度由图像与候选输出共同约束。** 测量中间调、暗部、亮部及新产生的截断；暗部纹理变化只能作为噪声风险线索，不能当成真实信噪比。欠曝相似度可作辅助证据，不能单独触发大幅提亮。
3. **保留不确定状态。** 低调人像、混合光、蓝调时段和逆光没有可靠的单一答案。取景不一致或相似度接近时应减小调整幅度；本轮没有拟合“可信度阈值”，不能把 cosine 换成百分比置信度。
4. **下一轮验证实际调整。** 保留这 8 张作为已知回归案例，再找新的肤色、霓虹夜景、混合光、正常曝光照片；对比当前自动输出与加入保护约束后的输出，检查有没有损害原有氛围。尚未支持默认集成或广泛照片画质通过。

现有描述还缺少“人工布光人像”“蓝调时段”“逆光”“低调”等属性。是否增加这些属性应在下一轮预先冻结描述再验证；不能根据这批结果即时换提示词并宣称准确率提高。此次没有改变模型、15 个描述、文本向量、LUT 或处理强度。

## 输入与复现范围

- 使用上轮固定的 MobileCLIP2-S0 FP32 图像 ONNX、15 个文本描述向量；原始 cosine 及各组前两名保存在 results.json，均非校准概率。
- 8 张新选公共照片、2 张人工暗化对照、两种取景、两轮重复。两种取景是短边 256 后中心裁剪与全画面 256 灰色留边。全部 10 个中心裁剪像素与当前 torchvision 原处理逐字节一致。
- 4 张下载原文件并核对 Commons 字节数、SHA1 和 JPEG 内容；其余 4 个原文件返回 HTTP429，按图片源明确提供的替代方式取得 API 声明的 960px 公开预览，核对 JPEG 内容、尺寸和本地 SHA256。它们不是已验证的完整分辨率原文件。全部输入再制成最大边 512 的本地诊断图。
- 全部 8 张图在推理前目视检查并记录 expected_labels.json；不明确的光线单独说明，没有按文件名生成答案，没有因推理结果替换样本。下载限制和预览替代在推理前写入 PROTOCOL.md，保留初次失败记录。
- 这些照片此前未用于 ShotSense 的该诊断，但可能已出现在外部预训练数据中；本轮不宣称独立测试集或未知照片总体准确率。缩略图不用于验证原图画质、降噪、锐化、分辨率兼容或输出效果。
- 原文件／预览、对照、模型、文本向量、提示词和 7 个当前实现／依赖文件的完整性检查见 verification.json。网页处理逻辑没有修改。

## 照片来源与署名

所有照片来自 Wikimedia Commons。联系图做了缩小、排列、文字标注及人脸框标注；暗化对照另作已声明的亮度变换。联系图按 [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) 提供，各底层照片保留以下许可与署名。完整来源、当前文件页 revision、下载 URL、原始尺寸、输入类型和哈希见 inputs.json、commons_metadata.json 与 commons_preview_metadata.json。

| 照片来源 | 作者 | 许可 | 下载输入 |
|---|---|---|---|
| [Man in denim jacket posing](https://commons.wikimedia.org/wiki/File:Man_in_denim_jacket_posing_(Unsplash).jpg) | Brooke Cagle / brookecagle | [CC0](https://creativecommons.org/publicdomain/zero/1.0/) | 原文件 |
| [Shadowed Portrait](https://commons.wikimedia.org/wiki/File:Shadowed_Portrait_(Unsplash).jpg) | WillSpirit SBLN / willspirit | [CC0](https://creativecommons.org/publicdomain/zero/1.0/) | 原文件 |
| [The New York City Skyline at night](https://commons.wikimedia.org/wiki/File:The_New_York_City_Skyline_at_night.jpg) | Bruce Emmerling | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) | 原文件 |
| [Milky Way 3](https://commons.wikimedia.org/wiki/File:Milky_Way_3.jpg) | Dennis | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | 原文件 |
| [Joshua tree silhouette at sunset](https://commons.wikimedia.org/wiki/File:Joshua_tree_silhouette_at_sunset_(50330255608).jpg) | Joshua Tree National Park | Public domain | 960px 公开预览 |
| [The Scratch Kitchen Interior](https://commons.wikimedia.org/wiki/File:The_Scratch_Kitchen_Interior.jpg) | Xanadu73 | [CC0](https://creativecommons.org/publicdomain/zero/1.0/) | 960px 公开预览 |
| [Crepuscular rays in ggp 2](https://commons.wikimedia.org/wiki/File:Crepuscular_rays_in_ggp_2.jpg) | Brocken Inaglory | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) | 960px 公开预览 |
| [Labrador pup](https://commons.wikimedia.org/wiki/File:Labrador_pup.jpg) | Bobjack tom andy | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) | 960px 公开预览 |

运行顺序：prepare.py（公开预览使用 --published-previews）、目视检查／冻结 expected_labels.json、validate.py、summarize.py。已有完成的结果受到覆盖保护；它不是正式应用模块。
