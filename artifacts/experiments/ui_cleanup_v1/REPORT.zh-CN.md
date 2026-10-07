# ShotSense 页面整理 · 2026-10-06

主页面现在按「上传 → 对比 → 保存」排列。处理方式和场景/白平衡实验开关放到默认收起的侧栏；微调、细节查看、处理说明和 JSON 默认折叠。自动模式和自然调色模式共用结果布局，保留原来的手动及实验功能。

主要保存按钮直接把 PNG 写入本机 Downloads；浏览器下载在「Other download options」中。实际浏览器检查发现准备下载后折叠区会收起，已改为在按钮回调中记录准备状态，使 PNG 按钮保持可见。

## 验证

- 最终全套测试：91 passed，0 failed，0 skipped；1 条原有的可选 Matplotlib 警告。日志：`full_tests.log`。
- 真实 IAB 上传用户提供的 `images (2).jpeg`，准备并点击 PNG 下载；得到 678 × 452 PNG，其文件字节和像素与当前默认 worker/render 输出一致。细节：`browser_download_verification.json`。
- 浏览器直接下载文件：`/Users/evanchen/Downloads/shotsense-natural-color-s100-678x452-da7fcbc5.png`。本机主要保存按钮另由现有 AppTest 验证，未声称本次在真实浏览器点击过该按钮。
- 桌面 1280 × 900 检查到标题顺序为 ShotSense、1 · Add a photo、2 · Compare the result、3 · Save your photo，无横向溢出。窄窗口检查了上传、结果、保存区，图片使用原生响应式排列；临时浏览器视口已恢复。
- 截图：`home-desktop.jpg`、`result-desktop.jpg`、`home-narrow.jpg`、`result-narrow.jpg`。
- 本次生产代码仅修改 `app/streamlit_app.py`；基线中的处理器、模型和锁定依赖哈希保持一致。没有改变照片算法和默认模式；场景/白平衡实验仍默认关闭。本记录是界面和功能验证，不是新一轮照片质量验收。

`before.py` 和 `after.py` 保存本次前后页面源文件；`baseline.json` 与 `verification.json` 保存核验结果。首次下载区可见性问题和修复前测试日志也保留。
