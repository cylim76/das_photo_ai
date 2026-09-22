# DAS Photo AI

DAS Photo的独立OCR推理与业务字段提取服务。

当前版本为`0.4.1`：Windows可以使用Mock/JSON进行开发测试；P3可以通过独立Docker容器使用PaddleOCR GPU直接识别图片，并在通用OCR缺少箱号校验位时调用英文识别模型和上下文复核完成后备识别。

- [v1开发说明书](docs/development_plan_v1_20260922.md)
- [v0.2.0评估基线](docs/evaluation_v0.2.0_20260922.md)
- [v0.2.0 P3端到端评估](docs/p3_evaluation_v0.2.0_20260922.md)
- [P3部署说明](docs/p3_deployment_v0.2.0_20260922.md)
- [0.4.0 P3升级与回归说明](docs/p3_upgrade_v0.4.0_20260922.md)
- [0.4.1 P3复核升级说明](docs/p3_upgrade_v0.4.1_20260922.md)

## 当前能力

- Mock OCR引擎；
- 读取现有PaddleOCR JSON；
- PaddleOCR CPU/GPU图片推理引擎；
- 图片上传API；
- OCR模型单例加载和并发保护；
- 箱号提取、ISO 6346校验和校验位推算；
- 区分`verified`、`unverified`和`mismatch`；
- 铅封号候选提取与噪声过滤；
- 铅封号原图、顺时针90度、逆时针90度自动识别与候选融合；
- 箱号校验位区域裁剪、放大对照测试工具；
- 箱号校验位单字符直接识别、去框预处理和多变体安全融合实验工具；
- 正式图片API中的英文校验位后备识别、模型复用和审计信息；
- 英文单字符未验证时的上下文2/4倍及灰度通用OCR复核；
- P3 GPU Docker部署配置；
- 离线回归和自动测试。

## Windows开发环境

业务代码兼容Python 3.10～3.14，普通环境推荐Python 3.12。

```powershell
cd D:\RPA\das_photo_ai
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements-dev.txt
.\.venv\Scripts\python -m app
```

如果本机只有其他兼容版本，可将`py -3.12`换成`py -3`。Windows不需要安装PaddleOCR即可运行Mock/JSON接口和测试。

默认地址：

- 服务：`http://127.0.0.1:8800`
- API文档：`http://127.0.0.1:8800/docs`
- 健康检查：`http://127.0.0.1:8800/api/v1/health`
- 引擎状态：`http://127.0.0.1:8800/api/v1/models`

## JSON识别

```powershell
.\.venv\Scripts\python -m app.cli `
  D:\RPA\ai-lab\ocr-eval-20260922\container_test\F119755_res.json `
  --target container_number
```

API路径：

```text
POST /api/v1/recognize
```

## 图片识别

API路径：

```text
POST /api/v1/recognize/image
```

Linux示例：

```bash
curl -X POST http://127.0.0.1:8800/api/v1/recognize/image \
  -F 'engine=paddle_gpu' \
  -F 'targets=container_number,seal_number' \
  -F 'image=@/path/to/photo.jpg'
```

Windows没有PaddleOCR时可以用Mock验证上传流程：

```powershell
curl.exe -X POST http://127.0.0.1:8800/api/v1/recognize/image `
  -F "engine=mock" `
  -F "targets=container_number,seal_number" `
  -F "image=@D:\path\to\photo.jpg"
```

## 校验位结果

如果图片只识别到前10位：

```json
{
  "value": "HASU506039",
  "observed_value": "HASU506039",
  "suggested_value": "HASU5060396",
  "observed_check_digit": null,
  "calculated_check_digit": "6",
  "verification": "unverified"
}
```

如果图片识别到完整11位，并且校验位一致：

```json
{
  "observed_value": "MSCU6639870",
  "suggested_value": "MSCU6639870",
  "observed_check_digit": "0",
  "calculated_check_digit": "0",
  "verification": "verified"
}
```

`suggested_value`不能当成图片完整识别结果；DAS Photo接入时必须向操作人员显示验证状态。

## P3 Docker部署

使用已验证的基础镜像：

```text
paddlepaddle/paddle:3.3.0-gpu-cuda12.9-cudnn9.9
```

在P3项目目录执行：

```bash
docker compose build
docker compose up -d
docker compose logs -f das-photo-ai
```

Compose默认：

- 使用`paddle_gpu`；
- 使用`gpu:0`；
- 启动时预加载模型；
- 预加载`en_PP-OCRv5_mobile_rec`校验位专用模型；
- 通用OCR只得到箱号前10位时自动运行校验位后备识别；
- 只绑定P3本机`127.0.0.1:8800`；
- 复用`/home/lucas/ai-lab/cache`下现有模型缓存；
- 不配置永久代理；
- 不影响现有MySQL容器。

详细步骤见[P3部署说明](docs/p3_deployment_v0.2.0_20260922.md)。
从0.3.0升级并运行正式API回归见[0.4.0 P3升级与回归说明](docs/p3_upgrade_v0.4.0_20260922.md)。

## 测试

```powershell
.\.venv\Scripts\python -m pytest
```

历史JSON回归：

```powershell
.\.venv\Scripts\python -m scripts.evaluate_existing_ocr `
  D:\RPA\das_photo\.codex-tmp\ocr_eval\ocr_eval_data.json `
  --ocr-root D:\RPA\ai-lab\ocr-eval-20260922 `
  --show-errors
```

正确答案只用于离线评估，运行时API不会读取DAS Photo数据库。

先在Windows从已有评估数据生成不含本地路径的精简答案清单：

```powershell
.\.venv\Scripts\python -m scripts.create_evaluation_manifest `
  D:\RPA\das_photo\.codex-tmp\ocr_eval\ocr_eval_data.json `
  D:\RPA\ai-lab\ocr-eval-20260922\ocr_eval_manifest_v1_20260922.json
```

将清单复制到P3的`/home/lucas/ai-lab/`后，执行真实图片端到端评估：

```bash
python3 -m scripts.evaluate_image_api \
  --report /home/lucas/ai-lab/ocr_eval_manifest_v1_20260922.json \
  --container-root /home/lucas/ai-lab/container_test \
  --seal-root /home/lucas/ai-lab/sealno_test \
  --api-base http://127.0.0.1:8800/api/v1 \
  --engine paddle_gpu \
  --output-dir /home/lucas/ai-lab/evaluation-runs/v0.2.0-baseline
```

程序逐张调用图片接口，并保存`summary.json`、`details.json`、可续跑的
`details.ndjson`及每张图片的原始API响应。中断后在相同命令末尾增加
`--resume`即可跳过已经完成的图片；增加`--resume --retry-failures`可重试失败项。

如果测试机本身保存DAS Photo数据库，也可以用`--database 数据库路径`代替
`--report 答案清单路径`；数据库始终按只读模式打开。

### 铅封号三方向对照测试

三方向测试不会修改服务或原始照片。它将每张铅封照片以内存临时文件的形式生成原图、
顺时针90度和逆时针90度三个版本，并通过现有图片API分别识别。P3可直接借用已经构建
好的服务镜像运行脚本，无需在宿主机安装Pillow，也不会停止或重启正在运行的服务：

```bash
cd /home/lucas/rpa/das_photo_ai
sudo docker run --rm \
  --network host \
  --user "$(id -u):$(id -g)" \
  --entrypoint python \
  -v "$PWD":/workspace:ro \
  -v /home/lucas/ai-lab:/home/lucas/ai-lab \
  -w /workspace \
  das-photo-ai:0.2.0 \
  -m scripts.evaluate_seal_orientations \
  --report /home/lucas/ai-lab/ocr_eval_manifest_v1_20260922.json \
  --seal-root /home/lucas/ai-lab/input/sealno_test \
  --api-base http://127.0.0.1:8800/api/v1 \
  --engine paddle_gpu \
  --output-dir /home/lucas/ai-lab/evaluation-runs/v0.2.0-seal-orientations
```

输出目录包含`summary.json`、逐图`comparison.csv`、`details.json`和三个方向各自的原始
API响应。测试中断后，在同一命令末尾增加`--resume`即可继续。

### 箱号校验位局部OCR对照测试

箱号优化与铅封方向优化是两条独立验收线。下面的程序先识别原图，再根据图片中已经
观察到的箱主代码和序列号坐标，生成局部2倍、4倍和彩色转灰度4倍临时裁剪图，分别调用
现有API。它不会用ISO计算值冒充图片观察到的第11位，也不会修改原图、数据库或主服务：

```bash
cd /home/lucas/rpa/das_photo_ai
sudo docker run --rm \
  --network host \
  --user "$(id -u):$(id -g)" \
  --entrypoint python \
  -v "$PWD":/workspace:ro \
  -v /home/lucas/ai-lab:/home/lucas/ai-lab \
  -w /workspace \
  das-photo-ai:0.3.0 \
  -m scripts.evaluate_container_check_digit \
  --report /home/lucas/ai-lab/ocr_eval_manifest_v1_20260922.json \
  --container-root /home/lucas/ai-lab/input/container_test \
  --api-base http://127.0.0.1:8800/api/v1 \
  --engine paddle_gpu \
  --output-dir /home/lucas/ai-lab/evaluation-runs/v0.3.0-container-check-digit
```

结果重点查看`summary.json`中的`baseline_verified_exact`、
`selected_verified_exact`和`rescued_verified_exact`。只有局部OCR真正读到完整11位且
ISO 6346校验通过，才计为`verified_exact`。

### 箱号校验位倾斜矫正与单字符定向测试

`v032`配置用于重测上一轮仍未验证的照片。它在箱号横带上下各增加15%余量，增加轻微
倾斜矫正，并把右侧校验位单独裁成一张小图进行OCR。程序只允许拼接OCR实际返回的单个
数字；ISO 6346只负责验证，不会生成或选择图片中没有识别到的数字。

```bash
cd /home/lucas/rpa/das_photo_ai
sudo docker run --rm \
  --network host \
  --user "$(id -u):$(id -g)" \
  --entrypoint python \
  -v "$PWD":/workspace:ro \
  -v /home/lucas/ai-lab:/home/lucas/ai-lab \
  -w /workspace \
  das-photo-ai:0.3.0 \
  -m scripts.evaluate_container_check_digit \
  --profile v032 \
  --retry-summary /home/lucas/ai-lab/evaluation-runs/v0.3.1-right30/summary.json \
  --report /home/lucas/ai-lab/ocr_eval_manifest_v1_20260922.json \
  --container-root /home/lucas/ai-lab/input/container_test \
  --api-base http://127.0.0.1:8800/api/v1 \
  --engine paddle_gpu \
  --save-crops \
  --output-dir /home/lucas/ai-lab/evaluation-runs/v0.3.2-check-digit
```

程序从上一份`summary.json`的`still_unverified_or_wrong`读取文件名，因此本次只测试剩余
样本。生成的裁剪图保存在`output-dir/crops`，原始API响应保存在`raw`，拼接后的结果保存
在`postprocessed`。重点检查`isolated_check_digit_detected`、`verified_exact`、
`selected_false_verified`以及新的`still_unverified_or_wrong`。

### 箱号校验位直接识别测试

`v0.3.2`在剩余12张照片中救回6张，组合累计达到83/89。未救回的6张校验位图肉眼
可见，但OCR文字检测没有返回任何文字，其中方框和窄字符`1`可能影响检测。下面的实验
绕过文字检测模块，直接使用PaddleOCR的`TextRecognition`识别已经裁好的校验位小图。

每张图会生成三种输入：保留方框的彩色图、保留方框的灰度增强图、尝试去掉方框的灰度
增强图。程序只接受OCR实际输出的单个数字；两个以上变体一致时采用多数结果，只有一个
有效结果时必须达到较高置信度，变体冲突则保留人工确认。ISO 6346只在数字选定后验证，
不会参与选择或补出数字。

```bash
cd /home/lucas/rpa/das_photo_ai
sudo docker run --rm \
  --runtime=nvidia \
  --gpus all \
  --entrypoint python \
  -v "$PWD":/workspace:ro \
  -v /home/lucas/ai-lab:/home/lucas/ai-lab \
  -v /home/lucas/ai-lab/cache/paddlex:/root/.paddlex \
  -v /home/lucas/ai-lab/cache/paddleocr:/root/.paddleocr \
  -w /workspace \
  das-photo-ai:0.3.0 \
  -m scripts.evaluate_check_digit_recognition \
  --previous-run-dir /home/lucas/ai-lab/evaluation-runs/v0.3.2-check-digit \
  --device gpu:0 \
  --model-name PP-OCRv5_server_rec \
  --output-dir /home/lucas/ai-lab/evaluation-runs/v0.3.3-direct-recognition

sudo chown -R lucas:lucas \
  /home/lucas/ai-lab/evaluation-runs/v0.3.3-direct-recognition
```

这一步是独立评估，不修改正在运行的API服务，也不需要`--network host`。模型已经存在于
宿主机缓存时不需要联网；如果缓存里缺少`PP-OCRv5_server_rec`，PaddleOCR会尝试联网下载。
输出重点查看`direct_recognition_rescued`、`selected_false_verified`、
`selection_statuses`以及`still_unverified_or_wrong`，同时可在`crops`中查看三种实际输入图。

P3使用通用`PP-OCRv5_server_rec`时救回5/6；改用官方英文模型
`en_PP-OCRv5_mobile_rec`后救回6/6，`inner_gray`变体6张全部正确且平均置信度为
0.990724。固定89张开发集由此累计达到89/89，错误验证为0。0.4.0已把该策略接入正式
图片API，但此成绩仍需通过完整API回归和新照片盲测确认，不能直接当作生产准确率。

## 模型文件

模型权重和缓存不提交Git。容器通过宿主机目录挂载保存模型，重建容器不需要重新下载已有模型。
