# DAS Photo AI

DAS Photo的独立OCR推理与业务字段提取服务。

当前版本为`0.3.0`：Windows可以使用Mock/JSON进行开发测试；P3可以通过独立Docker容器使用PaddleOCR GPU直接识别图片。

- [v1开发说明书](docs/development_plan_v1_20260922.md)
- [v0.2.0评估基线](docs/evaluation_v0.2.0_20260922.md)
- [v0.2.0 P3端到端评估](docs/p3_evaluation_v0.2.0_20260922.md)
- [P3部署说明](docs/p3_deployment_v0.2.0_20260922.md)

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
- 只绑定P3本机`127.0.0.1:8800`；
- 复用`/home/lucas/ai-lab/cache`下现有模型缓存；
- 不配置永久代理；
- 不影响现有MySQL容器。

详细步骤见[P3部署说明](docs/p3_deployment_v0.2.0_20260922.md)。

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

## 模型文件

模型权重和缓存不提交Git。容器通过宿主机目录挂载保存模型，重建容器不需要重新下载已有模型。
