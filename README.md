# DAS Photo AI

DAS Photo的独立OCR推理与业务字段提取服务。

当前版本为`0.2.0`：Windows可以使用Mock/JSON进行开发测试；P3可以通过独立Docker容器使用PaddleOCR GPU直接识别图片。

- [v1开发说明书](docs/development_plan_v1_20260922.md)
- [v0.2.0评估基线](docs/evaluation_v0.2.0_20260922.md)
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

## 模型文件

模型权重和缓存不提交Git。容器通过宿主机目录挂载保存模型，重建容器不需要重新下载已有模型。
