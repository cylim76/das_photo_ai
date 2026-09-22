# DAS Photo AI 推理服务开发说明书

- 文档系列：v1
- 当前修订：v1.1
- 初版日期：2026-09-22
- 最近更新：2026-09-22
- 项目名称：DAS Photo AI Inference Service
- Windows目录：`D:\RPA\das_photo_ai`
- P3规划目录：`/home/lucas/rpa/das_photo_ai`
- 当前软件版本：`0.2.0`

## 1. 当前开发进度

| 阶段 | 状态 | 说明 |
|---|---|---|
| 第一阶段：Mock/JSON与字段提取 | 已完成 | 代码、API、测试和133份历史JSON回归已经完成 |
| 第一阶段语义修正 | 已完成 | 区分图片观察值、ISO推算值和独立验证结果 |
| 第二阶段：PaddleOCR图片推理代码 | 已完成 | 图片上传、CPU/GPU引擎、模型单例和Docker配置已实现 |
| 第二阶段：P3实际部署验收 | 待执行 | 需要在P3构建容器并用真实图片测试GPU推理 |
| 第三阶段：DAS Photo页面接入 | 未开始 | 等P3图片推理稳定后开发 |

因此，本项目目前处于“第二阶段代码完成，等待P3实机部署验证”的位置。

## 2. 建设目标

建立一个独立于DAS Photo的AI推理服务：

- DAS Photo负责照片、任务、标注、人工确认和业务数据；
- DAS Photo AI负责OCR、业务字段提取、规则校验和结构化返回；
- Paddle、CUDA和模型依赖不安装进DAS Photo；
- 后续箱号、铅封号、箱门参数和装箱数量可以共用同一服务框架。

## 3. 总体架构

```text
DAS Photo
    │ HTTP API
    ▼
DAS Photo AI
    ├─ API层
    │   ├─ JSON识别接口
    │   └─ 图片上传接口
    ├─ OCR引擎层
    │   ├─ MockEngine
    │   ├─ JsonEngine
    │   ├─ PaddleOcrEngine（CPU）
    │   └─ PaddleOcrEngine（GPU）
    ├─ 统一OCRDocument
    ├─ 字段提取层
    │   ├─ 集装箱号
    │   └─ 铅封号
    └─ 校验层
        └─ ISO 6346
```

核心原则：OCR引擎只负责识别文字、置信度和坐标；业务提取器只依赖统一的`OCRDocument`，不直接依赖PaddleOCR内部对象。

## 4. 第一阶段完成内容

1. FastAPI服务、配置、日志和统一错误返回。
2. 健康检查、引擎列表和JSON识别接口。
3. `MockEngine`用于无模型接口联调。
4. `JsonEngine`读取PaddleOCR已生成的JSON。
5. PaddleOCR原始结构转换为统一`OCRDocument`。
6. OCR文字块的顺序和空间组合。
7. 集装箱号提取、字符混淆纠正和ISO 6346校验。
8. 铅封号候选提取和日期、重量、箱型、船公司等噪声过滤。
9. 命令行测试工具和133份历史结果评估脚本。
10. Windows无显卡环境中的自动测试。

第一阶段运行时不读取DAS Photo数据库。数据库中的正确答案只允许用于离线评估。

## 5. 校验位的严格语义

校验位计算不能代替图片识别。API必须区分以下情况。

### 5.1 图片只识别到前10位

```json
{
  "value": "HASU506039",
  "observed_value": "HASU506039",
  "suggested_value": "HASU5060396",
  "observed_check_digit": null,
  "calculated_check_digit": "6",
  "check_digit_source": "calculated",
  "validation": "inferred",
  "verification": "unverified"
}
```

这表示主值仍是图片实际观察到的10位，完整箱号只作为建议值，不能说明前10位一定识别正确。

### 5.2 图片识别到完整11位且校验通过

```json
{
  "value": "MSCU6639870",
  "observed_value": "MSCU6639870",
  "suggested_value": "MSCU6639870",
  "observed_check_digit": "0",
  "calculated_check_digit": "0",
  "check_digit_source": "observed",
  "validation": "valid",
  "verification": "verified"
}
```

只有图片中独立观察到的第11位与ISO计算值一致，才能称为已验证。

### 5.3 图片观察值与计算值不一致

```json
{
  "value": "MSCU6639871",
  "observed_value": "MSCU6639871",
  "suggested_value": "MSCU6639870",
  "observed_check_digit": "1",
  "calculated_check_digit": "0",
  "check_digit_source": "observed",
  "validation": "invalid",
  "verification": "mismatch"
}
```

服务保留图片观察值，不能用计算值静默覆盖；DAS Photo必须提示人工确认。

## 6. 第二阶段完成内容

1. 新增`PaddleOcrEngine`，支持`paddle_cpu`和`paddle_gpu`。
2. 新增`POST /api/v1/recognize/image`图片上传接口。
3. 图片以临时文件交给PaddleOCR，完成后立即删除。
4. PaddleOCR结果对象通过`.json`转换为统一数据结构。
5. OCR模型按引擎和配置单例缓存，不会每个请求重新加载。
6. 同一个Paddle预测器使用线程锁保护，初期避免并发调用导致不稳定。
7. 支持启动时预加载模型，P3容器配置为预加载。
8. 限制上传文件类型和大小，默认最大25MB。
9. 增加Paddle不可用时的503错误，而不是无提示地退回Mock。
10. 增加Dockerfile、Compose、模型缓存挂载和GPU配置。

PaddleOCR官方Python接口使用`PaddleOCR(...)`初始化，通过`predict(image_path)`推理，并可从结果对象`.json`属性取得JSON。实现参考[PaddleOCR OCR Pipeline官方文档](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/pipeline_usage/OCR.en.md)。

## 7. 第二阶段尚未完成的实机工作

以下工作必须在P3执行，Windows无显卡开发机不能代替：

1. 构建`das-photo-ai:0.2.0`镜像。
2. 确认容器能看到RTX 5070 Ti。
3. 确认PaddleOCR 3.5.0模型可以从现有缓存加载。
4. 上传真实照片，检查`ocr_source=paddleocr:gpu:0`。
5. 记录首次加载时间、单张推理时间和GPU显存占用。
6. 使用89张箱号和44张铅封号原图重新跑端到端评估。
7. 针对校验位和铅封号研究局部裁剪、放大和二次OCR。

在这些项目完成之前，第二阶段只能称为“代码完成”，不能称为“P3部署验收完成”。

## 8. 技术与版本约定

- 业务代码兼容Python 3.10～3.14；
- Windows和P3非GPU环境推荐Python 3.12；
- FastAPI + Uvicorn + Pydantic；
- PaddlePaddle 3.3.0；
- PaddleOCR 3.5.0；
- PaddleX 3.5.2；
- 已验证GPU基础镜像：`paddlepaddle/paddle:3.3.0-gpu-cuda12.9-cudnn9.9`；
- P3 GPU容器跟随基础镜像内置Python 3.10；
- 默认OCR版本固定为`PP-OCRv5`，不会在未评估时自动升级模型；
- 默认端口：`8800`；
- GPU初期只运行一个Uvicorn worker。

## 9. API v1

### `GET /api/v1/health`

返回服务状态、服务版本、提取器版本、默认引擎和时间。

### `GET /api/v1/models`

返回：

- `mock`、`json`、`paddle_cpu`、`paddle_gpu`；
- PaddleOCR是否安装；
- 引擎是否已经加载；
- OCR版本、语言和GPU设备。

### `POST /api/v1/recognize`

处理Mock或已有PaddleOCR JSON，保留作为开发、回归和排错接口。

```json
{
  "engine": "json",
  "targets": ["container_number", "seal_number"],
  "ocr_result": {
    "rec_texts": ["HASU", "506039"],
    "rec_scores": [0.99, 0.99],
    "rec_boxes": [[100, 100, 200, 145], [215, 100, 360, 145]]
  }
}
```

### `POST /api/v1/recognize/image`

使用`multipart/form-data`上传图片：

| 字段 | 必填 | 说明 |
|---|---|---|
| `image` | 是 | JPG、PNG、BMP、WebP或TIFF |
| `engine` | 否 | `paddle_gpu`、`paddle_cpu`或用于测试的`mock` |
| `targets` | 否 | 逗号分隔，默认箱号和铅封号 |
| `request_id` | 否 | 调用方提供的追踪编号 |

Linux调用示例：

```bash
curl -X POST http://127.0.0.1:8800/api/v1/recognize/image \
  -F 'engine=paddle_gpu' \
  -F 'targets=container_number,seal_number' \
  -F 'image=@/path/to/photo.jpg'
```

## 10. 引擎生命周期

所有引擎实现统一接口：

```python
class OcrEngine:
    def recognize(self, input_data: EngineInput) -> OCRDocument:
        ...
```

`create_engine()`对实例做缓存。PaddleOCR模型第一次创建后会在后续请求中复用。生产容器设置：

```env
DAS_AI_ENGINE=paddle_gpu
DAS_AI_PRELOAD_MODEL=1
```

服务启动时加载模型；如果模型或GPU不可用，容器启动失败，从而避免表面在线但实际不能识别。

## 11. 配置项

```env
DAS_AI_HOST=127.0.0.1
DAS_AI_PORT=8800
DAS_AI_ENGINE=mock
DAS_AI_LOG_LEVEL=INFO
DAS_AI_MAX_CANDIDATES=10
DAS_AI_MAX_IMAGE_MB=25
DAS_AI_PRELOAD_MODEL=0
DAS_AI_PADDLE_DEVICE=gpu:0
DAS_AI_PADDLE_LANG=
DAS_AI_PADDLE_OCR_VERSION=PP-OCRv5
DAS_AI_PADDLE_DET_MODEL=
DAS_AI_PADDLE_REC_MODEL=
```

语言参数默认留空，沿用P3已经测试成功的PaddleOCR默认模型组合；只有明确比较其他语言模型时才设置。Windows默认仍使用`mock`或`json`，P3 Compose会覆盖为`paddle_gpu`。

## 12. 模型和缓存

- 模型权重不提交Git；
- PaddleX缓存挂载到`/root/.paddlex`；
- PaddleOCR缓存挂载到`/root/.paddleocr`；
- 自定义模型目录挂载到`/models`；
- 删除或重建容器不会删除宿主机缓存；
- 不在Compose中固化HTTP代理；
- 只有首次缺少模型或安装包时才需要外网。

## 13. 当前评估基线

使用历史OCR JSON重新按严格语义统计：

| 任务 | 数量 | 图片观察值正确 | 建议值正确 | 验证状态 |
|---|---:|---:|---:|---|
| 集装箱号 | 89 | 31（34.83%） | 89（100%） | 31已验证，58未验证 |
| 铅封号 | 44 | 31（70.45%） | 31（70.45%） | 无统一校验规则 |

“箱号建议值100%”只说明在这批有正确答案的历史样本中，主体加计算校验位与答案一致；不能表述为89张都从图片完整识别并验证成功。

## 14. 测试要求

当前自动测试覆盖：

1. ISO 6346计算和验证；
2. 完整、缺校验位和校验位不一致的箱号；
3. 字母数字混淆纠正；
4. 铅封号与噪声过滤；
5. PaddleOCR JSON及`res`外层结构；
6. Mock/JSON API；
7. 图片上传、文件类型和错误引擎；
8. 使用模拟PaddleOCR模块测试CPU/GPU参数、临时图片和结果转换；
9. 133份历史JSON离线回归。

Windows测试不能证明CUDA、PaddlePaddle和真实模型可用，GPU部分必须在P3验收。

## 15. 第二阶段验收标准

- P3容器启动并通过`/health`；
- `/models`显示`paddle_gpu`可用且已加载；
- 容器内Paddle检测到一块GPU；
- 图片接口可返回真实OCR文字、坐标和字段候选；
- 模型只加载一次；
- 临时上传文件请求结束后删除；
- 容器重建后模型缓存仍存在；
- 箱号明确区分`verified`、`unverified`和`mismatch`；
- 真实原图端到端评估结果被记录；
- 停止和删除AI容器不影响DAS Photo和MySQL容器。

## 16. 第三阶段：DAS Photo接入

第二阶段验收后开发：

1. AI服务开关、地址、API Key、超时和连接测试；
2. 标注页“AI标注”按钮；
3. AI预标注、人工确认、人工修改和失败状态；
4. 显示“已验证”“校验位推算”“校验不一致”；
5. AI结果不能静默覆盖人工标注；
6. 保存服务、OCR、模型和提取器版本；
7. 后续增加批量预标注后台任务。

## 17. 后续扩展

- 校验位区域局部检测和二次OCR；
- 铅封区域检测、旋转校正和专用识别；
- PP-YOLOE目标检测；
- 装箱数量和WLHF视觉任务；
- 训练集快照、导出和模型版本关联；
- 固定基准集上的模型对比报表。

## 18. 安全与审计

- 初期服务只绑定P3本机`127.0.0.1:8800`；
- 后续由DAS Photo转发或通过内网API Key调用；
- 日志不记录图片二进制和完整密钥；
- 上传图片只建立临时文件，推理后删除；
- 最大文件大小默认25MB；
- AI服务容器与DAS Photo、MySQL容器分别管理。

## 19. 版本记录

### v1.0 / 软件0.1.0

- 建立Mock/JSON接口和箱号、铅封号提取器。

### v1.1 / 软件0.2.0

- 修正校验位推算与验证语义；
- 新增图片上传接口；
- 新增PaddleOCR CPU/GPU引擎；
- 新增模型单例、启动预加载和并发保护；
- 新增P3 GPU Docker与Compose配置；
- 更新严格评估指标和第二阶段验收标准。
