# DAS Photo AI 0.4.0 P3升级与回归说明

本次升级把已经在P3独立实验中验证的英文校验位直接识别接入正式图片API。它只在通用
OCR取得合法前10位、但没有观察到第11位校验位时运行。

## 1. 升级内容

- 正式后备模型：`en_PP-OCRv5_mobile_rec`；
- 校验位输入：保留方框彩色、保留方框灰度、去框灰度三个变体；
- 选择规则：多变体一致，或唯一结果达到高置信度；
- ISO 6346只在OCR数字选定后验证，不生成或选择数字；
- 已识别完整11位时不运行后备模型；
- 后备模型失败时保留通用OCR的10位结果，不使整个请求失败；
- 通用OCR模型和校验位模型共用GPU推理锁；
- API在`results.container_number.postprocessing`返回后备流程审计信息。

固定开发集的独立实验结果是：通用流程和历次裁剪组合先达到83/89；英文校验位模型在
最后6张上识别6/6，累计89/89，错误验证为0。该结果属于开发集成绩，正式效果仍需全量
API回归和新照片盲测确认。

## 2. 更新与构建

```bash
cd /home/lucas/rpa/das_photo_ai
git pull --ff-only
git rev-parse --short HEAD

sudo docker compose build
sudo docker compose up -d
sudo docker compose ps
sudo docker compose logs --tail=200 das-photo-ai
```

Compose会构建`das-photo-ai:0.4.0`，同时保留原有模型缓存。已经下载到
`/home/lucas/ai-lab/cache/paddlex`的`en_PP-OCRv5_mobile_rec`不会重复下载。

## 3. 服务检查

```bash
curl -s http://127.0.0.1:8800/api/v1/health | python3 -m json.tool
curl -s http://127.0.0.1:8800/api/v1/models | python3 -m json.tool
```

预期：

- `version`和`extractor_version`均为`0.4.0`；
- `paddle_gpu.loaded`为`true`；
- `container_check_digit.enabled`为`true`；
- `container_check_digit.loaded`为`true`；
- `container_check_digit.model_name`为`en_PP-OCRv5_mobile_rec`。

## 4. 单张接口检查

选择一张以前只能识别前10位的箱号原图：

```bash
curl -s -X POST http://127.0.0.1:8800/api/v1/recognize/image \
  -F 'engine=paddle_gpu' \
  -F 'targets=container_number' \
  -F 'image=@/完整路径/照片.jpeg' \
  | python3 -m json.tool
```

后备识别成功时应看到：

```json
{
  "verification": "verified",
  "check_digit_source": "observed",
  "postprocessing": {
    "status": "applied",
    "strategy": "english_check_digit_direct_recognition",
    "model_name": "en_PP-OCRv5_mobile_rec"
  }
}
```

`postprocessing.status`也可能是：

- `not_needed`：通用OCR已经取得完整11位；
- `not_found`：三个变体没有可靠单数字；
- `conflict`：不同变体给出冲突数字；
- `error`：后备模型发生错误，接口仍保留原始10位结果。

## 5. 固定集完整回归

升级后必须从原始照片重新调用正式API，而不是复用此前保存的裁剪结果：

```bash
cd /home/lucas/rpa/das_photo_ai

sudo docker run --rm \
  --network host \
  --user "$(id -u):$(id -g)" \
  --entrypoint python \
  -v "$PWD":/workspace:ro \
  -v /home/lucas/ai-lab:/home/lucas/ai-lab \
  -w /workspace \
  das-photo-ai:0.4.0 \
  -m scripts.evaluate_image_api \
  --report /home/lucas/ai-lab/ocr_eval_manifest_v1_20260922.json \
  --container-root /home/lucas/ai-lab/input/container_test \
  --seal-root /home/lucas/ai-lab/input/sealno_test \
  --api-base http://127.0.0.1:8800/api/v1 \
  --engine paddle_gpu \
  --output-dir /home/lucas/ai-lab/evaluation-runs/v0.4.0-full-api
```

查看结果：

```bash
cat /home/lucas/ai-lab/evaluation-runs/v0.4.0-full-api/summary.json
```

验收重点：

- 箱号`observed_top1_exact`；
- 箱号`verified_exact`和`false_verified`，其中`false_verified`必须为0；
- 箱号`verification.verified`；
- 箱号`postprocessing.applied`；
- API错误必须为0；
- 逐图原始响应中不存在“ISO计算值冒充图片观察值”；
- 铅封号不能因本次箱号改动发生回退；
- 对比原有P95和最大耗时，确认后备识别耗时可接受。

## 6. 新照片盲测

全量回归完成后建立100～200张从未参与调试的新照片集。正确答案必须在运行AI前由人工
确认并冻结。开发集达到89/89不能代替盲测结果；盲测失败样本只有在首轮成绩保存后，才
能转入下一版开发集。
