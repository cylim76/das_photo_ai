# DAS Photo AI 0.4.1 P3复核升级说明

0.4.0正式API全量回归结果为：箱号85/89、错误验证0、铅封43/44。4张未达到
`verified_exact`的原因不同：

- `F123821`和`F125870`只有一个正确数字，置信度分别为0.663281和0.672433，略低于0.70；
- `F132522`两个带框变体误读为`2`，去框变体以0.996584识别为正确的`9`，ISO把错误结果拦截为`mismatch`；
- `F116275`的单字符模型未返回数字，但旧版上下文4倍和灰度4倍都能独立观察到完整11位。

0.4.1采用分层后备策略：

1. 通用OCR已经识别完整11位时不运行后备；
2. 英文模型先识别校验位三个变体；
3. 英文结果已经通过ISO验证时立即返回；
4. 英文结果未找到、低置信度或不一致时，才运行`context_2x`、`context_4x`和
   `grayscale_4x`通用OCR复核；
5. 上下文复核必须独立观察到与原前10位一致的完整11位并通过ISO验证；
6. 唯一单数字的自动采用阈值由0.70调整为0.65，多变体候选下限仍保持0.50；
7. 所有分支仍不读取正确答案，不用ISO计算值生成或挑选直接识别数字。

## P3升级

```bash
cd /home/lucas/rpa/das_photo_ai
git pull --ff-only
git rev-parse --short HEAD

sudo docker compose build
sudo docker compose up -d
sudo docker compose ps
sudo docker compose logs --tail=200 das-photo-ai
```

检查版本：

```bash
curl -s http://127.0.0.1:8800/api/v1/health | python3 -m json.tool
curl -s http://127.0.0.1:8800/api/v1/models | python3 -m json.tool
```

预期版本为`0.4.1`，`minimum_single_score`为`0.65`。

## 正式API全量回归

```bash
cd /home/lucas/rpa/das_photo_ai

sudo docker run --rm \
  --network host \
  --user "$(id -u):$(id -g)" \
  --entrypoint python \
  -v "$PWD":/workspace:ro \
  -v /home/lucas/ai-lab:/home/lucas/ai-lab \
  -w /workspace \
  das-photo-ai:0.4.1 \
  -m scripts.evaluate_image_api \
  --report /home/lucas/ai-lab/ocr_eval_manifest_v1_20260922.json \
  --container-root /home/lucas/ai-lab/input/container_test \
  --seal-root /home/lucas/ai-lab/input/sealno_test \
  --api-base http://127.0.0.1:8800/api/v1 \
  --engine paddle_gpu \
  --output-dir /home/lucas/ai-lab/evaluation-runs/v0.4.1-full-api
```

验收要求：

- API错误为0；
- `false_verified`必须为0；
- 箱号目标是89/89，但不能为了达到目标放宽错误验证规则；
- 铅封保持43/44；
- 重点检查上下文复核只在英文直接结果未验证时触发；
- 保存P95和最大耗时，确认稀有复核分支没有明显影响整体性能。
