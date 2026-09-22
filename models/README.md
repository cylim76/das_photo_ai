# Models

模型权重和 PaddleOCR 缓存不提交到 Git。

P3部署时，本目录或`DAS_AI_MODELS_DIR`指定目录以Docker volume形式挂载到容器`/models`，确保容器更新、删除和重建后自定义模型仍然保留。

PaddleOCR官方模型缓存分别挂载到容器`/root/.paddlex`和`/root/.paddleocr`，不存放在Git仓库中。当前Compose默认复用`/home/lucas/ai-lab/cache`下已经下载的模型。
