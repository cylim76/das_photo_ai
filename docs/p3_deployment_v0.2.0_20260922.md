# DAS Photo AI 0.2.0 P3部署说明

本说明用于P3 Ubuntu 24.04、RTX 5070 Ti和已安装NVIDIA Container Toolkit的环境。

## 1. 前提

已验证环境：

- NVIDIA驱动：595.84；
- GPU：RTX 5070 Ti 16GB；
- Docker：29.7.2；
- NVIDIA容器运行时可用；
- GPU基础镜像：`paddlepaddle/paddle:3.3.0-gpu-cuda12.9-cudnn9.9`；
- PaddlePaddle 3.3.0在CUDA 12.9镜像中可识别GPU；
- PaddleOCR 3.5.0和PaddleX 3.5.2已经在实验容器中验证。

以下操作不会修改DAS Photo数据库，不会停止MySQL容器，也不会安装CUDA到宿主机。

## 2. 同步代码

代码上传GitHub后，在P3执行：

```bash
cd /home/lucas/rpa
git clone <das_photo_ai仓库地址> das_photo_ai
cd /home/lucas/rpa/das_photo_ai
```

如果目录已经存在：

```bash
cd /home/lucas/rpa/das_photo_ai
git pull
```

## 3. 部署前检查

```bash
sudo docker image inspect \
  paddlepaddle/paddle:3.3.0-gpu-cuda12.9-cudnn9.9 \
  >/dev/null

sudo docker compose version

sudo docker run --rm \
  --runtime=nvidia \
  --gpus all \
  paddlepaddle/paddle:3.3.0-gpu-cuda12.9-cudnn9.9 \
  nvidia-smi
```

第三条命令只启动临时容器检查GPU，不会影响现有MySQL容器。

## 4. 模型缓存目录

Compose默认复用实验环境目录：

```text
/home/lucas/ai-lab/cache/paddlex
/home/lucas/ai-lab/cache/paddleocr
```

确认目录：

```bash
mkdir -p /home/lucas/ai-lab/cache/paddlex
mkdir -p /home/lucas/ai-lab/cache/paddleocr
mkdir -p /home/lucas/rpa/das_photo_ai/models
```

如果实际缓存不在这些位置，在项目目录建立不提交Git的`.env`：

```env
DAS_AI_PADDLEX_CACHE=/实际路径/paddlex
DAS_AI_PADDLEOCR_CACHE=/实际路径/paddleocr
DAS_AI_MODELS_DIR=/home/lucas/rpa/das_photo_ai/models
```

## 5. 构建镜像

```bash
cd /home/lucas/rpa/das_photo_ai
sudo docker compose build
```

构建会：

1. 复用本机已有CUDA 12.9 Paddle基础镜像；
2. 安装FastAPI、PaddleOCR 3.5.0和PaddleX 3.5.2；
3. 不安装CPU版PaddlePaddle；
4. 不写入永久代理配置。

如果安装包下载失败，先解决当次Docker构建网络，再重新执行同一命令。Docker会复用已经成功的构建层。

## 6. 启动服务

```bash
sudo docker compose up -d
```

查看状态：

```bash
sudo docker compose ps
sudo docker compose logs --tail=200 das-photo-ai
```

Compose设置`DAS_AI_PRELOAD_MODEL=1`，因此启动日志停止增长并不代表卡死。首次启动需要加载模型；模型缺失时还可能下载模型。

## 7. 检查GPU和服务

```bash
sudo docker exec das-photo-ai nvidia-smi

curl http://127.0.0.1:8800/api/v1/health

curl http://127.0.0.1:8800/api/v1/models
```

预期：

- 健康检查返回`status=ok`；
- 默认引擎为`paddle_gpu`；
- `paddle_gpu`显示`available=true`；
- 模型预加载完成后显示`loaded=true`；
- GPU设备为`gpu:0`。

## 8. 上传真实照片测试

```bash
curl -X POST http://127.0.0.1:8800/api/v1/recognize/image \
  -F 'engine=paddle_gpu' \
  -F 'targets=container_number,seal_number' \
  -F 'image=@/完整路径/F119755.jpeg' \
  | python3 -m json.tool
```

重点检查：

- `engine`为`paddle_gpu`；
- `ocr_source`为`paddleocr:gpu:0`；
- `ocr_items`包含真实文字和坐标；
- 箱号结果带`verification`；
- 只识别前10位时必须是`unverified`；
- 观察校验位与计算值一致时才是`verified`。

## 9. 查看资源使用

另开一个终端：

```bash
watch -n 1 nvidia-smi
```

Docker资源：

```bash
sudo docker stats das-photo-ai
```

初次验收记录：

- 容器启动到模型就绪耗时；
- 首张图片耗时；
- 后续单张图片耗时；
- 空闲显存；
- 推理峰值显存。

## 10. 与DAS Photo的连接

服务默认只映射：

```text
127.0.0.1:8800
```

P3宿主机上的DAS Photo可以直接调用：

```text
http://127.0.0.1:8800
```

当前不直接暴露给局域网。第三阶段接入DAS Photo后，再决定是否由DAS Photo转发或开放带API Key的内网访问。

## 11. 停止、启动和更新

停止：

```bash
sudo docker compose stop
```

启动：

```bash
sudo docker compose start
```

重启：

```bash
sudo docker compose restart
```

代码更新：

```bash
cd /home/lucas/rpa/das_photo_ai
git pull
sudo docker compose build
sudo docker compose up -d
```

## 12. 删除与恢复

删除服务容器和网络：

```bash
sudo docker compose down
```

此命令不会删除：

- `/home/lucas/ai-lab/cache`中的模型缓存；
- `/home/lucas/rpa/das_photo_ai/models`；
- 基础Paddle镜像；
- DAS Photo数据；
- MySQL容器和数据。

重新执行`sudo docker compose up -d`即可恢复。

如果以后确认不再使用，可以另行删除`das-photo-ai:0.2.0`镜像；删除镜像前应先确认不再有容器引用它。

## 13. 常见问题

### `/models`显示Paddle不可用

检查构建日志中`paddleocr==3.5.0`是否安装成功：

```bash
sudo docker exec das-photo-ai \
  python -c "import paddleocr, paddlex; print(paddleocr.__version__, paddlex.__version__)"
```

### 容器能运行`nvidia-smi`但Paddle看不到GPU

确认使用的是CUDA 12.9镜像，不要切回此前失败的CUDA 13.0镜像：

```bash
sudo docker exec das-photo-ai \
  python -c "import paddle; print(paddle.device.cuda.device_count())"
```

### 启动时尝试下载模型

说明挂载的缓存目录中缺少当前配置的模型。先检查Compose实际挂载：

```bash
sudo docker inspect das-photo-ai --format '{{json .Mounts}}'
```

不要为了推理长期开启代理。模型完整下载并保存在宿主机缓存后，本地推理不需要外网。
