FROM python:3.10.14 as modelscope
RUN pip config set global.index-url https://mirrors.tencentyun.com/pypi/simple && \
    pip install --upgrade pip && \
    pip install modelscope

RUN python -c "from modelscope import snapshot_download; snapshot_download('iic/CosyVoice-300M', local_dir='/CosyVoice-300M')"


FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

RUN sed -i 's/archive.ubuntu.com/repo.huaweicloud.com/g' /etc/apt/sources.list && \
    apt-get update && \
    pip config set global.index-url https://mirrors.huaweicloud.com/repository/pypi/simple && \
    pip config set global.trusted-host repo.huaweicloud.com && \
    pip config set global.timeout 120 && \
    pip install --upgrade pip

WORKDIR /app/cosy

# copy model
COPY --from=modelscope /CosyVoice-300M /app/cosy/pretrained_models/CosyVoice-300M

COPY requirements.txt /app/cosy/requirements.txt
RUN pip install -r requirements.txt
# 使用原始源安装 mcelery, 镜像不够新
RUN pip install -i https://pypi.org/simple mcelery==0.1.0

# copy code
COPY cosyvoice /app/cosy/cosyvoice
COPY third_party /app/cosy/third_party
COPY cosy_celery.py /app/cosy/cosy_celery.py

CMD celery -A cosy_celery worker -l INFO -Q cosy_infer -P solo -n cosy_worker