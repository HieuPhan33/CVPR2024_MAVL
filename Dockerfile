FROM nvcr.io/nvidia/pytorch:21.10-py3
RUN apt-get update && DEBIAN_FRONTEND=noninteractive  apt-get install -y ffmpeg libsm6 libxext6 libglib2.0-0 git ca-certificates && apt-get clean
COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt && pip install --upgrade google-cloud-storage
RUN pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip install timm