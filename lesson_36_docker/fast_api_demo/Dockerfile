FROM python:3.10

RUN apt-get update \
   && apt-get install -y \
       vim \
       build-essential \
       cmake \
       git \
       wget \
       unzip \
       libjpeg62-turbo-dev \
   && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
     bzip2 \
     g++ \
     git \
     graphviz \
     libgl1-mesa-glx \
     libhdf5-dev \
     openmpi-bin \
     screen && \
   rm -rf /var/lib/apt/lists/* \
   apt-get upgrade

RUN pip install opencv-python

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

COPY . /app

EXPOSE 5000
CMD ["uvicorn", "app:app", "--reload", "--host", "0.0.0.0", "--port", "5000"]
