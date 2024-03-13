# Docker deployment

## CPU-bound app

Requirements:

    * Docker



To deploy FastAPI application as Docker container, execute next instructions:

1. Build Docker image:

```bash
docker build -t fast_api_test .
```

2. Run Docker container with publishing port 5000 (as configured in Dockerfile):

```bash
docker run --rm -d -p 5000:5000 fast_api_test
```


## GPU-bound app

Requirements:

- CUDA compatible GPU
- installed [CUDA](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html)
- installed [NVIDIA container tookit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)


1. Build Docker image:

```bash
docker build -t fast_api_test .
```

2. Run Docker container with publishing port 5000 (as configured in Dockerfile):

```bash
docker run --rm -d -p 5000:5000 --gpus all fast_api_test
```


