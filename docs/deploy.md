# 部署

## Docker

```bash
docker build -t ascendforge:1.0.0 -f deploy/Dockerfile .
docker run -p 8000:8000 ascendforge:1.0.0
```

## Kubernetes

```bash
kubectl apply -f deploy/k8s/deployment.yaml
kubectl apply -f deploy/k8s/hpa.yaml
```

## Helm

```bash
helm install ascendforge deploy/helm
```
