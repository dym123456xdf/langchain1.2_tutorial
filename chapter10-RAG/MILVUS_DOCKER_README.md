# Milvus Standalone (Docker Compose) 启动说明

本目录的 `docker-compose.yml` 启的是 Milvus **standalone** 单节点模式,3 个容器:
- `milvus-etcd`  (quay.io/coreos/etcd:v3.5.25)         元数据存储
- `milvus-minio` (minio/minio:RELEASE.2024-05-28T...)  对象存储(向量/索引落盘)
- `milvus-standalone` (milvusdb/milvus:v3.0.0)         服务端,gRPC 监听 19530

> ⚠️ **版本对齐**:`pymilvus==3.0.1` 已装,服务端镜像必须 ≥ client 的 minor,所以固定 v3.0.0。
> 拉错版本会导致 `client.list_databases() / use_database()` 等 2.4 之后才有的 API 不可用。

## 1. 前置: Docker daemon 必须就绪

`docker info` 必须在 5 秒内返回,且能看到 "Server Version" 行。

如果 daemon 卡死(本次就遇到过),打开 **Docker Desktop → Troubleshoot → Restart Daemon**。

## 2. 国内镜像源(可选,但强烈建议)

`~/.docker/daemon.json` 已配:
```json
{
  "registry-mirrors": [
    "https://docker.m.daocloud.io",
    "https://mirror.ccs.tencentyun.com",
    "https://docker.mirrors.ustc.edu.cn"
  ]
}
```
改完需要 restart daemon 才生效。

## 3. 启动

```bash
cd /Users/daiyanmei/PycharmProjects/langchain1.2_tutorial/chapter10-RAG/
docker compose up -d
```

首启动会拉 ~1.3GB 镜像(milvusdb/milvus v3.0.0 占大头),耗时 2–10 分钟(取决于网络)。
镜像已缓存后冷启约 10 秒。

## 4. 验证

```bash
docker compose ps                     # 三个容器都应 healthy
lsof -iTCP:19530 -sTCP:LISTEN          # 应能看到 docker-proxy 在监听 19530
```

然后回到教程目录跑:

```bash
cd /Users/daiyanmei/PycharmProjects/langchain1.2_tutorial/chapter10-RAG/
/opt/anaconda3/envs/langchain1.2/bin/python -m "05_客服知识库拆分.milvus_init"
```

## 5. 停 / 清理

```bash
docker compose down            # 停容器,保留 volumes
docker compose down -v         # 停容器并删 volumes(会丢数据)
```

## 6. 数据持久化

volumes 挂载在本目录下的 `volumes/`,即 `./volumes/{etcd,minio,milvus}/`。
重启 Docker Desktop 数据不丢;`down -v` 会清空。