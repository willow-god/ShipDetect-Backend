# ShipRecog-Backend

ShipRecog-Backend 是一个基于 FastAPI 的船舶识别后端服务，主要对接算法端所训练的模型（YOLOv8 & PPOCRv4），实现高效的船舶检测和识别。

## 功能概述

- **模型推理**：封装 YOLOv8 进行目标检测，结合 PPOCRv4 进行文字识别。
- **API 服务**：提供 RESTful API，支持图片上传与推理。
- **用户登录**：计划支持 JWT 认证，提供用户权限管理。
- **模块化设计**：推理过程封装成类，方便维护与扩展。

## 项目结构

```
ShipRecog-Backend/
├── app/                     # 主应用目录
│   ├── api/                 # API 路由
│   │   ├── v1/              # API v1 版本
│   │   │   ├── endpoints/   # 具体路由
│   │   │   │   ├── auth.py  # 认证相关 API
│   │   │   │   ├── infer.py # 推理 API（调用模型推理）
│   │   │   │   ├── health.py# 健康检查接口
│   │   │   ├── router.py    # API 统一路由管理
│   ├── core/                # 核心模块
│   │   ├── config.py        # 配置管理
│   │   ├── security.py      # 认证与安全模块
│   ├── models/              # 模型封装
│   │   ├── yolov8_model.py  # YOLOv8 目标检测
│   │   ├── ppocr_model.py   # PPOCRv4 文字识别
│   ├── schemas/             # 数据模型（Pydantic）
│   │   ├── user.py          # 用户相关数据模型
│   │   ├── infer.py         # 推理请求/响应数据模型
│   ├── services/            # 业务逻辑层
│   │   ├── auth_service.py  # 认证服务
│   │   ├── infer_service.py # 推理逻辑封装
│   ├── main.py              # FastAPI 入口
├── tests/                   # 测试代码
│   ├── test_infer.py        # 推理 API 测试
│   ├── test_auth.py         # 认证 API 测试
├── requirements.txt         # 依赖列表
├── Dockerfile               # Docker 构建文件
├── .env                     # 环境变量配置
├── README.md                # 说明文档
```

## 快速开始

### 1. 安装依赖

```sh
pip install -r requirements.txt
```

### 2. 运行项目

```sh
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. 访问 API 文档

- Swagger UI: http://localhost:8000/docs
- Redoc: http://localhost:8000/redoc

## 配置文件

项目使用 `.env` 文件管理环境变量，例如：

```ini
SECRET_KEY="your_secret_key"
DATABASE_URL="sqlite:///./database.db"
MODEL_PATH_YOLO="models/yolov8.pt"
MODEL_PATH_PPOCR="models/ppocr"
```

## 未来计划

-  增加数据库存储推理结果
-  提供 WebSocket 支持实时推理
-  完善用户管理（注册、权限控制）

欢迎贡献代码或提出建议！
