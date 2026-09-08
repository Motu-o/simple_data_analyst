# -*- coding: utf-8 -*-
"""全局配置：存储目录、安全配置、Ollama 配置、FastAPI 实例与共享状态。

所有功能模块（routes/*.py）从这里导入共享对象，避免循环依赖。
"""
import os
import sys
from contextlib import asynccontextmanager
from typing import Dict, List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ================= 存储目录 =================
# 以程序所在位置为基准，统一使用绝对路径：
#   - 源码运行时：config.py 所在目录（backend 下）
#   - 打包为 exe 后：exe 所在目录（保证数据可持久化，不随临时解压目录消失）
if getattr(sys, "frozen", False):
    BASE_DIR = os.path.dirname(os.path.abspath(sys.executable))
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

upload_folder = os.path.join(BASE_DIR, "uploads")
dataset_folder = os.path.join(upload_folder, "datasets")      # 上传的数据集
cleaned_folder = os.path.join(upload_folder, "cleaned")        # 清洗结果
img_folder = os.path.join(upload_folder, "imgs")               # 预测图表
eda_export_folder = os.path.join(upload_folder, "eda_export")  # EDA 生成图
split_folder = os.path.join(upload_folder, "split")            # 训练集/测试集
model_folder = os.path.join(upload_folder, "models")           # 训练模型(best/last)

for _d in (dataset_folder, img_folder, cleaned_folder, eda_export_folder, split_folder, model_folder):
    os.makedirs(_d, exist_ok=True)

# ================= 安全配置 =================
# 禁止模型生成代码中出现的关键字（静态扫描，用于模型代码沙盒拦截）
forbidden_keywords = {
    "os.", "subprocess", "sys.", "__import__",
    "eval(", "exec(", "open(", "requests", "while True", "os.remove",
    "shutil", "signal", "multiprocess", "threading"
}
max_exec_seconds = 8  # 模型生成代码的最大运行时间（秒）

# ================= Ollama 配置 =================
ollama_url = "http://127.0.0.1:11434/api/generate"
# 系统固定使用的大模型（Ollama 本地模型，全系统统一）
DEFAULT_MODEL = "qwen2.5:7b-instruct-q4_K_M"

# ================= FastAPI 实例 =================
def _cleanup_models():
    """程序结束时清理训练模块产出的 best_loss / last_loss 模型文件。"""
    try:
        for f in os.listdir(model_folder):
            if f.endswith(".joblib"):
                os.remove(os.path.join(model_folder, f))
    except Exception:
        pass


@asynccontextmanager
async def lifespan(app):
    yield
    # 程序结束后自动清理训练模型（best_loss / last_loss），释放磁盘
    _cleanup_models()


app = FastAPI(title="数据集分析系统", lifespan=lifespan)

# 解决跨域（本地前后端分离开发/运行）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= 共享状态 =================
# 内存会话存储（数据分析问答 / 清洗问答共用）
chat_sessions: Dict[str, List[Dict]] = {}
