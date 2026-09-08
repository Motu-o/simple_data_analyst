# -*- coding: utf-8 -*-
"""数据集分析系统 - 后端入口。

启动方式（推荐）：
    cd D:/Code/VSCode/Python/data_analyst/backend
    python launcher.py        # 同时启动 8000(API) 与 8081(前端静态服务) 并自动打开浏览器

代码结构（按功能拆分，便于维护）：
    config.py                全局配置：FastAPI 实例、CORS、存储目录、安全配置、共享状态
    utils.py                 公共工具：Ollama 调用、模型代码提取/修复、JSON 提取、安全预检、数据文件读取
    launcher.py              一键启动器：uvicorn(8000) + 静态服务(8081) + 自动开浏览器
    routes/                  功能路由包（按功能分文件夹）
        chat/__init__.py     通用与聊天：模型列表 / 数据集上传 / 数据分析问答（大模型生成代码沙盒执行）
        image/__init__.py    图片数据模块：zip 图片集上传 / 概览 / 切分 / 统一分辨率 / ResNet50 训练 / 评估 / 推理
        clean/__init__.py    数据清洗：大模型意图解析 + 白名单 pandas 确定性执行
        eda/__init__.py      EDA 分析：确定性图表生成、按需绘图、特征分析报告、打包下载
        predict/__init__.py  预测分析：回归/分类预测、指标、特征重要性、图表、多模型集成
        train/__init__.py    模型训练：数据集切分、算法推荐、代码生成、超参数训练与 loss 曲线

前端（frontend/）：单页 index.html（数据清洗 / EDA 分析 / 模型训练 / 预测分析四大模块
+ 奶龙聊天 + 开局视频 start_up.mp4），通过 http://127.0.0.1:8000 调用本后端。
"""
from config import app

# 显式导入各功能路由包，完成路由注册（顺序无依赖）
import routes.chat     # noqa: F401
import routes.image    # noqa: F401
import routes.clean    # noqa: F401
import routes.eda      # noqa: F401
import routes.predict  # noqa: F401
import routes.train    # noqa: F401

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
