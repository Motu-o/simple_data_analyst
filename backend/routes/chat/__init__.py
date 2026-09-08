# -*- coding: utf-8 -*-
"""通用与聊天接口：模型列表 / 文件上传 / 数据分析问答（大模型生成代码沙盒执行）。"""
import os
import re
import threading
import base64
import uuid
import sys
from io import StringIO

import json
import requests
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from fastapi import UploadFile, Query
from fastapi.responses import StreamingResponse
from datetime import datetime

from config import app, chat_sessions, dataset_folder, cleaned_folder, DEFAULT_MODEL
from utils import (
    call_ollama, call_ollama_stream, extract_python_code, fix_plot_code, neutralize_df_reload,
    safe_code_precheck,
)

# 模型接口：仅返回系统固定的 qwen2.5:7b-instruct-q4_K_M
@app.get("/models")
async def get_models():
    ollama_base = "http://127.0.0.1:11434"
    try:
        resp = requests.get(f"{ollama_base}/api/tags", timeout=8)
        resp.raise_for_status()
        data = resp.json()
        all_models = data.get("models", [])
        fixed = [m for m in all_models if m.get("name") == DEFAULT_MODEL]
        return {"models": fixed if fixed else [], "error": "" if fixed else f"未找到模型 {DEFAULT_MODEL}，请先 pull"}
    except Exception as e:
        return {"models": [], "error": str(e)}

# 纯聊天接口（不依赖数据集）：根据用户输入推理意图直接回答
@app.post("/chat")
async def chat_plain(
    message: str = Query(),
    model_name: str = Query(default=DEFAULT_MODEL),
    temperature: float | None = Query(None),
):
    if not message.strip():
        return {"code": -1, "msg": "消息不能为空"}

    prompt = (
        "你是奶龙，一个活泼友好的AI聊天助手。请先理解用户输入的意图，"
        "然后用自然、简洁、热情的中文回答用户。不要编造事实。\n\n"
        f"用户：{message.strip()}\n奶龙："
    )
    res = call_ollama(prompt, model_name=model_name, stream=False, temperature=temperature)
    if "error" in res:
        return {"code": -1, "msg": res["error"]}

    return {"code": 0, "msg": "成功", "data": {"reply": res.get("response", "").strip()}}

# 纯聊天流式接口（SSE）：逐块输出，前端打字机效果
@app.post("/chat_stream")
def chat_stream(
    message: str = Query(),
    model_name: str = Query(default=DEFAULT_MODEL),
    temperature: float | None = Query(None),
):
    if not message.strip():
        return {"code": -1, "msg": "消息不能为空"}
    prompt = (
        "你是奶龙，一个活泼友好的AI聊天助手。请先理解用户输入的意图，"
        "然后用自然、简洁、热情的中文回答用户。不要编造事实。\n\n"
        f"用户：{message.strip()}\n奶龙："
    )

    def _gen():
        got = False
        for chunk in call_ollama_stream(prompt, model_name, temperature=temperature):
            if chunk is None:
                yield f"data: {json.dumps({'type': 'text', 'content': '❌ 模型调用失败，请确认 Ollama 已启动'}, ensure_ascii=False)}\n\n"
                break
            got = True
            yield f"data: {json.dumps({'type': 'text', 'content': chunk}, ensure_ascii=False)}\n\n"
        if not got:
            yield f"data: {json.dumps({'type': 'text', 'content': '奶龙没想好怎么回答～'}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(_gen(), media_type="text/event-stream")

# 文件上传接口
# 大数据集异步上传任务表：task_id -> {"status": "running"|"done"|"error", "result": {...}, "error": str}
upload_tasks = {}
# 超过该字节数（50MB）的表格式文件走异步任务模式，避免同步阻塞请求
ASYNC_UPLOAD_THRESHOLD = 50 * 1024 * 1024
# 图片集 zip 大小上限：100MB
MAX_IMAGE_ZIP_SIZE = 100 * 1024 * 1024

def _process_upload(save_path: str, original_name: str) -> dict:
    """读取数据集并生成上传信息（同步与异步共用）。"""
    ext = os.path.splitext(save_path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(save_path)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(save_path)
    else:
        raise ValueError("仅支持csv，excel文件")
    info = {
        "filename": original_name,
        "rows": len(df),
        "cols": list(df.columns),
        "numeric_cols": [str(c) for c in df.columns if pd.api.types.is_numeric_dtype(df[c])],
        "file_path": save_path,
        "preview_columns": [str(c) for c in df.columns],
        "sample": df.head(10).astype(object).where(df.head(10).notna(), None).to_dict(orient="records"),
    }
    return info

def _upload_worker(task_id: str, save_path: str, original_name: str):
    try:
        info = _process_upload(save_path, original_name)
        upload_tasks[task_id] = {"status": "done", "result": info}
    except Exception as e:
        upload_tasks[task_id] = {"status": "error", "error": str(e)}

@app.post("/upload_dataset")
async def load_dataset(file: UploadFile):
    # 获取原始文件名
    original_name = file.filename

    # 去除路径，只保留文件名
    original_name = os.path.basename(original_name)

    # 替换危险字符，防止路径穿越
    safe_name = re.sub(r'[^a-zA-Z0-9._-]', '_', original_name)

    # 限制文件名长度
    if len(safe_name) > 100:
        name, ext = os.path.splitext(safe_name)
        safe_name = name[:100] + ext

    # 获取扩展名（需要提前获取）
    ext = os.path.splitext(safe_name)[1].lower()

    # 保存上传文件
    save_path = os.path.abspath(os.path.join(dataset_folder, safe_name))

    # 如果文件已存在，添加时间戳
    if os.path.exists(save_path):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name, ext = os.path.splitext(safe_name)
        safe_name = f"{name}_{timestamp}{ext}"
        save_path = os.path.abspath(os.path.join(dataset_folder, safe_name))
        ext = ext.lower()  # 更新扩展名

    with open(save_path, "wb") as f:
        f.write(await file.read())

    # 大数据集（>50MB）走异步任务模式，不阻塞上传请求
    if os.path.getsize(save_path) > ASYNC_UPLOAD_THRESHOLD:
        task_id = f"up_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        upload_tasks[task_id] = {"status": "running"}
        threading.Thread(
            target=_upload_worker,
            args=(task_id, save_path, original_name),
            daemon=True,
        ).start()
        return {
            "code": 0,
            "msg": "上传成功（大数据集异步解析中）",
            "async": True,
            "task_id": task_id,
            "data": {
                "filename": original_name,
                "file_path": save_path,
                "async": True,
            },
        }

    # 中小数据集：同步解析，立即返回完整信息
    try:
        info = _process_upload(save_path, original_name)
    except Exception as e:
        return {"code": -1, "msg": f"读取文件失败: {str(e)}"}
    return {"code": 0, "msg": "上传成功", "data": info}

@app.get("/upload_status")
async def upload_status(task_id: str = Query()):
    """查询大数据集异步上传任务状态。"""
    task = upload_tasks.get(task_id)
    if not task:
        return {"code": -1, "msg": "任务不存在或已过期"}
    if task["status"] == "running":
        return {"code": 0, "msg": "处理中", "data": {"status": "running"}}
    if task["status"] == "error":
        return {"code": -1, "msg": task.get("error", "解析失败"), "data": {"status": "error"}}
    return {"code": 0, "msg": "上传成功", "data": {"status": "done", "info": task["result"]}}

# 图片集上传接口（zip 打包，解压到 uploads/image_sets/ 供后续模块使用）
@app.post("/chat_analyze")
def chat_analyze(
    file_path: str = Query(),
    user_query: str = Query(),
    model_name: str = Query(default=DEFAULT_MODEL),
    session_id: str | None = Query(None),
    temperature: float | None = Query(None),
):
    # 会话管理
    if not session_id or session_id not in chat_sessions:
        session_id = str(uuid.uuid4())
        chat_sessions[session_id] = []
    # 追加本轮问题到回话
    chat_sessions[session_id].append({"role": "user", "content": user_query})

    # 检验文件存在性
    if not os.path.exists(file_path):
        return {"code": -1, "msg": "数据集文件不存在，请先上传数据集"}

    # 额外安全校验：只允许读取upload目录下的文件，防止路径穿越
    abs_upload = os.path.abspath(dataset_folder)
    abs_cleaned = os.path.abspath(cleaned_folder)
    abs_file = os.path.abspath(file_path)
    if not (abs_file.startswith(abs_upload) or abs_file.startswith(abs_cleaned)):
        return {"code": -1, "msg": "禁止访问upload目录以外的文件"}

    # 读取数据集
    try:
        if file_path.lower().endswith(".csv"):
            df = pd.read_csv(file_path)
        elif file_path.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(file_path)
        else:
            return {"code": -1, "msg": "不支持该文件格式"}
    except Exception as e:
        return {"code": -1, "msg": f"读取数据集失败:{str(e)}"}

    # 列名统一为小写（去首尾空格），避免模型大小写幻觉导致 KeyError
    df.columns = [str(c).strip().lower() for c in df.columns]

    # 前N行采样
    max_sample_rows = 10
    df_sample = df.head(max_sample_rows)
    sample_text = df_sample.to_string()

    # 生成pandas代码，prompt强化安全约束
    code_prompt = f"""
数据集信息：
- 列名：{list(df.columns)}
- 数据类型：
{df.dtypes.to_string()}
- 前5行：
{df.head(5).to_string()}

用户问题：{user_query}

【严格规则】
只输出纯Python代码,不要任何文字解释。
可用:pd, plt, np
- 禁止使用 .set() 方法（如 .set(markers=...) 会报错），禁止 plt.show()。
- 绘图只允许最简单的写法（不要自定义颜色/线型/标记参数）：
  df['列名'].hist(bins=30)                    # 直方图
  df['列名'].value_counts().plot(kind='bar')   # 计数柱状图
  df['列名'].plot(kind='line')                 # 折线图
  df.plot.scatter(x='列A', y='列B')            # 散点图
  画完后必须用下方【绘图必须用这个格式】的保存代码。
- 数据集列名都是小写，绘图时直接用 df['列名']（如 df['age']）。
- 图片文件名必须用 uuid.uuid4()，不是 np.random.uuid4()。

【绘图必须用这个格式】
plt.figure(figsize=(10,6))
df['列名'].操作
plt.title('标题')
img_path = f"./uploads/imgs/{{uuid.uuid4()}}.png"
plt.savefig(img_path, dpi=120, bbox_inches='tight')
print(f"IMAGE_PATH:{{img_path}}")
plt.close()

【示例】count列直方图:
plt.figure(figsize=(10,6))
df['count'].hist(bins=30)
plt.title('count分布')
img_path = f"./uploads/imgs/{{uuid.uuid4()}}.png"
plt.savefig(img_path, dpi=120, bbox_inches='tight')
print(f"IMAGE_PATH:{{img_path}}")
plt.close()

现在写代码：
"""

    # 调用ollama
    ollama_code_res = call_ollama(code_prompt, model_name=model_name, stream=False, temperature=temperature)
    if "error" in ollama_code_res:
        return {"code": -1, "msg": ollama_code_res["error"]}

    raw_code = ollama_code_res.get("response", "")

    # 打印原始返回，便于调试
    print("=" * 60)
    print("Ollama 原始返回:")
    print(raw_code)
    print("=" * 60)

    clean_code = extract_python_code(raw_code)
    clean_code = neutralize_df_reload(clean_code)

    # 打印提取后的代码
    print("提取后的代码:")
    print(clean_code if clean_code else "(空)")
    print("=" * 60)

    if not clean_code or not clean_code.strip():
        return {"code": -1, "msg": "未能提取到有效的Python代码，请检查模型返回格式"}

    exec_info = ""
    image_base64 = None
    ok, pre_msg = safe_code_precheck(clean_code)

    def _run_plot(code: str):
        """捕获 stdout 执行模型代码，返回 (stdout_text, err)。"""
        old_stdout = sys.stdout
        mystdout = StringIO()
        sys.stdout = mystdout
        err = ""
        exec_globals = {
                "np": np,
                "pd": pd,
                "plt": plt,
                "uuid": uuid,
                "__builtins__": __builtins__,
        }
        exec_locals = {"df": df, "_result": None}
        try:
            exec(code, exec_globals, exec_locals)
        except Exception as e:
            err = str(e)
        finally:
            sys.stdout = old_stdout
            plt.close("all")
        return mystdout.getvalue(), err

    if not ok:
        exec_info = f"安全拦截:{pre_msg}"
    else:
        try:
            compile(clean_code, "<ai_generated>", "exec")
        except SyntaxError as e:
            exec_info = f"语法错误:{str(e)}"
        else:
            stdout_text, err = _run_plot(clean_code)
            retried = False
            if err and ("unexpected keyword" in err or "markers" in err or "got an unexpected" in err):
                fixed = fix_plot_code(clean_code)
                if fixed != clean_code:
                    stdout_text2, err2 = _run_plot(fixed)
                    if not err2:
                        stdout_text, err, retried = stdout_text2, "", True
            exec_info = stdout_text + ("\n" if stdout_text else "")
            if retried:
                exec_info += "（已自动修复绘图参数后重试成功）\n"
            if err:
                exec_info += f"执行异常:{err}"
            # 解析图片标记
            if "IMAGE_PATH:" in exec_info:
                img_pattern = r"IMAGE_PATH:(.+?)(?:\n|$)"
                matches = re.findall(img_pattern, exec_info)
                if matches:
                    img_path = matches[0].strip()
                    if os.path.exists(img_path):
                        with open(img_path, "rb") as f:
                            image_base64 = base64.b64encode(f.read()).decode("utf-8")

    # 生成总计报告
    summary_prompt = f"""
    数据集样本：
    {sample_text}
    用户问题：{user_query}

    下面是pandas运行的真实输出：
    {exec_info}

    要求：
    1. 严格基于运行输出做分析，禁止编造不存在的数据。
    2. 如果代码报错，明确说明报错原因，不做虚假推断。
    3. 输出简洁清晰的数据分析报告。
"""

    # 流式生成分析报告：文本逐块推送，图表在文本结束后推送，最后推送 done 收尾事件
    def _gen_stream():
        final_reply = ""
        try:
            for chunk in call_ollama_stream(summary_prompt, model_name, temperature=temperature):
                if chunk is None:
                    break
                final_reply += chunk
                yield f"data: {json.dumps({'type': 'text', 'content': chunk}, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'text', 'content': '❌ 生成报告失败:' + str(e)}, ensure_ascii=False)}\n\n"
        if image_base64:
            yield f"data: {json.dumps({'type': 'image', 'title': '分析图表', 'image_base64': image_base64}, ensure_ascii=False)}\n\n"
        chat_sessions[session_id].append({"role": "assistant", "content": final_reply or "（模型未返回分析文本）"})
        yield f"data: {json.dumps({'type': 'done', 'reply': final_reply, 'generated_code': clean_code, 'exec_info': exec_info, 'session_id': session_id, 'image_base64': image_base64, 'model_name': model_name}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(_gen_stream(), media_type="text/event-stream")
