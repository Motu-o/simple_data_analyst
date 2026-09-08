# -*- coding: utf-8 -*-
"""公共工具函数：Ollama 调用、模型代码提取/修复、JSON 容错提取、安全预检、
数据文件安全读取、matplotlib 中文字体设置。"""
import os
import re
import json
import requests
import pandas as pd
import matplotlib.pyplot as plt

from config import forbidden_keywords, ollama_url, dataset_folder, cleaned_folder, split_folder


def set_chinese_font():
    """设置 matplotlib 中文字体，避免图表标题乱码。"""
    try:
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass


def safe_code_precheck(code: str) -> tuple[bool, str]:
    """静态预扫描，拦截高危关键字。"""
    lower_code = code.lower()
    for kw in forbidden_keywords:
        if kw in lower_code:
            return False, f"安全拦截，检测到禁止关键字`{kw}`"
    return True, ""


def fix_plot_code(code: str) -> str:
    """自动修复模型绘图代码中的高频参数错误：
    1) matplotlib Line2D.set() 常见拼写错误 markers -> marker（单数）
    2) 移除无法识别的 .set(...) 美化调用（非绘图必需，避免抛未知参数异常）
    """
    code = re.sub(r"\bmarkers\s*=", "marker=", code)
    code = re.sub(r"\.set\([^()]*\)", "", code)
    return code


def neutralize_df_reload(code: str) -> str:
    """移除模型代码中重新读取数据集的语句（沙盒已注入 df，模型不应再 read 文件）。
    防止模型幻觉 df = pd.read_csv('不存在的文件') 覆盖注入的数据。"""
    out = []
    for line in code.splitlines():
        s = line.strip()
        if re.match(r"^df\s*=\s*pd\.read_(?:csv|excel)\s*\(", s):
            continue
        if re.match(r"^df\s*=\s*read_(?:csv|excel)\s*\(", s):
            continue
        out.append(line)
    return "\n".join(out)


def extract_python_code(text: str) -> str:
    """提取Python代码，保留所有代码内容"""
    if not text:
        return ""

    # 提取代码块
    match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # 如果没有代码块标记，尝试清理
    code = re.sub(r'^```.*$', '', text, flags=re.MULTILINE)
    code = re.sub(r'^python\s*$', '', code, flags=re.MULTILINE)
    return code.strip()


def normalize_fullwidth(code: str) -> str:
    """把全角字符（标点/字母/数字/空格）转成半角，防止模型生成的代码混入
    中文标点（如全角冒号：、全角逗号，）导致 Python 语法错误。"""
    code = code.replace("\u3000", " ")  # 全角空格 -> 半角空格
    out = []
    for ch in code:
        o = ord(ch)
        if 0xFF01 <= o <= 0xFF5E:  # 全角 ASCII 区（含标点/字母/数字）
            out.append(chr(o - 0xFEE0))
        else:
            out.append(ch)
    return "".join(out)


def extract_json(text: str):
    """从模型输出中容错提取 JSON（兼容代码块/解释文字/全角字符/对象或数组）"""
    if not text:
        return None
    m = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        text = m.group(1)
    text = normalize_fullwidth(text).strip()
    # 1) 整体解析
    try:
        return json.loads(text)
    except Exception:
        pass
    # 2) 数组形式 [ ... ]（优先，避免"带解释+数组"被截断成单个元素）
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            pass
    # 3) 对象形式 { ... }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            pass
    return None


def call_ollama(prompt: str, model_name: str, stream: bool = True, timeout: int = 120, temperature: float | None = None, num_predict: int | None = None):
    """调用本地 Ollama 模型接口。

    temperature: 可选采样温度（0~1），None 表示使用模型默认值。
    """
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": stream
    }
    opts = {}
    if temperature is not None:
        opts["temperature"] = float(temperature)
    if num_predict is not None:
        opts["num_predict"] = int(num_predict)
    if opts:
        payload["options"] = opts
    try:
        resp = requests.post(ollama_url, json=payload, timeout=timeout)
        if resp.status_code != 200:
            return {"error": f"ollama错误{resp.status_code}:{resp.text}"}
        return resp.json()
    except Exception as e:
        return {"error": f"请求ollama失败:{str(e)}"}


def call_ollama_stream(prompt: str, model_name: str, temperature: float | None = None, num_predict: int | None = None):
    """流式调用 Ollama，逐块 yield 文本增量。出错时 yield None 后结束。
    用于大模型对话流式输出（SSE）。"""
    payload = {"model": model_name, "prompt": prompt, "stream": True}
    opts = {}
    if temperature is not None:
        opts["temperature"] = float(temperature)
    if num_predict is not None:
        opts["num_predict"] = int(num_predict)
    if opts:
        payload["options"] = opts
    try:
        with requests.post(ollama_url, json=payload, stream=True, timeout=300) as resp:
            if resp.status_code != 200:
                yield None
                return
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except Exception:
                    continue
                chunk = data.get("response", "")
                if chunk:
                    yield chunk
                if data.get("done"):
                    break
    except Exception:
        yield None


def load_df_checked(file_path: str):
    """读取数据集（仅放行 datasets/cleaned 目录，防止路径穿越），
    并兼容 pandas 3.x 的 string 类型列转为 object。
    返回 DataFrame；校验失败抛异常。"""
    abs_upload = os.path.abspath(dataset_folder)
    abs_cleaned = os.path.abspath(cleaned_folder)
    abs_split = os.path.abspath(split_folder)
    abs_file = os.path.abspath(file_path)
    if not (abs_file.startswith(abs_upload) or abs_file.startswith(abs_cleaned) or abs_file.startswith(abs_split)):
        raise PermissionError("禁止访问upload目录以外的文件")
    if not os.path.exists(abs_file):
        raise FileNotFoundError("数据集文件不存在，请先上传数据集")
    if abs_file.lower().endswith(".csv"):
        df = pd.read_csv(abs_file)
    elif abs_file.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(abs_file)
    else:
        raise ValueError("不支持该文件格式")
    # 兼容 pandas 3.0：读取后 string 列默认是 str 类型，转为 object
    for _col in df.select_dtypes(include=["string"]).columns:
        df[_col] = df[_col].astype(object)
    return df
