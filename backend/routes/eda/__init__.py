# -*- coding: utf-8 -*-
"""EDA 数据分析模块：确定性图表生成（含绘制过程动画帧）、按需绘图、大模型问答建议、打包下载。"""
import os
import re
import json
import base64
import uuid
import zipfile
from typing import List, Dict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fastapi import Query
from fastapi.responses import FileResponse
from datetime import datetime

from config import app, img_folder, eda_export_folder, DEFAULT_MODEL
from utils import call_ollama, set_chinese_font, load_df_checked

# 语义同义词：把用户口语中的词映射到真实列名（按列名存在性判断）
SEMANTIC_HINTS = [
    (["舱位", "客舱", "仓位", "pclass", "等级"], "Pclass"),
    (["性别", "性別"], "Sex"),
    (["年龄", "年纪"], "Age"),
    (["票价", "船票"], "Fare"),
    (["存活", "生存", "survived"], "Survived"),
    (["名字", "姓名", "名称", "name"], "Name"),
    (["兄弟姐妹", "配偶", "sibsp"], "SibSp"),
]

def llm_parse_plot_intent(query: str, df: pd.DataFrame, model_name: str) -> Dict:
    """用大模型分析用户绘图意图，返回结构化结果 {chart_type, columns, reason}。
    解析失败时抛异常（调用方回退到纯规则解析）。"""
    col_names = [str(c) for c in df.columns]
    num_cols = [str(c) for c in df.select_dtypes(include="number").columns]
    cat_cols = [str(c) for c in df.columns if str(c) not in num_cols]
    prompt = f"""你是专业的数据可视化助手。用户想针对数据集绘制图表，请分析用户的绘图意图，只返回一个 JSON 对象（不要输出任何其他文字，不要用 Markdown 代码块）：
{{
  "chart_type": "line|hist|bar|box|pie|scatter|corr|missing",
  "columns": ["需要用到的列名"],
  "reason": "用一句中文说明选择该图表和列的理由"
}}

图表类型说明：
- line: 折线图（观测某数值列随行/时间趋势）
- hist: 直方图（单个或多个数值列分布）
- bar: 柱状图（类别对比、取值分布、比率对比）
- box: 箱线图（数值分布与离群点，可结合类别分组）
- pie: 饼图（类别占比）
- scatter: 散点图（两个数值列的关系）
- corr: 相关性热力图（所有数值列两两相关性）
- missing: 缺失值柱状图（各列缺失数量）

数据集可用列：
- 数值列：{num_cols}
- 类别列：{cat_cols}

用户要求：{query}

要求：
1. columns 只能从上述可用列中选，不要编造列名。
2. 若用户明确提到的列在数据集中不存在，columns 返回 []，并在 reason 中写"数据集中不存在列：<列名>"。
3. 若用户未指定图表类型，根据语义推断最合适的类型与列。"""
    res = call_ollama(prompt, model_name=model_name, stream=False, timeout=120)
    if "error" in res:
        raise ValueError(res["error"])
    text = res.get("response", "").strip()
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("模型未返回有效 JSON")
    obj = json.loads(m.group(0))
    ct = str(obj.get("chart_type", "")).strip().lower()
    valid = {"line", "hist", "bar", "box", "pie", "scatter", "corr", "missing", "violin"}
    if ct not in valid:
        ct = ""
    cols = [str(c) for c in (obj.get("columns") or [])]
    cols = [c for c in cols if c in col_names]
    return {"chart_type": ct, "columns": cols, "reason": str(obj.get("reason", ""))}

def _box_stats(series):
    """计算一组数值的箱线图统计（q1/中位数/q3/须边界/异常点）。"""
    s = pd.Series(series).dropna().astype(float).values
    if len(s) == 0:
        return None
    q1, med, q3 = np.percentile(s, [25, 50, 75])
    iqr = q3 - q1
    lo = q1 - 1.5 * iqr
    hi = q3 + 1.5 * iqr
    s_in = s[(s >= lo) & (s <= hi)]
    out = [float(x) for x in s[(s < lo) | (s > hi)]]
    return {
        "min": float(s_in.min()),
        "q1": float(q1),
        "med": float(med),
        "q3": float(q3),
        "max": float(s_in.max()),
        "outliers": out[:200],
    }

def _violin_stats(series, n=80):
    """计算一组数值的箱线统计 + 核密度估计（用于小提琴图）。"""
    s = pd.Series(series).dropna().astype(float).values
    if len(s) == 0:
        return None
    q1, med, q3 = np.percentile(s, [25, 50, 75])
    iqr = q3 - q1
    lo = q1 - 1.5 * iqr
    hi = q3 + 1.5 * iqr
    s_in = s[(s >= lo) & (s <= hi)]
    out = [float(x) for x in s[(s < lo) | (s > hi)]]
    # 核密度估计（scipy gaussian_kde），归一化到 [0,1] 供前端缩放
    try:
        grid = np.linspace(float(s.min()), float(s.max()), n)
        if len(np.unique(s)) < 3:
            dens = np.ones(n)
        else:
            from scipy.stats import gaussian_kde
            dens = gaussian_kde(s, bw_method="scott")(grid)
            dmax = dens.max()
            if dmax > 0:
                dens = dens / dmax
        kde = {"x": [float(v) for v in grid], "y": [float(v) for v in dens]}
    except Exception:
        kde = {"x": [], "y": []}
    return {
        "min": float(s_in.min()),
        "q1": float(q1),
        "med": float(med),
        "q3": float(q3),
        "max": float(s_in.max()),
        "outliers": out[:200],
        "kde": kde,
    }

def parse_plot_request(query: str, df: pd.DataFrame, llm_intent: dict = None):
    """用纯规则解析用户的绘图/分析要求 → (chart_type, cols, meta)。
    支持组合语义：对比/比率（存活率、均值）、整体分布、按列分组等。
    无法识别时抛 ValueError（附可用图表类型提示）。"""
    raw_q = str(query).strip()
    q = raw_q.lower()
    if not q:
        raise ValueError("请输入绘图/分析要求，如：绘制 Age 的直方图 / Sex 与 Survived 的关系 / 相关性热力图")
    col_names = [str(c) for c in df.columns]
    num_cols = [str(c) for c in df.select_dtypes(include="number").columns]
    cat_cols = [str(c) for c in df.columns if str(c) not in num_cols]

    def find_cols() -> List[str]:
        found = []
        for c in sorted(col_names, key=len, reverse=True):
            if c and c.lower() in q and c not in found:
                found.append(c)
        # 语义同义词补充（如"舱位"→Pclass、"性别"→Sex）
        for kws, tgt in SEMANTIC_HINTS:
            if tgt in col_names and tgt not in found and any(k in q for k in kws):
                found.append(tgt)
        return found

    # 校验用户明确提到的类列名是否存在：存在才绘图，不存在则明确报错（而不是静默绘制其他列）
    _ident_re = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
    _stop_tokens = {
        "hist", "bar", "box", "pie", "line", "scatter", "corr", "plot", "chart",
        "figure", "fig", "the", "and", "of", "for", "by", "with", "vs", "to",
        "in", "on", "at", "a", "an", "is", "are", "was", "were", "data",
        "distribution", "frequency", "value", "values", "draw", "show", "make",
        "histogram", "heatmap", "count", "counts", "number", "column", "columns",
        "feature", "features", "using", "use", "over", "across", "from", "each",
        "plotting", "analysis", "analytics", "boxplot", "scatterplot", "lineplot",
        "violin", "violinplot",
    }
    for _t in _ident_re.findall(raw_q):
        _tl = _t.lower()
        if _tl in _stop_tokens:
            continue
        if any(str(c).lower() == _tl for c in col_names):
            continue
        raise ValueError(
            f"数据集中不存在列：{_t}。\n可用列：{', '.join(col_names)}"
        )

    def _low_card(c: str) -> bool:
        try:
            return 0 < df[c].dropna().nunique() <= 10
        except Exception:
            return False

    meta = {"rate": False, "target": None, "group": None, "overall": False}

    # 1. 整体分布：所有数值列并列分布对比（"整体分布/所有数值列/分布对比"）
    if any(k in q for k in ("整体", "所有数值", "全部数值", "所有数字")) and any(k in q for k in ("分布", "对比", "比较", "直方", "图")):
        meta["overall"] = True
        if llm_intent and llm_intent.get("columns"):
            return "hist", llm_intent["columns"], meta
        return "hist_overall", find_cols(), meta

    # 2. 比率/均值统计（"存活率/占比/均值/平均"）→ 找目标列
    rate = any(k in q for k in ("率", "占比", "比例", "均值", "平均"))
    target = None
    if rate:
        if any(k in q for k in ("存活", "生存", "通过", "命中", "优惠", "中奖")):
            # 找 0/1 布尔列（如 Survived）
            for c in num_cols:
                s = df[c].dropna()
                try:
                    uniq = set(s.astype(int).unique())
                except Exception:
                    continue
                if len(s) and uniq <= {0, 1} and c.lower() not in ("passengerid", "id"):
                    target = c
                    break
        if target is None:
            found = find_cols()
            target = next((c for c in found if c in num_cols), None) or (num_cols[0] if num_cols else None)
    meta["rate"] = rate
    meta["target"] = target

    # 3. 分组列（"按X/根据X/各X/每X"）→ 类别列或低基数整数列
    if any(k in q for k in ("按", "根据", "各", "每个", "分")):
        found = find_cols()
        group = next((c for c in found if c in cat_cols), None)
        if group is None:
            group = next((c for c in found if c in num_cols and _low_card(c) and c != target), None)
        meta["group"] = group

    # 4. 图类型
    chart_type = None
    if any(k in q for k in ("相关", "corr", "热力")):
        chart_type = "correlation"
    elif any(k in q for k in ("直方", "hist", "分布图", "频数")):
        chart_type = "hist"
    elif any(k in q for k in ("折线", "line", "趋势")):
        chart_type = "line"
    elif any(k in q for k in ("散点", "scatter", "关系")):
        chart_type = "scatter"
    elif any(k in q for k in ("箱线", "box", "箱型")):
        chart_type = "box"
    elif any(k in q for k in ("小提琴", "violin")):
        chart_type = "violin"
    elif any(k in q for k in ("饼图", "pie")):
        chart_type = "pie"
    elif any(k in q for k in ("缺失", "missing", "缺失值")):
        chart_type = "missing"
    elif any(k in q for k in ("柱状", "bar", "条形", "对比", "分布")):
        chart_type = "bar"
    elif meta["rate"] and (meta["target"] or meta["group"]):
        chart_type = "bar"

    if not chart_type:
        # 没指定图类型但有列名：默认按列类型选（类别→柱状，数值→直方图）
        cols_found = find_cols()
        if cols_found:
            c0 = cols_found[0]
            chart_type = "bar" if c0 in cat_cols else "hist"
        else:
            raise ValueError(
                "无法识别图表类型。支持的绘图要求示例：\n"
                "· 相关性热力图\n"
                "· Age 直方图 / Age 折线图 / Age 与 Survived 的散点图\n"
                "· Sex 的柱状图（取值分布）/ Sex 与 Age 的箱线图\n"
                "· 各列缺失值柱状图 / 类别占比饼图\n"
                "· 各舱位存活率对比 / 数值列整体分布 / 按 Sex 对比 Age 均值"
            )
    # 大模型意图覆盖（若已给出合法图类型）；用户显式指定"小提琴"时以规则为准
    if "小提琴" in raw_q or "violin" in q:
        chart_type = "violin"
    elif llm_intent and llm_intent.get("chart_type") in {"correlation", "hist", "line", "scatter", "box", "pie", "missing", "bar", "violin"}:
        chart_type = llm_intent["chart_type"]
    # 列：大模型推荐优先，否则规则提取
    cols = find_cols()
    if llm_intent and llm_intent.get("columns"):
        cols = llm_intent["columns"]
    return chart_type, cols, meta

def _shot_frame(fig, tag, save_dir):
    """把当前 fig 状态保存为一帧 PNG(base64)，用于前端实时展现绘制过程。"""
    fname = f"anim_{tag}_{uuid.uuid4().hex[:8]}.png"
    p = os.path.join(save_dir or img_folder, fname)
    try:
        fig.savefig(p, dpi=90, bbox_inches="tight")
        with open(p, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        try:
            os.remove(p)
        except Exception:
            pass
        return b64
    except Exception:
        return None

def _cum_slices(n, steps=8):
    """把 n 个元素切成 steps 步的累积切分点（去重），如 n=10,steps=8 → [0,1,3,4,5,6,8,9,10]。"""
    if n <= 0:
        return [0]
    if n <= steps:
        return list(range(0, n + 1))
    res = [0]
    for i in range(1, steps):
        res.append(int(round(n * i / steps)))
    res.append(n)
    out = []
    for v in res:
        if v not in out:
            out.append(v)
    return out

def draw_plot(df: pd.DataFrame, chart_type: str, cols: List[str], save_dir: str = None, meta: dict = None, animate: bool = True) -> List[Dict]:
    """按用户绘图要求，用 pandas/matplotlib 独立绘制图像（确定性，不依赖大模型）。
    返回 [{title, filename, image_base64, frames?}]；animate=True 时附带 frames 过程帧供前端实时展现绘制过程。"""
    meta = meta or {}
    set_chinese_font()

    num_cols = [str(c) for c in df.select_dtypes(include="number").columns]
    cat_cols = [str(c) for c in df.columns if str(c) not in num_cols]
    charts: List[Dict] = []

    def _save():
        fname = f"plot_{uuid.uuid4().hex[:12]}.png"
        img_path = os.path.join(save_dir or img_folder, fname)
        plt.savefig(img_path, dpi=100, bbox_inches="tight")
        plt.close("all")
        with open(img_path, "rb") as f:
            return {"filename": fname, "image_base64": base64.b64encode(f.read()).decode("utf-8")}

    def _pick_numeric_cols():
        # 从要求列中选数值列；没有则用全部数值列
        picked = [c for c in cols if c in num_cols]
        return picked or num_cols

    def _render_animated(figsize, n, tag, setup, partial, finalize=None):
        """绘制过程动画：背景帧 → 数据分步增长 → 完成帧。返回 (fig, frames)。
        setup(ax) 设背景样式；partial(ax, k) 画前 k 个元素；finalize(ax) 收尾标注。"""
        frames: List[str] = []
        fig, ax = plt.subplots(figsize=figsize)

        def _rec():
            if not animate:
                return
            b64 = _shot_frame(fig, tag, save_dir or img_folder)
            if b64:
                frames.append(b64)

        setup(ax)
        _rec()
        if animate:
            for k in _cum_slices(max(n, 1), 8)[1:]:
                ax.cla()
                setup(ax)
                partial(ax, k)
                _rec()
        ax.cla()
        setup(ax)
        partial(ax, n)
        if finalize:
            finalize(ax)
        _rec()
        fig.tight_layout()
        return fig, frames

    if chart_type == "correlation":
        if len(num_cols) < 2:
            return []
        corr = df[num_cols].corr(numeric_only=True)
        n = len(corr)
        from matplotlib.patches import Rectangle
        from matplotlib import cm
        norm = plt.Normalize(vmin=-1, vmax=1)
        cmap = cm.RdBu_r
        cells = [(i, j) for i in range(n) for j in range(n)]

        def _setup(ax):
            ax.set_xticks(range(n))
            ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=9)
            ax.set_yticks(range(n))
            ax.set_yticklabels(corr.columns, fontsize=9)
            ax.set_xlim(-0.5, n - 0.5)
            ax.set_ylim(n - 0.5, -0.5)
            ax.set_title("数值列相关性热力图", fontsize=13)
            for i, j in cells:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="white", linewidth=1))

        def _partial(ax, k):
            for idx in cells[:k]:
                i, j = idx
                v = corr.iloc[i, j]
                if not pd.isna(v):
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=cmap(norm(v)), edgecolor="white", linewidth=1))

        def _finalize(ax):
            for i, j in cells:
                v = corr.iloc[i, j]
                if not pd.isna(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7)

        fig, frames = _render_animated((max(7, 0.9 * n), max(6, 0.8 * n)), n * n, "corr", _setup, _partial, _finalize)
        # 最终保存一张带 colorbar 的完整热力图
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(max(7, 0.9 * n), max(6, 0.8 * n)))
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(n))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(n))
        ax.set_yticklabels(corr.columns, fontsize=9)
        for i, j in cells:
            v = corr.iloc[i, j]
            if not pd.isna(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7)
        plt.colorbar(im, ax=ax)
        ax.set_title("数值列相关性热力图", fontsize=13)
        fig.tight_layout()
        r = _save(); r["title"] = "数值列相关性热力图"
        if frames:
            r["frames"] = frames
        r["series"] = {
            "type": "corr",
            "cols": [str(x) for x in corr.columns],
            "matrix": [[(None if pd.isna(corr.iloc[i, j]) else round(float(corr.iloc[i, j]), 3)) for j in range(n)] for i in range(n)],
        }
        charts.append(r)

    elif chart_type == "hist":
        plot_cols = _pick_numeric_cols()[:6]
        for c in plot_cols:
            s = df[c].dropna()
            if len(s) == 0:
                continue
            data = s.values
            vmin, vmax = float(data.min()), float(data.max())
            if vmin == vmax:
                vmin, vmax = vmin - 1, vmax + 1
            _, edges = np.histogram(data, bins=30, range=(vmin, vmax))
            centers = (edges[:-1] + edges[1:]) / 2
            bw = edges[1] - edges[0]

            def _setup(ax):
                ax.set_title(f"{c} 直方图", fontsize=12)
                ax.set_xlabel(c)
                ax.set_ylabel("频数")

            def _partial(ax, k):
                sub = data[:k]
                if len(sub):
                    cc, _ = np.histogram(sub, bins=30, range=(vmin, vmax))
                    ax.bar(centers, cc, width=bw * 0.9, color="steelblue", edgecolor="white")

            fig, frames = _render_animated((9, 4), len(data), "hist", _setup, _partial)
            r = _save(); r["title"] = f"{c} 直方图"
            if frames:
                r["frames"] = frames
            # 供前端 Chart.js 交互式渲染的直方图数据（bin 中心 + 频数）
            cc_full, _ = np.histogram(data, bins=30, range=(vmin, vmax))
            r["series"] = {
                "type": "hist",
                "labels": [round(float(x), 4) for x in centers],
                "counts": [int(x) for x in cc_full],
                "col": c,
                "total": int(len(data)),
            }
            charts.append(r)

    elif chart_type == "hist_overall":
        # 数值列整体分布：多列并列直方图（整体同步增长动画）
        plot_cols = [c for c in cols if c in num_cols] or num_cols
        plot_cols = plot_cols[:6]
        if not plot_cols:
            return []
        n = len(plot_cols)
        hists = []
        for c in plot_cols:
            s = df[c].dropna()
            if len(s) == 0:
                hists.append(None)
                continue
            data = s.values
            vmin, vmax = float(data.min()), float(data.max())
            if vmin == vmax:
                vmin, vmax = vmin - 1, vmax + 1
            _, edges = np.histogram(data, bins=30, range=(vmin, vmax))
            centers = (edges[:-1] + edges[1:]) / 2
            hists.append({"centers": centers, "edges": edges, "data": data, "range": (vmin, vmax), "bw": edges[1] - edges[0]})
        frames: List[str] = []
        maxlen = max((len(h["data"]) if h else 0) for h in hists)
        fig, axes = plt.subplots(nrows=1, ncols=n, figsize=(4.0 * n, 4))
        if n == 1:
            axes = [axes]
        fig.suptitle("数值列整体分布对比", fontsize=13)

        def _draw(k):
            for ax, c, h in zip(axes, plot_cols, hists):
                ax.clear()
                ax.set_title(c, fontsize=11)
                ax.set_xlabel(c)
                ax.set_ylabel("频数")
                if h is None:
                    continue
                sub = h["data"][:k]
                if len(sub):
                    cc, _ = np.histogram(sub, bins=30, range=h["range"])
                    ax.bar(h["centers"], cc, width=h["bw"] * 0.9, color="steelblue", edgecolor="white")

        def _rec():
            if animate:
                b64 = _shot_frame(fig, "histall", save_dir or img_folder)
                if b64:
                    frames.append(b64)

        _draw(0)
        _rec()
        if animate:
            for k in _cum_slices(max(maxlen, 1), 8)[1:]:
                _draw(k)
                _rec()
        _draw(maxlen)
        _rec()
        fig.tight_layout()
        r = _save(); r["title"] = "数值列整体分布对比"
        if frames:
            r["frames"] = frames
        # 供前端 Chart.js 交互式渲染的多列直方图数据
        items = []
        for h, col in zip(hists, plot_cols):
            if h is None:
                continue
            cc_all, _ = np.histogram(h["data"], bins=30, range=h["range"])
            items.append({
                "col": col,
                "labels": [round(float(x), 4) for x in h["centers"]],
                "counts": [int(x) for x in cc_all],
            })
        r["series"] = {"type": "hist_all", "cols": [it["col"] for it in items], "items": items}
        charts.append(r)

    elif chart_type == "line":
        plot_cols = _pick_numeric_cols()[:4]
        for c in plot_cols:
            s = df[c].dropna().reset_index(drop=True)
            if len(s) < 2:
                continue
            xs = s.index.values
            ys = s.values

            def _setup(ax):
                ax.set_title(f"{c} 折线图", fontsize=12)
                ax.set_xlabel("行索引")
                ax.set_ylabel(c)
                ax.grid(alpha=0.3)

            def _partial(ax, k):
                if k >= 2:
                    ax.plot(xs[:k], ys[:k], marker="o", markersize=2.5, linewidth=1, color="#2E8B57")

            fig, frames = _render_animated((9, 3.6), len(ys), "line", _setup, _partial)
            r = _save(); r["title"] = f"{c} 折线图"
            if frames:
                r["frames"] = frames
            # 供前端 Chart.js 交互式渲染的数值序列。
            # 只观测趋势：按总数据量动态确定抽样分度值 step（数据越多步长越大），
            # 趋势点数控制在约 60~120 个，避免过密导致折线拥挤成竖线、影响观感。
            n_pts = len(ys)
            if n_pts <= 80:
                step = 1
            elif n_pts <= 400:
                step = max(1, n_pts // 80)
            elif n_pts <= 2000:
                step = max(1, n_pts // 100)
            elif n_pts <= 10000:
                step = max(1, n_pts // 120)
            else:
                step = max(1, n_pts // 150)
            sx = xs[::step].tolist()
            sy = ys[::step].tolist()
            r["series"] = {"x": sx, "y": sy, "col": c, "step": int(step), "total": int(n_pts)}
            charts.append(r)

    elif chart_type == "scatter":
        specified = [c for c in cols if c in num_cols]
        x = y = None
        if len(specified) >= 2:
            x, y = specified[0], specified[1]
        elif len(specified) == 0 and len(cols) == 0:
            # 完全没指定列 → 默认用前两个数值列
            picked = num_cols[:2]
            if len(picked) >= 2:
                x, y = picked[0], picked[1]
        # 其余情况（指定了列但数值列不足 2 个）不静默兜底，返回空以便前端提示列名问题
        if x is None or y is None:
            return []
        # 若给出类别列，按类别着色
        group_col = None
        for c in cols:
            if c in cat_cols:
                group_col = c
        title = f"{x} 与 {y} 的散点图"
        if group_col and df[group_col].nunique() <= 10:
            xs_all, ys_all, labs = [], [], []
            for g, sub in df.groupby(group_col):
                xs_all.extend(sub[x].values)
                ys_all.extend(sub[y].values)
                labs.extend([str(g)] * len(sub))
            xs_all = np.array(xs_all)
            ys_all = np.array(ys_all)
            labs = np.array(labs)
            n_pts = len(xs_all)

            def _setup(ax):
                ax.set_xlabel(x)
                ax.set_ylabel(y)
                ax.set_title(title, fontsize=12)
                ax.grid(alpha=0.3)

            def _partial(ax, k):
                for g in np.unique(labs[:k]):
                    m = labs[:k] == g
                    ax.scatter(xs_all[:k][m], ys_all[:k][m], s=35, alpha=0.7, label=g)

            def _finalize(ax):
                ax.legend(title=group_col, fontsize=8)

            fig, frames = _render_animated((9, 5.5), n_pts, "scatter", _setup, _partial, _finalize)
        else:
            xs_all = df[x].values
            ys_all = df[y].values
            n_pts = len(xs_all)

            def _setup(ax):
                ax.set_xlabel(x)
                ax.set_ylabel(y)
                ax.set_title(title, fontsize=12)
                ax.grid(alpha=0.3)

            def _partial(ax, k):
                if k > 0:
                    ax.scatter(xs_all[:k], ys_all[:k], s=35, alpha=0.7, color="#2E8B57")

            fig, frames = _render_animated((9, 5.5), n_pts, "scatter", _setup, _partial)
        r = _save(); r["title"] = title
        if frames:
            r["frames"] = frames
        # 供前端 Chart.js 交互式渲染的散点数据（抽样 ≤800 点，过滤 NaN）
        step = max(1, n_pts // 800)
        if group_col and df[group_col].nunique() <= 10:
            pts = []
            for i in range(0, n_pts, step):
                xv, yv = float(xs_all[i]), float(ys_all[i])
                if not (np.isnan(xv) or np.isnan(yv)):
                    pts.append({"x": xv, "y": yv, "label": str(labs[i])})
            r["series"] = {"type": "scatter", "colx": x, "coly": y, "group": group_col, "points": pts}
        else:
            pts = []
            for i in range(0, n_pts, step):
                xv, yv = float(xs_all[i]), float(ys_all[i])
                if not (np.isnan(xv) or np.isnan(yv)):
                    pts.append({"x": xv, "y": yv})
            r["series"] = {"type": "scatter", "colx": x, "coly": y, "points": pts}
        charts.append(r)

    elif chart_type == "box":
        group_col = next((c for c in cols if c in cat_cols), None)
        value_cols = [c for c in cols if c in num_cols] or num_cols[:8]
        if group_col and value_cols:
            df_box = df[[group_col] + value_cols].dropna(subset=value_cols)
            for vc in value_cols[:4]:
                groups = [g for g, _ in df_box.groupby(group_col)]
                if len(groups) == 0:
                    continue

                def _setup(ax):
                    ax.set_title(f"{vc} 按 {group_col} 分组箱线图", fontsize=12)
                    ax.set_xlabel(group_col)
                    ax.set_ylabel(vc)
                    ax.grid(alpha=0.3)

                def _partial(ax, k):
                    sub = df_box[df_box[group_col].isin(groups[:k])]
                    if len(sub):
                        sub.boxplot(column=vc, by=group_col, ax=ax, grid=False)

                def _finalize(ax):
                    ax.set_title(f"{vc} 按 {group_col} 分组箱线图", fontsize=12)

                fig, frames = _render_animated((max(8, 0.8 * len(groups)), 5), len(groups), "box", _setup, _partial, _finalize)
                r = _save(); r["title"] = f"{vc} × {group_col} 箱线图"
                if frames:
                    r["frames"] = frames
                box_items = []
                for g in groups:
                    st = _box_stats(df_box[df_box[group_col].astype(str) == str(g)][vc])
                    if st:
                        st["label"] = str(g)
                        box_items.append(st)
                r["series"] = {"type": "box", "col": vc, "group_col": group_col, "items": box_items}
                charts.append(r)
        elif value_cols:
            vcols = value_cols[:8]
            xs = list(range(len(vcols)))

            def _setup(ax):
                ax.set_title("数值列箱线图", fontsize=12)
                ax.set_xticks(xs)
                ax.set_xticklabels(vcols, rotation=30, fontsize=9)

            def _partial(ax, k):
                if k > 0:
                    ax.boxplot([df[vc].dropna().values for vc in vcols[:k]], positions=xs[:k])

            fig, frames = _render_animated((max(7, 0.8 * len(vcols)), 5), len(vcols), "box", _setup, _partial)
            r = _save(); r["title"] = "数值列箱线图"
            if frames:
                r["frames"] = frames
            box_items = []
            for vc in vcols:
                st = _box_stats(df[vc].dropna())
                if st:
                    st["label"] = str(vc)
                    box_items.append(st)
            r["series"] = {"type": "box", "items": box_items}
            charts.append(r)

    elif chart_type == "violin":
        # 小提琴图：matplotlib violinplot 绘制动画帧 + scipy KDE 供前端 Chart.js 交互渲染
        group_col = next((c for c in cols if c in cat_cols), None)
        value_cols = [c for c in cols if c in num_cols] or num_cols[:8]
        if group_col and value_cols:
            df_v = df[[group_col] + value_cols].dropna(subset=value_cols)
            for vc in value_cols[:4]:
                groups = [str(g) for g in df_v[group_col].unique()]
                if len(groups) == 0:
                    continue

                def _setup(ax):
                    ax.set_title(f"{vc} 按 {group_col} 分组小提琴图", fontsize=12)
                    ax.set_xlabel(group_col)
                    ax.set_ylabel(vc)
                    ax.grid(alpha=0.3)

                def _partial(ax, k):
                    sub = df_v[df_v[group_col].isin(groups[:k])]
                    if len(sub):
                        data = [sub[sub[group_col] == g][vc].dropna().values for g in groups[:k]]
                        try:
                            ax.violinplot(data, positions=list(range(1, k + 1)), showmedians=True)
                        except Exception:
                            pass

                fig, frames = _render_animated((max(8, 0.8 * len(groups)), 5), len(groups), "violin", _setup, _partial)
                r = _save(); r["title"] = f"{vc} × {group_col} 小提琴图"
                if frames:
                    r["frames"] = frames
                items = []
                for g in groups:
                    st = _violin_stats(df_v[df_v[group_col].astype(str) == g][vc])
                    if st:
                        st["label"] = g
                        items.append(st)
                r["series"] = {"type": "violin", "col": vc, "group_col": group_col, "items": items}
                charts.append(r)
        elif value_cols:
            vcols = value_cols[:8]
            xs = list(range(1, len(vcols) + 1))

            def _setup(ax):
                ax.set_title("数值列小提琴图", fontsize=12)
                ax.set_xticks(xs)
                ax.set_xticklabels(vcols, rotation=30, fontsize=9)

            def _partial(ax, k):
                if k > 0:
                    data = [df[vc].dropna().values for vc in vcols[:k]]
                    try:
                        ax.violinplot(data, positions=xs[:k], showmedians=True)
                    except Exception:
                        pass

            fig, frames = _render_animated((max(7, 0.8 * len(vcols)), 5), len(vcols), "violin", _setup, _partial)
            r = _save(); r["title"] = "数值列小提琴图"
            if frames:
                r["frames"] = frames
            items = []
            for vc in vcols:
                st = _violin_stats(df[vc].dropna())
                if st:
                    st["label"] = str(vc)
                    items.append(st)
            r["series"] = {"type": "violin", "items": items}
            charts.append(r)

    elif chart_type == "pie":
        pie_col = next((c for c in cols if c in cat_cols), None) or (cat_cols[0] if cat_cols else None)
        if pie_col:
            vc = df[pie_col].astype(str).value_counts().head(8)
            if len(vc) >= 1:
                labels = vc.index.tolist()
                vals = vc.values

                def _setup(ax):
                    ax.axis("equal")
                    ax.set_title(f"{pie_col} 取值占比饼图", fontsize=12)

                def _partial(ax, k):
                    if k > 0:
                        ax.pie(vals[:k], labels=labels[:k], autopct="%1.1f%%", startangle=90)

                fig, frames = _render_animated((7, 6), len(vals), "pie", _setup, _partial)
                r = _save(); r["title"] = f"{pie_col} 取值占比"
                if frames:
                    r["frames"] = frames
                r["series"] = {"type": "pie", "labels": [str(x) for x in labels], "values": [int(v) for v in vals]}
                charts.append(r)

    elif chart_type == "missing":
        miss_series = df.isna().sum()
        miss_series = miss_series[miss_series > 0]
        if len(miss_series) == 0:
            return []
        xs = list(range(len(miss_series)))
        ys = miss_series.values
        labs = miss_series.index.tolist()

        def _setup(ax):
            ax.set_title("各列缺失值数量（柱状图）", fontsize=12)
            ax.set_ylabel("缺失数")
            ax.set_xticks(xs)
            ax.set_xticklabels(labs, rotation=45, fontsize=9)

        def _partial(ax, k):
            if k > 0:
                ax.bar(xs[:k], ys[:k], color="#D55E00", edgecolor="white")

        def _finalize(ax):
            for i in range(len(ys)):
                ax.text(i, ys[i], str(int(ys[i])), ha="center", va="bottom", fontsize=8)

        fig, frames = _render_animated((max(7, 0.7 * len(labs)), 4.5), len(ys), "missing", _setup, _partial, _finalize)
        r = _save(); r["title"] = "各列缺失值数量"
        if frames:
            r["frames"] = frames
        r["series"] = {"type": "bar", "labels": [str(x) for x in labs], "values": [int(v) for v in ys], "ytitle": "缺失数", "color": "#D55E00", "fmt": "int"}
        charts.append(r)

    elif chart_type == "bar":
        # 组合语义：分组+比率/均值对比（如"各舱位存活率对比""按Sex对比Age均值"）
        cat_c = meta.get("group") or next((c for c in cols if c in cat_cols), None)
        num_c = meta.get("target") or next((c for c in cols if c in num_cols), None)
        rate = bool(meta.get("rate"))
        if cat_c and num_c:
            grouped = df.groupby(cat_c)[num_c].agg(["mean", "count"])
            grouped = grouped.sort_values("mean", ascending=False)
            is_rate = False
            if rate:
                # 目标列是 0/1 布尔列时才按"率"展示（如存活率）
                try:
                    uniq = set(df[num_c].dropna().astype(int).unique())
                    is_rate = uniq <= {0, 1}
                except Exception:
                    is_rate = False
            if is_rate:
                val = grouped["mean"] * 100
                title = f"各{cat_c}的{num_c}率对比"
                ylabel = f"{num_c} 率 (%)"
                fmt = lambda v: f"{v:.1f}%"
            else:
                val = grouped["mean"]
                title = f"{num_c} 按 {cat_c} 分组的均值对比"
                ylabel = f"{num_c} 均值"
                fmt = lambda v: f"{v:.2f}"
            ys = val.values
            labs = val.index.tolist()
            xs = list(range(len(ys)))

            def _setup(ax):
                ax.set_title(title, fontsize=12)
                ax.set_ylabel(ylabel)
                ax.set_xticks(xs)
                ax.set_xticklabels(labs, rotation=45, fontsize=9)
                ax.grid(axis="y", alpha=0.3)

            def _partial(ax, k):
                if k > 0:
                    ax.bar(xs[:k], ys[:k], color="#4C72B0", edgecolor="white")

            def _finalize(ax):
                for i in range(len(ys)):
                    ax.text(i, ys[i], fmt(ys[i]), ha="center", va="bottom", fontsize=8)

            fig, frames = _render_animated((max(8, 0.8 * len(ys)), 5), len(ys), "bar", _setup, _partial, _finalize)
            r = _save(); r["title"] = title
            if frames:
                r["frames"] = frames
            r["series"] = {"type": "bar", "labels": [str(x) for x in labs], "values": [float(v) for v in ys], "ytitle": ylabel, "is_rate": is_rate, "fmt": "pct" if is_rate else "num"}
            charts.append(r)
        elif cat_c:
            vc = df[cat_c].astype(str).value_counts().head(10)
            ys = vc.values
            labs = vc.index.tolist()
            xs = list(range(len(ys)))

            def _setup(ax):
                ax.set_title(f"{cat_c} 取值分布柱状图", fontsize=12)
                ax.set_ylabel("出现次数")
                ax.set_xticks(xs)
                ax.set_xticklabels(labs, rotation=45, fontsize=9)
                ax.grid(axis="y", alpha=0.3)

            def _partial(ax, k):
                if k > 0:
                    ax.bar(xs[:k], ys[:k], color="#4C72B0", edgecolor="white")

            def _finalize(ax):
                for i in range(len(ys)):
                    ax.text(i, ys[i], str(int(ys[i])), ha="center", va="bottom", fontsize=8)

            fig, frames = _render_animated((max(8, 0.8 * len(ys)), 5), len(ys), "bar", _setup, _partial, _finalize)
            r = _save(); r["title"] = f"{cat_c} 取值分布"
            if frames:
                r["frames"] = frames
            r["series"] = {"type": "bar", "labels": [str(x) for x in labs], "values": [int(v) for v in ys], "ytitle": "出现次数", "fmt": "int"}
            charts.append(r)
        elif num_c:
            s = df[num_c].dropna()
            ys = s.values
            xs = list(range(len(ys)))

            def _setup(ax):
                ax.set_title(f"{num_c} 数值柱状图", fontsize=12)
                ax.set_ylabel(num_c)
                ax.set_xticks(xs[:: max(1, len(xs) // 10)])
                ax.tick_params(axis="x", rotation=45)

            def _partial(ax, k):
                if k > 0:
                    ax.bar(xs[:k], ys[:k], color="#4C72B0", edgecolor="white")

            fig, frames = _render_animated((9, 4.5), len(ys), "bar", _setup, _partial)
            r = _save(); r["title"] = f"{num_c} 数值柱状图"
            if frames:
                r["frames"] = frames
            r["series"] = {"type": "bar", "labels": [str(x) for x in xs], "values": [float(v) for v in ys], "ytitle": num_c, "fmt": "num"}
            charts.append(r)

    return charts

@app.post("/eda_plot")
async def eda_plot(
    file_path: str = Query(),
    query: str = Query(..., description="用户绘图/分析要求"),
    model_name: str = Query(default=DEFAULT_MODEL),
):
    """按用户输入的分析要求，后端独立运行 pandas/matplotlib 代码绘制图像（确定性，不调用大模型）。
    失败时返回结构化运行日志 logs 供前端展示。"""
    logs = []
    try:
        df = load_df_checked(file_path)
        logs.append(f"[1/5] 读取数据集：{os.path.basename(file_path)}（{len(df)} 行 × {len(df.columns)} 列）")
    except Exception as e:
        logs.append(f"[1/5] 读取数据集失败：{e}")
        return {"code": -1, "msg": str(e), "logs": logs}

    # 大模型意图分析（先分析用户输入，再根据分析结果绘图；失败则回退纯规则）
    llm_intent = None
    llm_reason = ""
    try:
        llm_intent = llm_parse_plot_intent(query, df, model_name)
        llm_reason = (llm_intent or {}).get("reason", "")
        logs.append(f"[2/5] 大模型意图分析：成功（{str(llm_reason)[:80]}）")
    except Exception as e:
        logs.append(f"[2/5] 大模型意图分析失败：{e}（已回退规则解析）")
        llm_intent = None

    # 解析绘图要求（大模型意图优先，规则兜底）
    try:
        chart_type, cols, meta = parse_plot_request(query, df, llm_intent=llm_intent)
        logs.append(f"[3/5] 解析绘图要求：类型={chart_type}，列={cols}")
    except ValueError as e:
        logs.append(f"[3/5] 解析绘图要求失败：{e}")
        return {"code": -1, "msg": str(e), "logs": logs}

    try:
        charts = draw_plot(df, chart_type, cols, save_dir=eda_export_folder, meta=meta)
        logs.append(f"[4/5] 绘制图表：生成 {len(charts)} 张")
    except Exception as e:
        import traceback
        traceback.print_exc()
        logs.append(f"[4/5] 绘制图表异常：{e}")
        return {"code": -1, "msg": f"绘图失败: {str(e)}", "logs": logs}
    if not charts:
        logs.append("[4/5] 未生成任何图表（绘图需求无法解析出可用图表）")
        return {"code": -1, "msg": "未能为该要求生成图表，请换一种表述，如：Age 直方图 / 相关性热力图 / Sex 与 Age 箱线图", "logs": logs}

    logs.append("[5/5] 绘图完成")
    return {
        "code": 0,
        "msg": "绘图完成",
        "data": {
            "chart_type": chart_type,
            "charts": charts,
            "model_name": model_name,
            "llm_reason": llm_reason,
            "llm_used": bool(llm_intent),
        },
        "logs": logs,
    }

@app.post("/eda_feature_report")
async def eda_feature_report(
    file_path: str = Query(),
    model_name: str = Query(default=DEFAULT_MODEL),
):
    """EDA 绘图前：由大模型基于数据集特征统计生成特征分析报告
    （特征重要性、特征相关性、特征工程建议）。pandas 确定性计算特征概况，模型负责解读与建议。"""
    try:
        df = load_df_checked(file_path)
    except Exception as e:
        return {"code": -1, "msg": str(e)}

    num_cols = [str(c) for c in df.select_dtypes(include="number").columns]
    cat_cols = [str(c) for c in df.columns if str(c) not in num_cols]
    total = len(df)

    # 目标变量推断：0/1 二值数值列
    target_col = None
    for c in num_cols:
        s = df[c].dropna()
        try:
            uniq = set(s.astype(int).unique())
        except Exception:
            continue
        if len(s) and uniq <= {0, 1} and str(c).lower() not in ("passengerid", "id"):
            target_col = c
            break

    # 1) 数据规模与重复
    dup = int(df.duplicated().sum())
    # 2) 字段明细：类型/缺失/唯一/示例
    field_rows = []
    for c in df.columns:
        nuniq = int(df[c].nunique())
        nmiss = int(df[c].isna().sum())
        is_num = c in num_cols
        dv = df[c].dropna()
        sample = str(dv.iloc[0]) if dv.shape[0] > 0 else ""
        field_rows.append(f"- {c}: 类型={'数值' if is_num else '类别'}, 缺失{nmiss}, 唯一{nuniq}, 示例='{sample}'")
    field_text = "\n".join(field_rows)
    # 3) 描述统计
    desc_text = df[num_cols].describe().round(3).to_string() if num_cols else "（无数值列）"
    # 4) 类别分布
    cat_dist = []
    for c in cat_cols:
        vc = df[c].astype(str).value_counts()
        if len(vc):
            cat_dist.append(f"{c}: 唯一{int(df[c].nunique())} 最常见='{vc.index[0]}'({int(vc.iloc[0])}次/{vc.iloc[0]/total*100:.1f}%)")
    cat_dist_text = "\n".join(cat_dist) or "（无类别列）"
    # 5) 缺失值数量与比例
    miss_text = "；".join([f"{c}:{int(df[c].isna().sum())}条({df[c].isna().mean()*100:.2f}%)" for c in df.columns if df[c].isna().sum() > 0]) or "无"
    # 6) 目标变量分布
    target_dist = ""
    if target_col:
        vc = df[target_col].value_counts().sort_index()
        target_dist = "；".join([f"{k}:{int(v)}人({v/total*100:.2f}%)" for k, v in vc.items()])
    # 7) 单变量偏度
    skew_text = "；".join([f"{c}:{df[c].dropna().skew():.2f}" for c in num_cols]) or "（无数值列）"
    # 8) 双变量：分类特征 vs 目标
    bivar_cat = []
    if target_col:
        for c in cat_cols:
            if df[c].nunique() <= 10:
                g = df.groupby(c)[target_col].agg(["mean", "count"])
                bivar_cat.append(f"{c} → {target_col}生还率: " + "；".join([f"{k}:{row['mean']*100:.1f}%(n={int(row['count'])})" for k, row in g.iterrows()]))
    bivar_cat_text = "\n".join(bivar_cat) or "（无目标变量或类别列）"
    # 9) 数值 vs 目标均值对比
    bivar_num = []
    if target_col:
        for c in num_cols:
            if c == target_col:
                continue
            m0 = df[df[target_col] == 0][c].mean()
            m1 = df[df[target_col] == 1][c].mean()
            bivar_num.append(f"{c}: 目标0均值{m0:.2f} vs 目标1均值{m1:.2f}")
    bivar_num_text = "\n".join(bivar_num) or "（无目标变量）"
    # 10) 相关性矩阵 + 强相关对
    corr_text = "（数值列少于2列，无法计算相关性）"
    strong_pairs = []
    if len(num_cols) >= 2:
        corr = df[num_cols].corr(numeric_only=True)
        corr_text = corr.round(3).to_string()
        for i in range(len(corr)):
            for j in range(i + 1, len(corr)):
                v = corr.iloc[i, j]
                if not pd.isna(v) and abs(v) >= 0.5:
                    strong_pairs.append(f"{corr.columns[i]}×{corr.columns[j]}={v:.3f}")
    strong_text = "，".join(strong_pairs) or "无明显强相关（|r|≥0.5）"
    # 11) 与目标变量的相关系数
    target_corr = ""
    if target_col and len(num_cols) >= 2:
        tcs = []
        for c in num_cols:
            if c == target_col:
                continue
            r = df[c].corr(df[target_col])
            if not pd.isna(r):
                tcs.append(f"{c}:{r:.3f}")
        target_corr = "；".join(tcs)

    report_prompt = f"""你是资深数据分析师与机器学习特征工程专家。基于下面给定数据集的确定性统计结果，生成一份完整的「数据探索性分析报告」（Markdown 格式）。所有数字必须直接使用给定的统计结果，不得编造或自行计算；报告中所有表格与要点组织必须遵循末尾给出的模板章节结构。

数据集确定性统计结果：
- 规模：{total} 行 × {len(df.columns)} 列；重复行：{dup}
- 目标变量（0/1 二值列）：{target_col or "未检测到"}
- 数值列（{len(num_cols)}）：{num_cols}
- 类别列（{len(cat_cols)}）：{cat_cols}
- 字段明细：
{field_text}
- 数值列描述统计：
{desc_text}
- 类别列分布：
{cat_dist_text}
- 缺失值（数量与比例）：{miss_text}
- 目标变量分布：{target_dist or "（无目标变量）"}
- 数值列偏度：{skew_text}
- 分类特征与目标关系：
{bivar_cat_text}
- 数值特征与目标均值对比：
{bivar_num_text}
- 数值列相关性矩阵：
{corr_text}
- 强相关对：{strong_text}
- 与目标变量的相关系数：{target_corr or "（无目标变量）"}

【输出要求】
必须严格按照以下 10 个章节模板输出完整报告。这 10 个二级标题必须原样出现、顺序固定、一个都不能少；不得改写标题、不得合并章节、不得省略任何章节、不得提前结束，必须一直输出到「## 10. 关键结论与建模建议」完成为止。

章节模板：
## 1. 数据集概述
说明数据集来源背景（如不可知则结合字段特点合理描述）、任务类型（二分类/多分类/回归/聚类，根据目标变量或数据特点判断）。

## 2. 字段（指标）含义说明
用表格列出每个字段：| 字段 | 类型 | 含义 |，含义需结合取值特点说明（如"唯一标识，无预测意义"、"目标变量"、"类别编码"等）。

## 3. 数据规模与结构
- 行数×列数、重复行数
- 数值型/类别型字段清单
- 数值型字段描述统计表（count/mean/std/min/25%/50%/75%/max，使用给定 describe 结果）
- 类别型字段常见值分布（最常见值及频次占比）
- 关键发现（均值/极值/分布特征）

## 4. 数据质量检查
- 缺失值统计表：| 字段 | 缺失数量 | 缺失比例 |
- 说明缺失最严重的字段及处理方向（丢弃/填补策略）

## 5. 目标变量分布（若有 0/1 目标变量）
- 分布统计表 | 类别 | 人数 | 占比 |
- 类别是否平衡，对建模评估指标选择的影响

## 6. 单变量分布分析
- 数值型特征偏度表 | 特征 | 偏度 | 分布特征 |（结合偏度值解读右偏/左偏/近似正态）
- 指出需 log 变换或分箱的特征

## 7. 双变量分析：特征与目标关系
- 各分类特征按目标变量的生还率/均值表（使用给定数据）
- 数值特征在目标 0/1 下的均值对比
- 提炼最强预测因子与结论

## 8. 数值特征相关性分析
- 给出相关性矩阵要点 + 强相关对（|r|≥0.5）
- 与目标变量的相关系数排序
- 指出共线性风险

## 9. 衍生特征探索
- 基于字段特点提出可构造的衍生特征（组合特征、比率、分箱、编码等），说明理由

## 10. 关键结论与建模建议
- 核心发现（列要点）
- 缺失值处理建议表：| 字段 | 缺失比例 | 处理方案 |
- 特征工程建议表：| 原始特征 | 工程操作 | 理由 |
- 建模推荐（算法、评估指标、验证策略）"""

    res = call_ollama(report_prompt, model_name=model_name, stream=False, timeout=300, num_predict=4096)
    if "error" in res:
        return {"code": -1, "msg": res["error"]}
    report = res.get("response", "")
    if not report.strip():
        return {"code": -1, "msg": "模型未返回分析报告，请重试"}

    # 章节完整性校验：缺失或标题不符的章节由模型二次补写，并按章节号插入正确位置
    import re as _re

    def _sec_num(sec):
        m = _re.search(r"##\s*(\d+)", sec)
        return int(m.group(1)) if m else 99

    required = ["## 1. 数据集概述", "## 2. 字段", "## 3. 数据规模", "## 4. 数据质量",
                "## 5. 目标变量", "## 6. 单变量", "## 7. 双变量", "## 8. 数值特征相关性",
                "## 9. 衍生特征", "## 10. 关键结论"]
    missing = [k for k in required if k not in report]
    if missing:
        fix_prompt = f"""你正在完善一份「数据探索性分析报告」（Markdown）。当前报告缺少以下章节，请补写这些章节。

需要补写的章节（二级标题必须与下面完全一致）：
{chr(10).join(missing)}

补写要求：
1. 只输出这些章节对应的 Markdown 内容，每个章节以二级标题开头；
2. 数字与统计口径须与已有报告保持一致，不得编造不存在的字段；
3. 遵循数据探索性分析报告的规范，内容详实、可直接用于机器学习建模。

已生成的报告（仅用于衔接与保持口径一致，不要重复输出其已有内容）：
{report}"""
        res2 = call_ollama(fix_prompt, model_name=model_name, stream=False, timeout=300, num_predict=2048)
        if "error" not in res2 and res2.get("response", "").strip():
            extra = res2["response"].strip()
            secs = sorted([s for s in _re.split(r"(?=##\s*\d)", extra) if s.strip()], key=_sec_num)
            parts = _re.split(r"(?=##\s*\d)", report)
            heads = [_sec_num(p) for p in parts]
            for sec in secs:
                n = _sec_num(sec)
                idx = next((i for i, h in enumerate(heads) if h > n), None)
                if idx is None:
                    parts.append(sec.strip())
                else:
                    parts.insert(idx, sec.strip())
            report = "".join(parts)

    # 结构化诊断 + 处方（确定性计算，不依赖大模型；供前端"诊断→处方→一键应用"闭环）
    from routes.preprocess import diagnose_column
    diag_rows = []
    for c in num_cols:
        d = diagnose_column(df[c], c)
        nmiss = int(df[c].isna().sum())
        miss_ratio = round(nmiss / total * 100, 2) if total else 0.0
        if d["skew"] > 1:
            level = "严重右偏"
        elif d["skew"] < -1:
            level = "严重左偏"
        else:
            level = "近似正态"
        sug = "无需处理"
        if d["skew"] > 1 and d["positive"]:
            sug = "对数变换(log1p)"
        elif d["skew"] > 1:
            sug = "分箱处理"
        elif d["skew"] < -1 and d["positive"]:
            sug = "Box-Cox 变换"
        elif d["skew"] < -1:
            sug = "分箱处理"
        if nmiss > 0:
            sug = ("填充缺失(中位数) → " + sug) if sug != "无需处理" else "填充缺失(中位数)"
        diag_rows.append({
            "column": c, "skew": d["skew"], "skew_level": level,
            "outlier_ratio": d["outlier_ratio"], "missing": nmiss, "missing_ratio": miss_ratio,
            "suggestion": sug,
        })

    quality_issues = []
    if dup > 0:
        quality_issues.append({"type": "duplicate", "detail": f"重复行 {dup} 行（{dup / total * 100:.2f}%）", "action": "删除重复行"})
    for r in diag_rows:
        if r["missing_ratio"] >= 30:
            quality_issues.append({"type": "missing", "detail": f"{r['column']} 缺失 {r['missing_ratio']}%", "action": "考虑删除该列或按业务填充"})
    bad = [r for r in diag_rows if r["skew_level"] != "近似正态"]
    if bad:
        quality_issues.append({"type": "skew", "detail": f"{len(bad)} 个数值列偏态严重：" + "、".join(r["column"] for r in bad), "action": "执行对数/Box-Cox/分箱变换"})
    high_out = [r for r in diag_rows if r["outlier_ratio"] > 0.1]
    if high_out:
        quality_issues.append({"type": "outlier", "detail": f"{len(high_out)} 列存在异常群组：" + "、".join(r["column"] for r in high_out), "action": "变换后复查离群占比"})

    clean_advice = [it["action"] for it in quality_issues]
    preprocess_advice = [{"column": r["column"], "action": r["suggestion"]} for r in diag_rows if r["suggestion"] != "无需处理"]
    diagnosis = {
        "columns": diag_rows,
        "quality_issues": quality_issues,
        "verdict": "数据整体质量良好，可直接建模" if not quality_issues else f"发现 {len(quality_issues)} 项数据质量/分布问题，建议按下方处方处理后再建模",
        "clean_advice": clean_advice,
        "preprocess_advice": preprocess_advice,
    }

    return {"code": 0, "msg": "ok", "data": {"report": report, "model_name": model_name, "diagnosis": diagnosis}}

@app.get("/download_plot_zip")
async def download_plot_zip(files: str = Query(default="", description="逗号分隔的 PNG 文件名，须为后端已生成的绘图文件")):
    """把按需绘制的图打包成 zip 下载（仅放行 eda_export 目录内已生成的 PNG）。"""
    names = [n.strip() for n in files.split(",") if n.strip()]
    if not names:
        return {"code": -1, "msg": "没有可打包的图像"}
    abs_export = os.path.abspath(eda_export_folder)
    zip_name = f"eda_plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    zip_path = os.path.abspath(os.path.join(eda_export_folder, zip_name))
    try:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            written = 0
            for n in names:
                abs_p = os.path.abspath(os.path.join(eda_export_folder, n))
                if not abs_p.startswith(abs_export) or not os.path.exists(abs_p):
                    continue
                zf.write(abs_p, n)
                written += 1
            if written == 0:
                return {"code": -1, "msg": "未找到可打包的图像文件"}
    except Exception as e:
        return {"code": -1, "msg": f"打包失败: {str(e)}"}
    return FileResponse(zip_path, filename=zip_name, media_type="application/zip")

# ============ 数据概览与探索：数据质量报告（确定性 pandas 计算，不调用大模型） ============
@app.post("/data_overview")
async def data_overview(file_path: str = Query()):
    """一键数据质量报告：数据形状/列名与类型、数值/文本统计摘要、缺失与重复。"""
    try:
        df = load_df_checked(file_path)
    except Exception as e:
        return {"code": -1, "msg": str(e)}
    total = int(len(df))
    if total == 0:
        return {"code": -1, "msg": "数据集为空，无法生成数据质量报告"}

    # 1. 概览：列名 + 归一化数据类型
    cols = []
    for c in df.columns:
        cname = str(c)
        dt = df[c].dtype
        if pd.api.types.is_datetime64_any_dtype(dt):
            type_name = "datetime"
        elif pd.api.types.is_numeric_dtype(dt):
            type_name = "int" if pd.api.types.is_integer_dtype(dt) else "float"
        else:
            type_name = "object"
        cols.append({"name": cname, "type": type_name, "dtype": str(dt)})

    # 2. 数值列统计摘要：均值/标准差/分位数
    num_summary = []
    for c in df.select_dtypes(include="number").columns:
        s = pd.to_numeric(df[c], errors="coerce")
        has = bool(s.notna().any())
        num_summary.append({
            "column": str(c),
            "count": int(s.notna().sum()),
            "mean": round(float(s.mean()), 4) if has else None,
            "std": round(float(s.std()), 4) if s.count() > 1 else None,
            "min": round(float(s.min()), 4) if has else None,
            "q25": round(float(s.quantile(0.25)), 4) if has else None,
            "q50": round(float(s.quantile(0.5)), 4) if has else None,
            "q75": round(float(s.quantile(0.75)), 4) if has else None,
            "max": round(float(s.max()), 4) if has else None,
        })

    # 3. 文本列统计摘要：唯一值数量 + 众数
    text_summary = []
    for c in df.select_dtypes(exclude="number").columns:
        cname = str(c)
        s_obj = df[c].astype(str)
        mode_ser = s_obj.mode()
        mode = str(mode_ser.iloc[0]) if len(mode_ser) else ""
        text_summary.append({
            "column": cname,
            "nunique": int(df[c].nunique(dropna=True)),
            "mode": mode,
            "mode_count": int((s_obj == mode).sum()),
        })

    # 4. 缺失统计：每列缺失数 + 比例
    missing = []
    for c in df.columns:
        n = int(df[c].isna().sum())
        missing.append({"column": str(c), "missing": n,
                        "ratio": round(n / total * 100, 2) if total else 0.0})

    # 5. 整行重复记录
    dup = int(df.duplicated().sum())

    return {
        "code": 0,
        "msg": "数据质量报告生成完成",
        "data": {
            "shape": {"rows": total, "cols": int(df.shape[1])},
            "columns": cols,
            "numeric_summary": num_summary,
            "text_summary": text_summary,
            "missing": missing,
            "duplicates": {"count": dup, "ratio": round(dup / total * 100, 2) if total else 0.0},
        },
    }
