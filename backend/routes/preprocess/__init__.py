# -*- coding: utf-8 -*-
"""智能预处理流水线（按附件流程图实现）：

    [原始数据]
        ↓
    [初步EDA] → (发现: 数据偏态、存在异常群组)
        ↓
    [预处理A] → (执行: 对数变换、标准化)
        ↓
    [再次EDA] → (验证: 变换后是否符合正态假设？异常群组是否被拉近？)
        ↓ 如果效果不好
    [预处理B] → (改用: Box-Cox变换或分箱处理)
        ↓ 直到通过验证
    [最终建模数据集]

接入点：模型训练前的数值特征准备（train._prepare_xy）与独立接口 /preprocess_run。
标准化步骤由训练侧 StandardScaler 承担（本流水线负责"变换尝试 + 迭代验证"）。
"""
import numpy as np
import pandas as pd
from scipy import stats
from fastapi import Query

from config import app
from routes.clean import _validate_and_load

# 验证阈值：|偏度| <= 1 视为通过（正态假设近似成立），离群占比不高于基线
SKEW_OK = 1.0
MIN_SAMPLES = 8


def diagnose_column(series, name):
    """初步 EDA：单列偏度、IQR 离群占比、取值范围、是否全正。"""
    s = pd.to_numeric(series, errors="coerce").dropna()
    info = {"name": name, "n": int(len(s))}
    if len(s) < MIN_SAMPLES:
        info.update({"skew": 0.0, "outlier_ratio": 0.0, "positive": True, "min": 0.0, "max": 0.0})
        return info
    skew = float(s.skew()) if s.nunique() > 1 else 0.0
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        out_ratio = 0.0
    else:
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        out_ratio = float(((s < lo) | (s > hi)).mean())
    info.update(
        {
            "skew": round(float(skew), 4),
            "outlier_ratio": round(float(out_ratio), 4),
            "positive": bool((s > 0).all()),
            "min": float(s.min()),
            "max": float(s.max()),
        }
    )
    return info


def validate_transform(transformed, before):
    """再次 EDA：变换后 |skew|<=1 且离群占比不恶化；附 Shapiro-Wilk p 值供参考。"""
    s = pd.to_numeric(transformed, errors="coerce").dropna()
    if len(s) < MIN_SAMPLES:
        return True, {"skew_after": 0.0, "outlier_after": 0.0, "shapiro_p": None}
    skew = float(s.skew()) if s.nunique() > 1 else 0.0
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        out = 0.0
    else:
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        out = float(((s < lo) | (s > hi)).mean())
    try:
        shapiro_p = float(stats.shapiro(s)[1])
        if not np.isfinite(shapiro_p):
            shapiro_p = None
    except Exception:
        shapiro_p = None
    ok = abs(skew) <= SKEW_OK and out <= max(before["outlier_ratio"], 0.05)
    return ok, {"skew_after": round(float(skew), 4), "outlier_after": round(float(out), 4), "shapiro_p": shapiro_p}


def apply_transform(series, method, params):
    """按训练时保存的变换参数对序列做同款正变换（训练/预测对齐用）。"""
    s = pd.to_numeric(series, errors="coerce")
    if method == "log1p":
        return np.log1p(s.clip(lower=0.0))
    if method == "boxcox":
        lam = float(params.get("lambda", 0.0))
        s = s.clip(lower=1e-9)
        if lam == 0:
            return np.log(s)
        return (np.power(s, lam) - 1.0) / lam
    if method == "qcut":
        bins = params.get("bins")
        if bins:
            # 边界外值裁剪到首/末箱边界后分箱，避免浮点精度导致 NaN
            bb = [float(b) for b in bins]
            s = s.clip(lower=bb[0], upper=bb[-1])
            t = pd.cut(s, bins=bb, labels=False).astype(float)
            return t.fillna(0.0)
        return s
    return s


def smart_preprocess(df, num_cols, logs=None):
    """按附件流程对数值列迭代处理，返回 (df_out, transforms, steps)。

    transforms: {列名: {"method": "none|log1p|boxcox|qcut",
                         "lambda": float|None, "bins": [...]|None,
                         "skew_before": .., "skew_after": .., "shapiro_p": ..}}
    """
    df = df.copy()
    transforms = {}
    steps = []
    for col in num_cols:
        before = diagnose_column(df[col], col)
        steps.append(f"[初步EDA] {col}：偏度={before['skew']}，离群占比={before['outlier_ratio']}，"
                     f"范围[{before['min']},{before['max']}]")
        if abs(before["skew"]) <= SKEW_OK or before["n"] < MIN_SAMPLES:
            transforms[col] = {"method": "none", "lambda": None, "bins": None,
                               "skew_before": before["skew"], "skew_after": before["skew"],
                               "shapiro_p": None}
            steps.append(f"[验证] {col}：偏度已达标，保持原值 ✓")
            continue

        # ---- 预处理A：对数变换（正偏且全正） ----
        applied = False
        if before["skew"] > 0 and before["positive"]:
            t = np.log1p(pd.to_numeric(df[col], errors="coerce").clip(lower=0.0))
            ok, v = validate_transform(t, before)
            if ok:
                transforms[col] = {"method": "log1p", "lambda": None, "bins": None,
                                   "skew_before": before["skew"], **v, "passed": True}
                df[col] = t
                applied = True
                steps.append(f"[预处理A] {col}：对数变换(log1p) → 偏度={v['skew_after']} ✓")
            else:
                steps.append(f"[预处理A] {col}：log1p 后偏度={v['skew_after']} 仍未达标，改用预处理B")

        # ---- 预处理B：Box-Cox（全正）或分箱 ----
        if not applied:
            s = pd.to_numeric(df[col], errors="coerce")
            if before["positive"]:
                try:
                    sc = s.dropna()
                    if sc.min() > 0 and sc.nunique() > 3:
                        _, lam = stats.boxcox(sc)
                        lam = round(float(lam), 6)  # 取整后保存，保证训练/预测完全一致
                        t = pd.Series(apply_transform(s, "boxcox", {"lambda": lam}), index=df.index)
                        ok, v = validate_transform(t, before)
                        if ok:
                            transforms[col] = {"method": "boxcox", "lambda": lam, "bins": None,
                                               "skew_before": before["skew"], **v, "passed": True}
                            df[col] = t
                            applied = True
                            steps.append(f"[预处理B] {col}：Box-Cox(λ={lam:.4f}) → 偏度={v['skew_after']} ✓")
                except Exception as e:
                    steps.append(f"[预处理B] {col}：Box-Cox 不可用（{e}），尝试分箱")
            if not applied:
                try:
                    s2 = s.fillna(s.median())
                    # 精确分位边界（不依赖 qcut 的圆整 categories），首末箱 ±eps 保证全覆盖
                    qs = sorted(set(s2.quantile([0.25, 0.5, 0.75]).tolist()))
                    if not qs or len(qs) > 3:
                        raise ValueError("分位数边界异常")
                    edges = [float(s2.min()) - 1e-9] + [float(x) for x in qs] + [float(s2.max()) + 1e-9]
                    t = pd.Series(pd.cut(s2, bins=edges, labels=False).astype(float), index=df.index)
                    ok, v = validate_transform(t, before)
                    transforms[col] = {"method": "qcut", "lambda": None, "bins": edges,
                                       "skew_before": before["skew"], **v, "passed": bool(ok)}
                    df[col] = t
                    applied = True
                    n_bins = len(edges) - 1
                    if ok:
                        steps.append(f"[预处理B] {col}：分箱({n_bins}箱) → 偏度={v['skew_after']} ✓")
                    else:
                        steps.append(f"[预处理B] {col}：分箱({n_bins}箱) → 偏度={v['skew_after']}（未完全达标，作为兜底保留）")
                except Exception as e:
                    transforms[col] = {"method": "none", "lambda": None, "bins": None,
                                       "skew_before": before["skew"], "skew_after": before["skew"],
                                       "shapiro_p": None}
                    steps.append(f"[预处理B] {col}：分箱失败（{e}），保持原值")
    return df, transforms, steps


def transforms_to_text(transforms):
    """把每列的变换方案转成中文描述（用于训练日志/报告）。"""
    name_map = {"none": "保持原值", "log1p": "对数变换 log1p", "boxcox": "Box-Cox 变换",
                "qcut": "分位数分箱"}
    parts = []
    for col, tr in (transforms or {}).items():
        method = tr.get("method", "none")
        if method == "none":
            continue
        text = f"{col}: {name_map.get(method, method)}"
        if method == "boxcox" and tr.get("lambda") is not None:
            text += f"(λ={tr['lambda']})"
        if tr.get("skew_before") is not None and tr.get("skew_after") is not None:
            text += f" [偏度 {tr['skew_before']}→{tr['skew_after']}]"
        parts.append(text)
    return "；".join(parts)


@app.post("/preprocess_run")
async def preprocess_run(file_path: str = Query(), target_column: str = Query(default="")):
    """独立调用智能预处理：初步EDA → 预处理A → 再次EDA → (预处理B) → 最终建模数据集。"""
    try:
        df = _validate_and_load(file_path)
    except Exception as e:
        return {"code": -1, "msg": str(e)}

    if target_column:
        df = df.drop(columns=[target_column]) if target_column in df.columns else df
    num_cols = [str(c) for c in df.select_dtypes(include="number").columns]
    if not num_cols:
        return {"code": -1, "msg": "数据集中没有数值列，无法执行智能预处理"}

    logs = []
    logs.append(f"[1/4] 读取数据集：{len(df)} 行 × {len(df.columns)} 列，数值列 {len(num_cols)} 个")
    # 数值列缺失填充（中位数，与训练侧 _prepare_xy 口径一致），保证变换后无 NaN
    for c in num_cols:
        n0 = int(df[c].isna().sum())
        if n0:
            df[c] = df[c].fillna(df[c].median())
            logs.append(f"[填充] {c}：中位数填充 {n0} 个缺失值")
    df_out, transforms, steps = smart_preprocess(df, num_cols, logs)
    logs.extend(steps)

    changed = [c for c, t in transforms.items() if t.get("method") != "none"]
    logs.append(f"[2/4] 预处理完成：{len(changed)} 个数值列被变换（{transforms_to_text(transforms) or '无'}）")
    logs.append("[3/4] 再次EDA验证完成，输出最终建模数据集")

    # 保存最终建模数据集（cleaned 目录，文件名带 _preprocessed）
    import os
    from routes.clean import _save_cleaned
    out_path, out_name = None, None
    try:
        out_path, out_name = _save_cleaned(df_out, file_path)
        out_name = out_name.replace("_clean", "_preprocessed")
        out_path = os.path.join(os.path.dirname(out_path), out_name)
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".csv":
            df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
        else:
            df_out.to_excel(out_path, index=False)
        logs.append(f"[4/4] 最终建模数据集已保存：{out_name}")
    except Exception as e:
        logs.append(f"[4/4] 保存最终建模数据集失败：{e}")

    # 返回前清洗：任何非有限浮点数一律转 None，保证 JSON 可序列化
    def _json_safe(obj):
        if isinstance(obj, dict):
            return {k: _json_safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_json_safe(v) for v in obj]
        if isinstance(obj, tuple):
            return [_json_safe(v) for v in obj]
        if isinstance(obj, float) and not np.isfinite(obj):
            return None
        if isinstance(obj, np.floating):
            f = float(obj)
            return None if not np.isfinite(f) else f
        if isinstance(obj, np.integer):
            return int(obj)
        return obj

    return {
        "code": 0,
        "msg": "智能预处理完成",
        "data": _json_safe({
            "steps": steps,
            "logs": logs,
            "transforms": transforms,
            "summary": transforms_to_text(transforms),
            "n_rows": int(len(df_out)),
            "n_cols": int(len(df_out.columns)),
            "cleaned_file_path": out_path,
            "cleaned_filename": out_name,
            "preview": [{k: (None if isinstance(v, float) and (np.isnan(v) or np.isinf(v)) else v)
                         for k, v in row.items()}
                        for row in df_out.head(8).to_dict(orient="records")],
        }),
    }
