# -*- coding: utf-8 -*-
"""数据清洗模块：大模型意图解析 + 白名单 pandas 确定性执行 + 规则兜底，
以及清洗建议问答、清洗结果下载。"""
import os
import re
import json
from typing import List, Dict

import pandas as pd
from fastapi import Query
from fastapi.responses import FileResponse
from datetime import datetime

from config import app, dataset_folder, cleaned_folder, DEFAULT_MODEL
from utils import call_ollama, extract_json

# 清洗操作白名单（模型只能从这里选，后端确定性执行，杜绝自由代码风险）
CLEAN_OPS = (
    "drop_duplicates", "drop_missing", "fill_missing", "strip",
    "to_numeric", "to_datetime", "rename", "replace",
    "drop_columns", "keep_columns", "col_math",
)

def _check_col(df: pd.DataFrame, col):
    if col not in df.columns:
        raise ValueError(f"列不存在: {col}")

def apply_clean_op(df: pd.DataFrame, op: str, params: Dict) -> pd.DataFrame:
    """按白名单操作确定性执行，返回新 df。参数非法时抛 ValueError。"""
    df = df.copy()
    if op == "drop_duplicates":
        subset = params.get("subset") or None
        if subset:
            for c in subset:
                _check_col(df, c)
        df = df.drop_duplicates(subset=subset, keep="first")
    elif op == "drop_missing":
        how = params.get("how", "any")
        subset = params.get("subset") or None
        if subset:
            for c in subset:
                _check_col(df, c)
        df = df.dropna(how=how, subset=subset)
    elif op == "fill_missing":
        method = params.get("method", "mean")
        columns = params.get("columns")
        if columns:
            for c in columns:
                _check_col(df, c)
            cols = columns
        else:
            cols = list(df.select_dtypes(include="number").columns)
        if method == "mean":
            for c in cols:
                if pd.api.types.is_numeric_dtype(df[c]):  # 均值/中位数仅对数值列有效，避免 string.mean() 报错
                    df[c] = df[c].fillna(df[c].mean())
        elif method == "median":
            for c in cols:
                if pd.api.types.is_numeric_dtype(df[c]):
                    df[c] = df[c].fillna(df[c].median())
        elif method == "mode":
            for c in cols:
                m = df[c].mode()
                if len(m):
                    df[c] = df[c].fillna(m[0])
        elif method == "zero":
            for c in cols:
                df[c] = df[c].fillna(0)
        elif method == "ffill":
            df[cols] = df[cols].ffill()
        elif method == "bfill":
            df[cols] = df[cols].bfill()
        elif method == "constant":
            value = params.get("value")
            for c in cols:
                df[c] = df[c].fillna(value)
        else:
            raise ValueError(f"不支持的填充方法: {method}")
    elif op == "strip":
        columns = params.get("columns")
        if columns:
            for c in columns:
                _check_col(df, c)
            cols = columns
        else:
            cols = [c for c in df.columns if df[c].dtype == object]
        for c in cols:
            if df[c].dtype == object:  # 仅对字符串列去除首尾空格，数值列跳过
                df[c] = df[c].astype(object).str.strip()
    elif op == "to_numeric":
        for c in (params.get("columns") or []):
            _check_col(df, c)
            df[c] = pd.to_numeric(df[c], errors="coerce")
    elif op == "to_datetime":
        for c in (params.get("columns") or []):
            _check_col(df, c)
            df[c] = pd.to_datetime(df[c], errors="coerce")
    elif op == "rename":
        mapping = params.get("mapping") or {}
        df = df.rename(columns=mapping)
    elif op == "replace":
        column = params.get("column")
        _check_col(df, column)
        df[column] = df[column].replace(params.get("old"), params.get("new"))
    elif op == "drop_columns":
        for c in (params.get("columns") or []):
            _check_col(df, c)
        df = df.drop(columns=params.get("columns") or [])
    elif op == "keep_columns":
        for c in (params.get("columns") or []):
            _check_col(df, c)
        df = df[params.get("columns") or []]
    elif op == "col_math":
        col1 = params.get("col1")
        col2 = params.get("col2")
        operator = params.get("operator")
        new_name = (params.get("new_name") or "").strip()
        if not col1 or not col2:
            raise ValueError("数值运算：请选择两个列")
        _check_col(df, col1)
        _check_col(df, col2)
        if col1 == col2:
            raise ValueError("数值运算：两个列不能相同")
        if operator not in ("+", "-", "*", "/"):
            raise ValueError(f"数值运算：不支持的运算符: {operator}")
        if not new_name:
            raise ValueError("数值运算：新列名不能为空")
        if new_name in df.columns:
            raise ValueError(f"数值运算：新列名已存在: {new_name}")
        s1 = pd.to_numeric(df[col1], errors="coerce")
        s2 = pd.to_numeric(df[col2], errors="coerce")
        if operator == "+":
            res = s1 + s2
        elif operator == "-":
            res = s1 - s2
        elif operator == "*":
            res = s1 * s2
        else:
            res = s1 / s2
        df[new_name] = res  # 追加到最后一列
    else:
        raise ValueError(f"未知操作: {op}")
    return df

def _writeback_math_to_source(file_path: str, ops) -> None:
    """数值运算：把计算结果列直接写回原数据集文件（仅追加新列，不改动原列）。
    支持平铺与嵌套 params 两种操作格式；新列已存在时跳过，避免重复覆盖。"""
    items = ops or []
    math_ops = [it for it in items if isinstance(it, dict) and it.get("op") == "col_math"]
    if not math_ops:
        return
    raw = _validate_and_load(file_path)
    changed = False
    for it in math_ops:
        p = it.get("params") if isinstance(it.get("params"), dict) else {k: v for k, v in it.items() if k != "op"}
        col1 = p.get("col1")
        col2 = p.get("col2")
        operator = p.get("operator")
        new_name = (p.get("new_name") or "").strip()
        if not col1 or not col2:
            raise ValueError("数值运算：请选择两个列，无法写回原数据集")
        if col1 not in raw.columns or col2 not in raw.columns:
            raise ValueError(f"列不存在于原数据集: {col1} / {col2}")
        if operator not in ("+", "-", "*", "/"):
            raise ValueError(f"数值运算：不支持的运算符: {operator}")
        if not new_name:
            raise ValueError("数值运算：新列名不能为空，无法写回原数据集")
        if new_name in raw.columns:
            continue  # 已存在跳过
        s1 = pd.to_numeric(raw[col1], errors="coerce")
        s2 = pd.to_numeric(raw[col2], errors="coerce")
        if operator == "+":
            res = s1 + s2
        elif operator == "-":
            res = s1 - s2
        elif operator == "*":
            res = s1 * s2
        else:
            res = s1 / s2
        raw[new_name] = res
        changed = True
    if changed:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".csv":
            raw.to_csv(file_path, index=False, encoding="utf-8-sig")
        else:
            raw.to_excel(file_path, index=False)

# 操作中文名（用于纯 pandas 清洗的确定性总结，不依赖模型）
OP_CN = {
    "drop_duplicates": "删除重复行",
    "drop_missing": "删除缺失行",
    "fill_missing": "填充缺失值",
    "strip": "去除首尾空格",
    "to_numeric": "转为数值",
    "to_datetime": "转为日期",
    "rename": "重命名列",
    "replace": "替换值",
    "drop_columns": "删除列",
    "keep_columns": "保留列",
    "col_math": "数值运算",
}

def parse_clean_query(query: str, df: pd.DataFrame):
    """用纯规则把用户自然语言清洗指令解析为白名单操作列表（不依赖大模型）。
    返回 (ops, text_ops)，ops 供 apply_clean_op 确定性执行，text_ops 用于生成总结。
    无法识别的指令抛出 ValueError（附可用指令示例）。"""
    q = str(query).strip()
    if not q:
        raise ValueError("清洗指令不能为空，请输入如：删除重复行 / 用均值填充 Age 缺失")
    ql = q.lower()

    col_names = [str(c) for c in df.columns]

    def find_cols() -> List[str]:
        """提取指令中提到的真实列名（按长度降序匹配，避免子串误匹配）。"""
        found = []
        for c in sorted(col_names, key=len, reverse=True):
            if c and c.lower() in ql and c not in found:
                found.append(c)
        return found

    ops: List[Dict] = []
    text_ops: List[str] = []

    # 1) 删除重复行
    if any(k in ql for k in ("重复", "去重", "duplicate")):
        ops.append({"op": "drop_duplicates"})
        text_ops.append("删除重复行")

    # 2) 删除缺失行（先于"填充"判定，避免"删除缺失行"被误归为填充）
    if any(k in ql for k in (
        "删除缺失行", "删除缺失值行", "去掉缺失行", "删除含缺失", "删除有缺失",
        "删除空行", "删掉有缺失", "dropna", "drop missing",
    )):
        ops.append({"op": "drop_missing"})
        text_ops.append("删除缺失行")

    # 3) 填充缺失
    if any(k in ql for k in ("填充", "填补", "补全", "fill")):
        method = "mean"
        value = None
        if any(k in ql for k in ("中位", "median")):
            method = "median"
        elif any(k in ql for k in ("众数", "mode")):
            method = "mode"
        elif any(k in ql for k in ("前向", "ffill")):
            method = "ffill"
        elif any(k in ql for k in ("后向", "bfill")):
            method = "bfill"
        elif any(k in ql for k in ("常量", "指定值", "constant")):
            m = re.search(r"(?:常量|指定值|constant)\D*?(\d+(?:\.\d+)?)", ql)
            method = "constant"
            value = float(m.group(1)) if m else 0.0
        elif any(k in ql for k in ("零", "0")) and any(k in ql for k in ("填", "补")):
            method = "zero"
        op = {"op": "fill_missing", "method": method}
        if method == "constant":
            op["value"] = value
        c_found = find_cols()
        if c_found:
            op["columns"] = c_found
        ops.append(op)
        method_cn = {
            "mean": "均值", "median": "中位数", "mode": "众数",
            "zero": "0", "ffill": "前向填充", "bfill": "后向填充", "constant": "常量",
        }.get(method, method)
        text_ops.append(
            f"用{method_cn}填充缺失值" + (f"（列：{'、'.join(c_found)}）" if c_found else "")
        )

    # 4) 去除首尾空格
    if any(k in ql for k in ("空格", "strip", "去空白")):
        op = {"op": "strip"}
        c_found = find_cols()
        if c_found:
            op["columns"] = c_found
        ops.append(op)
        text_ops.append("去除首尾空格")

    # 5) 转数值
    if any(k in ql for k in ("转数值", "转数字", "转成数值", "to_numeric")):
        c_found = find_cols()
        if not c_found:
            raise ValueError("转数值需要指定列，如：把 Age 转数值")
        ops.append({"op": "to_numeric", "columns": c_found})
        text_ops.append(f"将{'、'.join(c_found)}转为数值")

    # 6) 转日期
    if any(k in ql for k in ("转日期", "转时间", "转成日期", "to_datetime")):
        c_found = find_cols()
        if not c_found:
            raise ValueError("转日期需要指定列，如：把 Date 转日期")
        ops.append({"op": "to_datetime", "columns": c_found})
        text_ops.append(f"将{'、'.join(c_found)}转为日期")

    # 7) 重命名列：把 A 改成 B / 重命名 A 为 B
    m = re.search(
        r"(?:把|将)?\s*([^\s，。、]+)\s*(?:重命名|改名|改成|改为|更名为)\s*(?:为|成)?\s*([^\s，。]+)", q
    )
    if m:
        old, new = m.group(1), m.group(2)
        if old in col_names:
            ops.append({"op": "rename", "mapping": {old: new}})
            text_ops.append(f"将列“{old}”重命名为“{new}”")

    # 8) 替换值：把 A 列中的 X 替换成 Y / 将 X 替换为 Y
    m = re.search(
        r"(?:把|将)?\s*([^\s，。、]+)列?[中里]?\s*([^\s，。]+?)\s*(?:替换成|替换为|换成|replace)\s*([^\s，。]+)", q
    )
    if m:
        col, old, new = m.group(1), m.group(2), m.group(3)
        if col in col_names:
            ops.append({"op": "replace", "column": col, "old": old, "new": new})
            text_ops.append(f"将列“{col}”中的“{old}”替换为“{new}”")

    # 9) 删除列 / 保留列
    if any(k in ql for k in ("删除列", "去掉列", "移除列", "删除字段")):
        c_found = find_cols()
        if c_found:
            ops.append({"op": "drop_columns", "columns": c_found})
            text_ops.append(f"删除列：{'、'.join(c_found)}")
    if any(k in ql for k in ("保留列", "只保留")):
        c_found = find_cols()
        if c_found:
            ops.append({"op": "keep_columns", "columns": c_found})
            text_ops.append(f"只保留列：{'、'.join(c_found)}")

    if not ops:
        raise ValueError(
            "无法识别清洗指令。支持示例：\n"
            "· 删除重复行\n"
            "· 用均值/中位数填充缺失值（可指定列，如：用均值填充 Age 缺失）\n"
            "· 删除缺失行\n"
            "· 去除首尾空格\n"
            "· 把 Age 转数值 / 把 Date 转日期\n"
            "· 把 A 重命名为 B / 删除列 X / 只保留列 X"
        )
    return ops, text_ops

def ops_to_text(ops: List[Dict]) -> str:
    """把清洗操作计划转成中文描述（用于确定性总结，不依赖模型）。"""
    parts = []
    for item in ops or []:
        if not isinstance(item, dict):
            continue
        op = item.get("op", "")
        params = {k: v for k, v in item.items() if k != "op"}
        name = OP_CN.get(op, str(op))
        if op == "fill_missing":
            cols = "、".join(params.get("columns") or ["数值列"])
            method_cn = {
                "mean": "均值", "median": "中位数", "mode": "众数", "zero": "0",
                "ffill": "前向填充", "bfill": "后向填充", "constant": "常量",
            }.get(params.get("method", "mean"), params.get("method", "mean"))
            parts.append(f"用{method_cn}填充缺失值（{cols}）")
        elif op in ("strip", "to_numeric", "to_datetime", "drop_columns", "keep_columns"):
            cols = params.get("columns")
            if cols:
                parts.append(f"{name}（{'、'.join(cols)}）")
            else:
                parts.append(name)
        elif op == "rename":
            mapping = params.get("mapping") or {}
            parts.append("重命名列（" + "、".join(f"{k}→{v}" for k, v in mapping.items()) + "）")
        elif op == "replace":
            parts.append(f"替换值（{params.get('column')}: {params.get('old')}→{params.get('new')}）")
        else:
            parts.append(name)
    return "；".join(parts)

def compute_quality_report(df: pd.DataFrame) -> Dict:
    """计算数据质量指标，用于清洗前后对比"""
    return {
        "rows": int(len(df)),
        "cols": int(len(df.columns)),
        "missing_total": int(df.isna().sum().sum()),
        "missing_by_col": {str(k): int(v) for k, v in df.isna().sum().items()},
        "duplicated_rows": int(df.duplicated().sum()),
        "dtypes": {str(k): str(v) for k, v in df.dtypes.items()},
    }

def quality_report_to_text(report: Dict) -> str:
    lines = [
        f"- 行数: {report['rows']}, 列数: {report['cols']}",
        f"- 缺失值总数: {report['missing_total']}",
        f"- 重复行数: {report['duplicated_rows']}",
    ]
    missing_parts = [f"{k}={v}" for k, v in report["missing_by_col"].items() if v > 0]
    lines.append(f"- 各列缺失值: {', '.join(missing_parts) if missing_parts else '无'}")
    return "\n".join(lines)

def _validate_and_load(file_path: str):
    """校验路径安全并读取数据集（含 pandas 3.x string 兼容）。返回 DataFrame。"""
    if not os.path.exists(file_path):
        raise FileNotFoundError("数据集文件不存在，请先上传数据集")
    abs_upload = os.path.abspath(dataset_folder)
    abs_cleaned = os.path.abspath(cleaned_folder)
    abs_file = os.path.abspath(file_path)
    if not (abs_file.startswith(abs_upload) or abs_file.startswith(abs_cleaned)):
        raise PermissionError("禁止访问upload目录以外的文件")
    if file_path.lower().endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("不支持该文件格式")
    for _col in df.select_dtypes(include=["string"]).columns:
        df[_col] = df[_col].astype(object)
    return df

def _save_cleaned(df: pd.DataFrame, file_path: str) -> str:
    """保存清洗后的文件到 cleaned 目录，返回 (path, name)。"""
    ext = os.path.splitext(file_path)[1].lower()
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    clean_name = f"{base_name}_clean{ext}"
    clean_path = os.path.abspath(os.path.join(cleaned_folder, clean_name))
    if ext == ".csv":
        df.to_csv(clean_path, index=False, encoding="utf-8-sig")
    else:
        df.to_excel(clean_path, index=False)
    return clean_path, clean_name

def _make_diff(before: Dict, after: Dict) -> Dict:
    return {
        "rows_before": before["rows"],
        "rows_after": after["rows"],
        "rows_removed": before["rows"] - after["rows"],
        "missing_before": before["missing_total"],
        "missing_after": after["missing_total"],
        "duplicated_before": before["duplicated_rows"],
        "duplicated_after": after["duplicated_rows"],
    }

def _eda_recheck(before_df: pd.DataFrame, after_df: pd.DataFrame) -> Dict:
    """清洗后 EDA 复查（治疗 → 再验证）：对比清洗前后偏度/异常群组/缺失，
    给出是否达标结论与仍需处理清单，与 EDA 诊断闭环。"""
    from routes.preprocess import diagnose_column

    def _summ(dff: pd.DataFrame) -> Dict:
        num = [str(c) for c in dff.select_dtypes(include="number").columns]
        sk, out = [], []
        for c in num:
            d = diagnose_column(dff[c], c)
            sk.append((c, d["skew"]))
            out.append((c, d["outlier_ratio"]))
        return {
            "bad_skew": [c for c, s in sk if abs(s) > 1],
            "skew_cols": {c: round(float(s), 3) for c, s in sk},
            "high_out": [c for c, o in out if o > 0.1],
            "outlier_cols": {c: round(float(o), 3) for c, o in out},
            "missing_total": int(dff.isna().sum().sum()),
            "num_cols": len(num),
        }

    b, a = _summ(before_df), _summ(after_df)
    still = []
    if a["bad_skew"]:
        still.append(f"仍有偏态列（{'、'.join(a['bad_skew'])}），建议对数/Box-Cox/分箱变换")
    if a["high_out"]:
        still.append(f"仍有异常群组（{'、'.join(a['high_out'])}），建议变换后复查离群占比")
    if a["missing_total"] > 0:
        still.append(f"仍有缺失 {a['missing_total']} 个，建议继续填充或删除")
    ok = len(still) == 0
    return {
        "before": b,
        "after": a,
        "verdict": "清洗后数据质量达标，可进入建模" if ok else "清洗完成，但仍有数据质量问题，建议继续处理",
        "still_needed": still,
    }

@app.post("/clean_dataset")
async def clean_dataset(
    file_path: str = Query(),
    user_query: str = Query(),
    model_name: str = Query(default=DEFAULT_MODEL),
):
    logs: List[str] = []

    # 1. 读取数据集（校验路径安全）
    try:
        df = _validate_and_load(file_path)
        logs.append(f"[1/6] 读取数据集：{os.path.basename(file_path)} 成功（{len(df)} 行 × {len(df.columns)} 列）")
    except Exception as e:
        logs.append(f"[1/6] 读取数据集失败：{e}")
        return {"code": -1, "msg": str(e), "logs": logs}

    before_report = compute_quality_report(df)

    # 3. 让模型分析用户意图，输出清洗操作计划（JSON），后端白名单确定性执行
    plan_prompt = f"""
你是数据清洗意图解析器。用户会用自然语言提出数据清洗需求，请理解他的意图，
从下面的操作库中选择 0~N 个清洗操作，按执行顺序组成 JSON 的 ops 数组输出。
只输出 JSON，不要任何解释、代码或 Markdown。

数据集信息：
- 列名：{list(df.columns)}
- 数据类型：
{df.dtypes.to_string()}
- 前5行：
{df.head(5).to_string()}
- 数据质量诊断：
{quality_report_to_text(before_report)}

用户清洗要求：{user_query}

操作库（op 取值，op 与参数平级写在同一个 JSON 对象里）：
- drop_duplicates：删除重复行。参数：subset（可选列名数组或 null）
- drop_missing：删除缺失行。参数：how（"any"或"all"）、subset（可选列名数组或 null）
- fill_missing：填充缺失值。参数：columns（列名数组）、method（"mean"|"median"|"mode"|"zero"|"ffill"|"bfill"|"constant"）、value（仅 method=constant）
- strip：去除字符串列首尾空格。参数：columns（列名数组）
- to_numeric：把列转为数值。参数：columns（列名数组）
- to_datetime：把列转为日期。参数：columns（列名数组）
- rename：重命名列。参数：mapping（{{"旧名":"新名"}}）
- replace：替换某列的值。参数：column（列名）、old（原值）、new（新值）
- drop_columns：删除列。参数：columns（列名数组）
- keep_columns：只保留这些列。参数：columns（列名数组）

【严格规则】
1. 只输出 JSON，格式：{{"ops": [{{"op": "操作名", 参数1: 值1, 参数2: 值2}}, ...]}}（op 和其他参数平级）
2. params 里的列名必须是上面列名中真实存在的，且只对有意义的目标列操作（如只对数值列做 fill_missing mean）。
3. "op" 字段只能取操作库中的操作名，绝对不能填列名或其他词。
错误示例（禁止）：{{"op": "Survived", ...}} —— Survived 是列名不是操作名。
4. 不要添加操作库中没有的操作。
5. 若无需清洗，输出：{{"ops": []}}

【示例】删除重复行、用均值填充 age 缺失、去除 name 首尾空格：
{{"ops": [{{"op": "drop_duplicates", "subset": null}}, {{"op": "fill_missing", "columns": ["age"], "method": "mean"}}, {{"op": "strip", "columns": ["name"]}}]}}

现在输出清洗计划：
"""
    plan = None
    last_err = "无法生成清洗计划"
    df_cleaned = None
    for attempt in range(3):
        if attempt == 0:
            prompt = plan_prompt
        else:
            prompt = (
                f"上次输出的清洗计划无效：{last_err}\\n"
                f"数据集列名：{list(df.columns)}\\n"
                f"可用操作名（op 只能取这些）：{', '.join(CLEAN_OPS)}\\n"
                "请重新只输出合法的 JSON 操作计划（ops 数组），列名必须真实存在，格式："
                '{"ops": [{"op": "操作名", "params": {...}}]}'
            )
        logs.append(f"[2/6] 大模型意图解析：第 {attempt + 1} 次尝试...")
        ollama_res = call_ollama(prompt, model_name=model_name, stream=False, timeout=300)
        if "error" in ollama_res:
            logs.append(f"[2/6] 模型调用错误：{ollama_res['error']}")
            return {"code": -1, "msg": ollama_res["error"], "logs": logs}
        raw = ollama_res.get("response", "")
        print("=" * 60)
        print("[清洗] 模型输出:")
        print(raw)
        print("=" * 60)

        plan = extract_json(raw)
        if isinstance(plan, list):  # 模型直接输出了 ops 数组
            plan = {"ops": plan}
        if not plan or not isinstance(plan, dict) or not isinstance(plan.get("ops"), list):
            last_err = "无法解析出有效的 ops 列表（需为 JSON 格式）"
            logs.append(f"[2/6] 第 {attempt + 1} 次尝试失败：{last_err}；模型原始输出片段：{str(raw)[:200]}")
            print(f"[清洗] 第{attempt+1}次: {last_err}")
            continue

        # 白名单校验 + 顺序执行
        error = None
        try:
            df_cleaned = df.copy()
            for idx, item in enumerate(plan["ops"]):
                if not isinstance(item, dict) or "op" not in item:
                    raise ValueError("操作项缺少 op 字段")
                op = str(item["op"])
                if op not in CLEAN_OPS:
                    raise ValueError(f"操作不在白名单内: {op}")
                if "params" in item:
                    params = item.get("params")  # 兼容嵌套格式
                else:  # 平铺格式：op 之外的其他键即参数
                    params = {k: v for k, v in item.items() if k != "op"}
                if params is None:
                    params = {}
                if not isinstance(params, dict):
                    raise ValueError(f"操作 {op} 的 params 必须是对象")
                df_cleaned = apply_clean_op(df_cleaned, op, params)
                logs.append(f"[3/6] 执行操作 {idx + 1}/{len(plan['ops'])}：{op} 成功")
        except Exception as e:
            error = str(e)
            logs.append(f"[3/6] 执行操作失败：{error}")
            df_cleaned = None  # 失败时清空，避免 df.copy() 导致"假成功"
        if error:
            last_err = error
            logs.append(f"[2/6] 第 {attempt + 1} 次尝试失败：{error}")
            print(f"[清洗] 第{attempt+1}次: {last_err}")
            continue
        logs.append(f"[3/6] 白名单执行完成：共 {len(plan['ops'])} 项操作全部成功")
        break

    if df_cleaned is None or not isinstance(df_cleaned, pd.DataFrame):
        # 降级：模型意图解析失败时，用纯规则解析用户指令再执行一次（不依赖模型）
        logs.append("[4/6] 模型意图解析未成功，尝试规则兜底解析...")
        try:
            ops_fallback, text_fallback = parse_clean_query(user_query, df)
            df_cleaned = df.copy()
            for idx, item in enumerate(ops_fallback):
                op = str(item["op"])
                params = {k: v for k, v in item.items() if k != "op"}
                df_cleaned = apply_clean_op(df_cleaned, op, params)
                logs.append(f"[4/6] 规则执行操作 {idx + 1}/{len(ops_fallback)}：{op} 成功")
            plan = {"ops": ops_fallback}
            last_err = None
            logs.append(f"[4/6] 规则兜底执行完成：共 {len(ops_fallback)} 项操作")
        except Exception as fallback_err:
            last_err = f"模型意图解析失败（{last_err}），规则兜底也失败：{fallback_err}"
            logs.append(f"[4/6] 规则兜底失败：{fallback_err}")
        if df_cleaned is None or not isinstance(df_cleaned, pd.DataFrame):
            logs.append("[5/6] 清洗失败，无法得到清洗结果")
            if "操作不在白名单内" in last_err:
                bad_word = last_err.split(":", 1)[1].strip() if ":" in last_err else last_err
                return {
                    "code": -1,
                    "msg": (
                        f"模型生成的清洗计划无效：把“{bad_word}”当成了操作名（它可能是列名或其他词）。"
                        "已自动重试3次仍未成功。建议：①直接再点一次清洗重试；"
                        "②把要求说得简单直白，如“删除重复行”“用均值填充缺失值”分步清洗。"
                    ),
                    "logs": logs,
                }
            return {"code": -1, "msg": f"清洗失败（已自动重试3次）：{last_err}。请重试或简化清洗要求", "logs": logs}

    # 3.5 数值运算写回原数据集（在原文件上追加新列）
    try:
        _writeback_math_to_source(file_path, plan.get("ops") if isinstance(plan, dict) else None)
        logs.append("[3.5/6] 数值运算结果已写回原数据集")
    except Exception as e:
        logs.append(f"[3.5/6] 数值运算写回原数据集失败：{e}")
        return {"code": -1, "msg": f"数值运算写回原数据集失败: {str(e)}", "logs": logs}

    after_report = compute_quality_report(df_cleaned)

    # 4. 保存清洗后的文件
    try:
        clean_path, clean_name = _save_cleaned(df_cleaned, file_path)
        logs.append(f"[5/6] 保存清洗结果：{clean_name} 成功")
    except Exception as e:
        logs.append(f"[5/6] 保存清洗结果失败：{e}")
        return {"code": -1, "msg": f"保存清洗结果失败: {str(e)}", "logs": logs}

    # 5. 清洗前后对比（真实统计）
    diff = _make_diff(before_report, after_report)
    # 5.5 清洗后 EDA 复查（治疗 → 再验证，与 EDA 诊断闭环）
    try:
        eda_check = _eda_recheck(df, df_cleaned)
    except Exception as e:
        eda_check = {"verdict": f"EDA 复查失败：{e}", "before": None, "after": None, "still_needed": []}

    # 6. 确定性总结（严格基于已执行的操作与真实统计，不依赖模型）
    plan_ops = plan.get("ops") if isinstance(plan, dict) else None
    if plan_ops:
        summary = f"已按你的要求完成 {len(plan_ops)} 项清洗：{ops_to_text(plan_ops)}。"
    else:
        summary = "清洗完成。"
    summary += (
        f"清洗前：{before_report['rows']} 行 / 缺失 {before_report['missing_total']} / "
        f"重复 {before_report['duplicated_rows']}；清洗后：{after_report['rows']} 行 / "
        f"缺失 {after_report['missing_total']} / 重复 {after_report['duplicated_rows']}。"
    )
    logs.append("[6/6] 生成清洗总结，清洗完成")

    return {
        "code": 0,
        "msg": "清洗成功",
        "data": {
            "summary": summary,
            "before": before_report,
            "after": after_report,
            "diff": diff,
            "eda_check": eda_check,
            "cleaned_file_path": clean_path,
            "cleaned_filename": clean_name,
            "plan": plan,
            "model_name": model_name,
        },
        "logs": logs,
    }

@app.post("/clean_fast")
async def clean_fast(
    file_path: str = Query(),
    ops_json: str = Query(..., description="前端传入的确定性操作 JSON 数组"),
    model_name: str = Query(default=DEFAULT_MODEL),
):
    # 1. 读取数据集（校验路径安全）
    try:
        df = _validate_and_load(file_path)
    except Exception as e:
        return {"code": -1, "msg": str(e)}

    before_report = compute_quality_report(df)

    # 3. 解析并执行（前端传入的确定性操作，白名单校验，不依赖模型）
    try:
        ops = json.loads(ops_json)
        if not isinstance(ops, list):
            raise ValueError("操作参数必须是数组")
        df_cleaned = df.copy()
        for item in ops:
            if not isinstance(item, dict) or "op" not in item:
                raise ValueError("操作项缺少 op 字段")
            op = str(item["op"])
            if op not in CLEAN_OPS:
                raise ValueError(f"操作不在白名单内: {op}")
            params = {k: v for k, v in item.items() if k != "op"}
            if params is None:
                params = {}
            df_cleaned = apply_clean_op(df_cleaned, op, params)
    except Exception as e:
        return {"code": -1, "msg": f"执行失败: {str(e)}"}

    # 3.5 数值运算写回原数据集（在原文件上追加新列）
    try:
        _writeback_math_to_source(file_path, ops)
    except Exception as e:
        return {"code": -1, "msg": f"数值运算写回原数据集失败: {str(e)}"}

    after_report = compute_quality_report(df_cleaned)

    # 4. 保存清洗后的文件
    try:
        clean_path, clean_name = _save_cleaned(df_cleaned, file_path)
    except Exception as e:
        return {"code": -1, "msg": f"保存清洗结果失败: {str(e)}"}

    # 5. 清洗前后对比（真实统计）
    diff = _make_diff(before_report, after_report)
    # 5.5 清洗后 EDA 复查（治疗 → 再验证，与 EDA 诊断闭环）
    try:
        eda_check = _eda_recheck(df, df_cleaned)
    except Exception as e:
        eda_check = {"verdict": f"EDA 复查失败：{e}", "before": None, "after": None, "still_needed": []}

    # 6. 模型写清洗总结（严格基于真实统计）
    summary_prompt = f"""
数据清洗已完成，以下是清洗前后的真实统计：
清洗前：行数 {before_report['rows']}，缺失值 {before_report['missing_total']}，重复行 {before_report['duplicated_rows']}
清洗后：行数 {after_report['rows']}，缺失值 {after_report['missing_total']}，重复行 {after_report['duplicated_rows']}

要求：输出简洁的中文清洗报告，说明清洗前后数据质量变化，严格基于上述统计数字，禁止编造，控制在200字以内。
"""
    summary_res = call_ollama(summary_prompt, model_name=model_name, stream=False, timeout=300)
    summary = summary_res.get("response", "") if "error" not in summary_res else "（模型总结失败，可查看下方统计数据）"

    return {
        "code": 0,
        "msg": "清洗成功",
        "data": {
            "summary": summary,
            "before": before_report,
            "after": after_report,
            "diff": diff,
            "eda_check": eda_check,
            "cleaned_file_path": clean_path,
            "cleaned_filename": clean_name,
            "plan": ops,
            "model_name": model_name,
        },
    }

@app.get("/download_cleaned")
async def download_cleaned(file_path: str = Query()):
    abs_cleaned = os.path.abspath(cleaned_folder)
    abs_file = os.path.abspath(file_path)
    if not abs_file.startswith(abs_cleaned):
        return {"code": -1, "msg": "禁止下载upload目录以外的文件"}
    if not os.path.exists(abs_file):
        return {"code": -1, "msg": "文件不存在"}
    return FileResponse(abs_file, filename=os.path.basename(abs_file))

@app.post("/delete_cleaned")
async def delete_cleaned(file_path: str = Query()):
    """用户手动下载清洗后的数据后，自动清除该缓存文件（仅限 cleaned 目录内）。"""
    abs_cleaned = os.path.abspath(cleaned_folder)
    abs_file = os.path.abspath(file_path)
    if not abs_file.startswith(abs_cleaned):
        return {"code": -1, "msg": "禁止删除upload目录以外的文件"}
    if not os.path.exists(abs_file):
        return {"code": 0, "msg": "缓存文件不存在，无需删除", "data": {"deleted": False}}
    try:
        os.remove(abs_file)
        return {"code": 0, "msg": "缓存已清除", "data": {"deleted": True}}
    except Exception as e:
        return {"code": -1, "msg": f"删除缓存失败: {str(e)}"}
# ================= 智能清洗工作台 /clean_run =================
# 三大能力（全部由后端确定性执行，前端只传选项）：
#   1. 类型纠正：数字字符串→数值、时间戳→datetime、分类文本→category；转换失败置 NaN
#   2. 缺失值处理：删除高缺失列 / 常数填充 / 统计量填充 / 插值填充
#   3. 重复值处理：删除完全重复行
# 结果含 清洗前后质量对比 + 类型变化 + 删除列 + 重复行 + EDA 复查 + 下载文件

def _json_safe(obj):
    """递归清洗非有限浮点，保证 JSON 可序列化"""
    if isinstance(obj, float):
        import math
        return None if not math.isfinite(obj) else round(obj, 6)
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def _col_kind(s: pd.Series, col: str) -> str:
    """判断列类型倾向：numeric / datetime / category / text"""
    if s.dtype.kind in "ifu":  # int/float/uint
        return "numeric"
    if s.dtype == "datetime64[ns]":
        return "datetime"
    if s.dtype.name.startswith("category"):
        return "category"
    non_null = s.dropna()
    if len(non_null) == 0:
        return "text"
    lowered = str(col).lower()
    # 时间戳：列名含时间词 或 值匹配常见日期格式
    if any(k in lowered for k in ("time", "date", "timestamp", "datetime", "时间", "日期")):
        parsed = pd.to_datetime(non_null.astype(str), errors="coerce")
        if parsed.notna().mean() >= 0.8:
            return "datetime"
    # 数字字符串
    conv = pd.to_numeric(non_null.astype(str).str.replace(",", ""), errors="coerce")
    if len(conv) > 0 and conv.notna().mean() >= 0.8:
        return "numeric"
    # 分类文本：唯一值比例低
    if len(non_null) > 0 and non_null.nunique() / len(non_null) < 0.5:
        return "category"
    return "text"


def _type_fix(df: pd.DataFrame) -> Dict:
    """类型纠正。返回类型变化清单。转换失败的值由 coerce 置 NaN。"""
    changes = []
    for col in df.columns:
        s = df[col]
        if s.dtype.name.startswith("category"):
            continue
        kind = _col_kind(s, col)
        old = str(s.dtype)
        if kind == "numeric":
            df[col] = pd.to_numeric(s.astype(str).str.replace(",", ""), errors="coerce")
        elif kind == "datetime":
            df[col] = pd.to_datetime(s.astype(str), errors="coerce")
        elif kind == "category":
            df[col] = s.astype("category")
        new = str(df[col].dtype)
        if new != old:
            changes.append({"col": str(col), "from": old, "to": new,
                            "kind": kind, "failed_to_nan": int(df[col].isna().sum() - s.isna().sum())})
    return changes


def _missing_fix(df: pd.DataFrame, strategy: str, threshold: float) -> Dict:
    """缺失值处理。strategy: none/delete/const/stat/interpolate"""
    result = {"deleted_columns": [], "filled": []}
    if strategy == "none":
        return result
    missing_ratio = df.isna().mean()
    # 1) 删除策略：缺失率 > 阈值 删列
    if strategy == "delete":
        drop_cols = [str(c) for c in missing_ratio.index if missing_ratio[c] > threshold]
        for c in drop_cols:
            result["deleted_columns"].append({"col": c, "missing_ratio": round(float(missing_ratio[c]), 4)})
        if drop_cols:
            df.drop(columns=drop_cols, inplace=True)
        return result

    # 2/3/4) 填充策略
    for col in df.columns:
        s = df[col]
        if s.isna().sum() == 0:
            continue
        filled_count = int(s.isna().sum())
        kind = _col_kind(s, col)
        if strategy == "const":
            # 常数填充：数值填 0，分类/文本填"未知"
            if kind == "numeric":
                df[col] = s.fillna(0)
                used = "常数0"
            else:
                df[col] = s.fillna("未知")
                used = "常数「未知」"
        elif strategy == "stat":
            if kind == "numeric":
                # 偏态（|skew|>1）用中位数，正态用均值
                skew = float(s.skew()) if s.notna().nunique() > 1 else 0.0
                fill_val = float(s.median()) if abs(skew) > 1 else float(s.mean())
                df[col] = s.fillna(fill_val)
                used = f"统计量({'中位数' if abs(skew) > 1 else '均值'})"
            else:
                mode_vals = s.dropna().mode()
                fill_val = mode_vals.iloc[0] if len(mode_vals) > 0 else "未知"
                df[col] = s.fillna(fill_val)
                used = "众数"
        elif strategy == "interpolate":
            if kind == "numeric":
                df[col] = s.interpolate(method="linear", limit_direction="both")
                # 首尾仍可能 NaN → 用中位数兜底
                if df[col].isna().any():
                    df[col] = df[col].fillna(float(s.median()) if s.notna().nunique() else 0)
                used = "线性插值"
            else:
                df[col] = s.ffill().bfill().fillna("未知")
                used = "前向/后向填充"
        filled_now = int(df[col].isna().sum())
        result["filled"].append({"col": str(col), "method": used, "filled": filled_count,
                                 "remaining": filled_now})
    return result


def _dup_fix(df: pd.DataFrame) -> int:
    """删除完全重复行，返回删除行数"""
    before = len(df)
    df.drop_duplicates(inplace=True)
    return before - len(df)


@app.post("/clean_run")
async def clean_run(file_path: str = Query(), type_fix: bool = Query(True),
                    missing_strategy: str = Query("stat"), missing_threshold: float = Query(0.6),
                    dup_fix: bool = Query(True)):
    """智能清洗工作台：类型纠正 + 缺失值处理（删除/常数/统计量/插值）+ 重复值处理。
    全部后端确定性执行；返回清洗前后质量对比、执行明细与下载文件。"""
    try:
        df = _validate_and_load(file_path)
    except Exception as e:
        return {"code": -1, "msg": f"读取数据失败: {str(e)}"}
    before = compute_quality_report(df)
    before_df = df.copy()

    if missing_strategy not in ("none", "delete", "const", "stat", "interpolate"):
        missing_strategy = "stat"

    type_changes, missing_res, dup_removed = [], {}, 0
    try:
        if type_fix:
            type_changes = _type_fix(df)
        missing_res = _missing_fix(df, missing_strategy, missing_threshold)
        if dup_fix:
            dup_removed = _dup_fix(df)
    except Exception as e:
        return {"code": -1, "msg": f"清洗执行失败: {str(e)}"}

    after = compute_quality_report(df)
    diff = _make_diff(before, after)

    # 保存（category 列转 object 兼容 CSV）
    save_df = df.copy()
    for c in save_df.select_dtypes(include="category").columns:
        save_df[c] = save_df[c].astype(object)
    try:
        clean_path, clean_name = _save_cleaned(save_df, file_path)
    except Exception as e:
        return {"code": -1, "msg": f"保存清洗结果失败: {str(e)}"}

    eda_check = _eda_recheck(before_df, save_df)
    return {"code": 0, "msg": "清洗完成", "data": _json_safe({
        "before": before, "after": after, "diff": diff,
        "type_changes": type_changes,
        "missing": missing_res,
        "dup_removed": dup_removed,
        "eda_check": eda_check,
        "clean_path": clean_path, "clean_name": clean_name,
    })}
# ================= 特征构造 /fe_construct =================
# 四大类（全部由后端确定性执行，前端只传选项）：
#   1. 聚合统计特征：按分组列分组，对数值列计算均值/最大值/标准差/偏度/中位数/计数等
#   2. 组合交叉特征：两个数值列四则运算，生成新特征列
#   3. 时序特征：从时间列提取周期信息（年/月/日/星期/季度/周/小时）
#   4. 频次与计数特征：统计类别值全局出现次数，生成频次特征列
AGG_STATS = ("mean", "max", "min", "std", "skew", "median", "sum", "count")
AGG_STAT_CN = {"mean": "均值", "max": "最大值", "min": "最小值", "std": "标准差",
               "skew": "偏度", "median": "中位数", "sum": "求和", "count": "计数"}
TIME_PARTS = ("year", "month", "day", "weekday", "quarter", "week", "hour")
TIME_PART_CN = {"year": "年", "month": "月", "day": "日", "weekday": "星期",
                "quarter": "季度", "week": "周", "hour": "小时"}


def _fe_construct(df: pd.DataFrame, ops: List[Dict]):
    """特征构造四大类，白名单确定性执行。返回 (df, added_cols, summary_parts)。"""
    df = df.copy()
    added = []
    parts = []
    for item in ops or []:
        if not isinstance(item, dict) or not item.get("type"):
            raise ValueError("特征构造项缺少 type 字段")
        t = str(item["type"])
        if t == "agg":  # 聚合统计特征
            g = item.get("group_col")
            v = item.get("value_col")
            stats = item.get("stats") or []
            if not g or not v:
                raise ValueError("聚合统计：请选择分组列与聚合列")
            _check_col(df, g)
            _check_col(df, v)
            for st in stats:
                if st not in AGG_STATS:
                    raise ValueError("聚合统计：不支持的统计量 " + st)
            s = pd.to_numeric(df[v], errors="coerce")
            for st in stats:
                name = v + "_" + AGG_STAT_CN[st]
                if name in df.columns:
                    raise ValueError("聚合统计：新列已存在 " + name)
                df[name] = s.groupby(df[g]).transform(st)
                added.append(name)
                parts.append("按「" + g + "」分组计算「" + v + "」" + AGG_STAT_CN[st])
        elif t == "combine":  # 组合交叉特征（四则运算）
            col1 = item.get("col1")
            col2 = item.get("col2")
            operator = item.get("operator")
            new_name = (item.get("new_name") or "").strip()
            if not col1 or not col2:
                raise ValueError("组合交叉：请选择两个列")
            _check_col(df, col1)
            _check_col(df, col2)
            if col1 == col2:
                raise ValueError("组合交叉：两个列不能相同")
            if operator not in ("+", "-", "*", "/"):
                raise ValueError("组合交叉：不支持的运算符 " + operator)
            if not new_name:
                raise ValueError("组合交叉：新列名不能为空")
            if new_name in df.columns:
                raise ValueError("组合交叉：新列名已存在 " + new_name)
            s1 = pd.to_numeric(df[col1], errors="coerce")
            s2 = pd.to_numeric(df[col2], errors="coerce")
            if operator == "+":
                res = s1 + s2
            elif operator == "-":
                res = s1 - s2
            elif operator == "*":
                res = s1 * s2
            else:
                res = s1 / s2
            df[new_name] = res
            op_cn = {"+": "加", "-": "减", "*": "乘", "/": "除"}[operator]
            added.append(new_name)
            parts.append("「" + col1 + "」" + op_cn + "「" + col2 + "」→ " + new_name)
        elif t == "time":  # 时序特征
            tc = item.get("time_col")
            plist = item.get("parts") or []
            if not tc:
                raise ValueError("时序特征：请选择时间列")
            _check_col(df, tc)
            ts = pd.to_datetime(df[tc], errors="coerce")
            if ts.isna().all():
                raise ValueError("时序特征：时间列无法解析 " + tc)
            for p in plist:
                if p not in TIME_PARTS:
                    raise ValueError("时序特征：不支持的提取项 " + p)
                name = tc + "_" + TIME_PART_CN[p]
                if name in df.columns:
                    raise ValueError("时序特征：新列已存在 " + name)
                if p == "year":
                    df[name] = ts.dt.year
                elif p == "month":
                    df[name] = ts.dt.month
                elif p == "day":
                    df[name] = ts.dt.day
                elif p == "weekday":
                    df[name] = ts.dt.dayofweek
                elif p == "quarter":
                    df[name] = ts.dt.quarter
                elif p == "week":
                    df[name] = ts.dt.isocalendar().week.astype("Int64")
                elif p == "hour":
                    df[name] = ts.dt.hour
                added.append(name)
                parts.append("从「" + tc + "」提取" + TIME_PART_CN[p])
        elif t == "freq":  # 频次与计数特征
            cc = item.get("cat_col")
            new_name = (item.get("new_name") or (cc + "_频次")).strip()
            if not cc:
                raise ValueError("频次与计数：请选择类别列")
            _check_col(df, cc)
            if not new_name:
                raise ValueError("频次与计数：新列名不能为空")
            if new_name in df.columns:
                raise ValueError("频次与计数：新列已存在 " + new_name)
            df[new_name] = df[cc].map(df[cc].value_counts())
            added.append(new_name)
            parts.append("「" + cc + "」类别全局频次 → " + new_name)
        else:
            raise ValueError("未知特征构造类型: " + t)
    return df, added, parts


@app.post("/fe_construct")
async def fe_construct(file_path: str = Query(), ops_json: str = Query(..., description="特征构造操作 JSON 数组")):
    """特征构造工作台：聚合统计 / 组合交叉 / 时序 / 频次计数，全部后端确定性执行。"""
    try:
        df = _validate_and_load(file_path)
    except Exception as e:
        return {"code": -1, "msg": "读取数据失败: " + str(e)}
    before = compute_quality_report(df)
    try:
        ops = json.loads(ops_json)
        if not isinstance(ops, list):
            raise ValueError("操作参数必须是数组")
        df2, added, parts = _fe_construct(df, ops)
    except Exception as e:
        return {"code": -1, "msg": "特征构造失败: " + str(e)}
    after = compute_quality_report(df2)
    # category 列转 object 兼容 CSV
    save_df = df2.copy()
    for c in save_df.select_dtypes(include="category").columns:
        save_df[c] = save_df[c].astype(object)
    try:
        clean_path, clean_name = _save_cleaned(save_df, file_path)
    except Exception as e:
        return {"code": -1, "msg": "保存结果失败: " + str(e)}
    summary = "；".join(parts) if parts else "未执行任何特征构造"
    return {"code": 0, "msg": "特征构造完成", "data": _json_safe({
        "summary": summary,
        "added_cols": added,
        "before": before,
        "after": after,
        "clean_path": clean_path,
        "clean_name": clean_name,
    })}
# ================= 数据变换与规范化 /fe_transform =================
# 三类（全部由后端确定性执行，前端只传选项，替换所选数值列原值）：
#   1. standardize 标准化： (x - mean) / std，适用于线性模型、SVM、PCA
#   2. minmax 归一化：      (x - min) / (max - min) → [0,1]，适用于神经网络、距离计算（KNN）
#   3. robust 稳健缩放：    (x - median) / IQR，受异常值影响小
TRANSFORM_TYPES = ("standardize", "minmax", "robust")
TRANSFORM_CN = {"standardize": "标准化", "minmax": "归一化", "robust": "稳健缩放"}


def _fe_transform(df: pd.DataFrame, ops: List[Dict]):
    """数据变换与规范化，白名单确定性执行。返回 (df, changed)。"""
    df = df.copy()
    changed = []
    for item in ops or []:
        if not isinstance(item, dict) or not item.get("type"):
            raise ValueError("变换项缺少 type 字段")
        t = str(item["type"])
        if t not in TRANSFORM_TYPES:
            raise ValueError("不支持的变换类型: " + t)
        cols = item.get("columns") or []
        if not cols:
            raise ValueError(TRANSFORM_CN[t] + "：请选择至少一个数值列")
        for c in cols:
            _check_col(df, c)
            s = pd.to_numeric(df[c], errors="coerce")
            if s.nunique() < 2:
                raise ValueError(TRANSFORM_CN[t] + "：列「" + c + "」有效值不足，无法变换")
            before = {"mean": float(s.mean()), "std": float(s.std(ddof=0)), "min": float(s.min()), "max": float(s.max()), "median": float(s.median())}
            if t == "standardize":
                mean = float(s.mean())
                std = float(s.std(ddof=0))
                if std == 0 or not std or std != std:
                    raise ValueError("标准化：列「" + c + "」标准差为 0，无法标准化")
                df[c] = (s - mean) / std
            elif t == "minmax":
                mn = float(s.min())
                mx = float(s.max())
                if mx - mn == 0:
                    raise ValueError("归一化：列「" + c + "」最大值等于最小值，无法归一化")
                df[c] = (s - mn) / (mx - mn)
            elif t == "robust":
                med = float(s.median())
                q1 = float(s.quantile(0.25))
                q3 = float(s.quantile(0.75))
                iqr = q3 - q1
                if iqr == 0:
                    raise ValueError("稳健缩放：列「" + c + "」IQR 为 0，无法缩放")
                df[c] = (s - med) / iqr
            s2 = pd.to_numeric(df[c], errors="coerce")
            changed.append({"col": str(c), "type": t, "transform": TRANSFORM_CN[t],
                            "before": before,
                            "after": {"mean": float(s2.mean()), "std": float(s2.std(ddof=0)),
                                      "min": float(s2.min()), "max": float(s2.max()),
                                      "median": float(s2.median())}})
    return df, changed


@app.post("/fe_transform")
async def fe_transform(file_path: str = Query(), ops_json: str = Query(..., description="变换操作 JSON 数组")):
    """数据变换与规范化：标准化 / 归一化 / 稳健缩放，替换所选数值列原值，全部后端确定性执行。"""
    try:
        df = _validate_and_load(file_path)
    except Exception as e:
        return {"code": -1, "msg": "读取数据失败: " + str(e)}
    before = compute_quality_report(df)
    try:
        ops = json.loads(ops_json)
        if not isinstance(ops, list):
            raise ValueError("操作参数必须是数组")
        df2, changed = _fe_transform(df, ops)
    except Exception as e:
        return {"code": -1, "msg": "变换失败: " + str(e)}
    after = compute_quality_report(df2)
    save_df = df2.copy()
    for c in save_df.select_dtypes(include="category").columns:
        save_df[c] = save_df[c].astype(object)
    try:
        clean_path, clean_name = _save_cleaned(save_df, file_path)
    except Exception as e:
        return {"code": -1, "msg": "保存结果失败: " + str(e)}
    parts = []
    seen = set()
    for ch in changed:
        k = (ch["col"], ch["transform"])
        if k not in seen:
            seen.add(k)
            parts.append("「" + ch["col"] + "」" + ch["transform"])
    summary = "；".join(parts) if parts else "未执行任何变换"
    return {"code": 0, "msg": "变换完成", "data": _json_safe({
        "summary": summary,
        "changed": changed,
        "before": before,
        "after": after,
        "clean_path": clean_path,
        "clean_name": clean_name,
    })}
