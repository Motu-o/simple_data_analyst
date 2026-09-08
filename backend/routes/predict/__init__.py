# -*- coding: utf-8 -*-
"""预测分析模块：对数据集做回归/分类预测，输出评估指标、特征重要性、预测结果与图表。
支持两种模式：1) 自动训练模型预测；2) 加载训练模块产出的 best_loss / last_loss 已训练模型预测。"""
import os
from collections import Counter

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fastapi import Query, UploadFile, File
from datetime import datetime

from config import app, cleaned_folder, model_folder, DEFAULT_MODEL
from utils import call_ollama, load_df_checked

def _load_trained_bundle(model_file):
    """安全加载训练模块保存的模型 bundle（仅允许 model_folder 内文件）。"""
    os.makedirs(model_folder, exist_ok=True)
    p = os.path.abspath(os.path.join(model_folder, os.path.basename(model_file)))
    if os.path.dirname(p) != os.path.abspath(model_folder) or not os.path.exists(p):
        raise ValueError(f"模型文件不存在: {model_file}")
    b = joblib.load(p)
    if "model" not in b:
        raise ValueError("模型文件格式不正确")
    return b

def _build_X_using_bundle(df, bundle):
    """按训练模型的特征工程（keep_num 中位数填充 + keep_cat one-hot 对齐）对数据集构建 X。"""
    keep_num = bundle.get("keep_num") or []
    keep_cat = bundle.get("keep_cat") or []
    median_map = bundle.get("median_map") or {}
    onehot_cols = bundle.get("onehot_columns") or []
    parts = []
    if keep_num:
        arrs = []
        for c in keep_num:
            if c not in df.columns:
                arrs.append(np.zeros(len(df)))
                continue
            col = pd.to_numeric(df[c], errors="coerce").fillna(median_map.get(c, 0))
            arrs.append(col.values.astype(float))
        parts.append(np.column_stack(arrs))
    if keep_cat:
        present = [c for c in keep_cat if c in df.columns]
        d = pd.get_dummies(df[present], drop_first=False) if present else pd.DataFrame(index=df.index)
        cols = []
        for col_name in onehot_cols:
            if col_name in d.columns:
                cols.append(d[col_name].values.astype(float))
            else:
                cols.append(np.zeros(len(df)))
        if cols:
            parts.append(np.column_stack(cols))
    if not parts:
        raise ValueError("数据集缺少模型所需的特征列")
    X = np.hstack(parts) if len(parts) > 1 else parts[0]
    return X, (keep_num + onehot_cols)

def _build_learning_curve(bundle, X, y, task):
    """对已训练模型（SGD）重建未拟合模型，在给定数据上计算学习曲线
    （不同训练样本占比下的训练/验证得分）。返回 Chart.js line series 或 None。"""
    try:
        from sklearn.model_selection import learning_curve, ShuffleSplit
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        algo = bundle.get("algorithm")
        params = bundle.get("params") or {}
        if algo == "linear_regression":
            from sklearn.linear_model import SGDRegressor
            from sklearn.compose import TransformedTargetRegressor
            model = TransformedTargetRegressor(
                regressor=SGDRegressor(alpha=float(params.get("alpha", 0.0001)), learning_rate="invscaling",
                                       eta0=min(float(params.get("learning_rate", 0.01)), 0.001), power_t=0.25,
                                       max_iter=500, tol=1e-3, random_state=42),
                transformer=StandardScaler())
            scoring = "neg_mean_squared_error"; ylab = "MSE（越小越好）"; neg = True
        elif algo == "logistic_regression":
            from sklearn.linear_model import SGDClassifier
            model = SGDClassifier(loss="log_loss", alpha=float(params.get("alpha", 0.0001)), learning_rate="constant",
                                  eta0=float(params.get("learning_rate", 0.01)), max_iter=500, tol=1e-3, random_state=42)
            scoring = "accuracy"; ylab = "Accuracy（越大越好）"; neg = False
        else:
            return None
        X = np.asarray(X, dtype=float); y = np.asarray(y)
        if len(X) > 2000:
            idx = np.random.RandomState(42).choice(len(X), 2000, replace=False)
            X, y = X[idx], y[idx]
        if len(X) < 30:
            return None
        sizes = np.linspace(0.2, 1.0, 5)
        cv = ShuffleSplit(n_splits=3, test_size=0.2, random_state=42)
        pipe = make_pipeline(StandardScaler(), model)
        tr_sizes, tr_scores, va_scores = learning_curve(pipe, X, y, train_sizes=sizes, cv=cv,
                                                        scoring=scoring, error_score="raise", random_state=42)
        tr_mean = [round(float(v), 4) for v in tr_scores.mean(axis=1)]
        va_mean = [round(float(v), 4) for v in va_scores.mean(axis=1)]
        if neg:
            tr_mean = [round(-float(v), 4) for v in tr_scores.mean(axis=1)]
            va_mean = [round(-float(v), 4) for v in va_scores.mean(axis=1)]
        return {
            "title": f"学习曲线（{algo}）",
            "type": "line",
            "series": {"type": "line", "labels": [f"{int(s * 100)}%" for s in sizes],
                       "train_scores": tr_mean, "val_scores": va_mean,
                       "train_label": "训练得分", "val_label": "验证得分",
                       "x_label": "训练样本占比", "y_label": ylab,
                       "hint": "横轴为训练样本占比；蓝线=训练得分，粉线=验证得分。曲线随样本量增大逐渐收敛且两线接近说明泛化良好；两线长期差距大说明过拟合。"},
        }
    except Exception:
        return None

@app.post("/predict")
async def predict_analysis(
    file_path: str = Query(),
    target: str = Query(default="", description="目标列名"),
    predict_type: str = Query(default="auto", description="auto/regression/classification"),
    features: str = Query(default="", description="逗号分隔的特征列，空则自动选择"),
    model_name: str = Query(default=DEFAULT_MODEL),
    model_file: str = Query(default="", description="训练模块产出的已训练模型文件名（best_loss/last_loss），非空则用该模型预测"),
):
    # 1. 读取数据集（校验路径安全）
    try:
        df = load_df_checked(file_path)
    except Exception as e:
        return {"code": -1, "msg": str(e)}

    # 1.5 使用训练模块产出的已训练模型预测
    if model_file:
        return await _predict_with_trained_model(df, model_file, target, model_name)

    if target not in df.columns:
        return {"code": -1, "msg": f"目标列不存在: {target}（可用列：{list(df.columns)}）"}

    # 3. 选择特征列
    feature_cols = []
    if features.strip():
        feature_cols = [c.strip() for c in features.split(",") if c.strip()]
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            return {"code": -1, "msg": f"特征列不存在: {missing}"}
    else:
        for c in df.columns:
            if c == target:
                continue
            if pd.api.types.is_numeric_dtype(df[c]):
                feature_cols.append(c)
            elif df[c].nunique(dropna=True) <= 10:
                feature_cols.append(c)
        if not feature_cols:
            return {"code": -1, "msg": "没有可用于预测的特征列，请手动指定特征"}

    # 4. 数据准备（移除目标缺失，数值填充中位数，类别 one-hot）
    mask = df[target].notna()
    df2 = df[mask].copy()
    if len(df2) < 5:
        return {"code": -1, "msg": "目标列有效样本过少（<5），无法训练模型"}
    X_raw = df2[feature_cols].copy()
    numeric_feats = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df2[c])]
    cat_feats = [c for c in feature_cols if c not in numeric_feats]
    for c in numeric_feats:
        X_raw[c] = X_raw[c].fillna(X_raw[c].median())
    if cat_feats:
        X = pd.get_dummies(X_raw, columns=cat_feats, drop_first=False)
    else:
        X = X_raw.copy()
    X = X.astype(float)
    feature_names = list(X.columns)
    n = len(X)

    # 5. 任务类型判断
    y_unique = df2[target].nunique(dropna=True)
    if predict_type == "classification":
        task = "classification"
    elif predict_type == "regression":
        task = "regression"
    else:
        if pd.api.types.is_numeric_dtype(df2[target]) and y_unique > 10:
            task = "regression"
        else:
            task = "classification"

    # 5.5 目标列类型与任务类型一致性检查
    if task == "regression" and not pd.api.types.is_numeric_dtype(df2[target]):
        return {"code": -1, "msg": f"目标列“{target}”不是数值类型（{df2[target].dtype}，可能是分类标签），无法做回归预测。请选择“分类”预测类型，或更换数值型目标列。"}

    # 6. 目标编码与模型训练（sklearn 确定性执行）
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score

    classes = []
    if task == "classification":
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = le.fit_transform(df2[target].astype(str))
        classes = [str(c) for c in le.classes_]
    else:
        y = df2[target].astype(float).values

    model = None
    metrics = {}
    split_note = ""
    if task == "regression":
        algorithm = "线性回归 LinearRegression"
        if n >= 20:
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LinearRegression().fit(Xtr, ytr)
            pte = model.predict(Xte)
            metrics = {
                "r2": round(float(r2_score(yte, pte)), 4),
                "mae": round(float(mean_absolute_error(yte, pte)), 4),
                "rmse": round(float(np.sqrt(mean_squared_error(yte, pte))), 4),
            }
        else:
            model = LinearRegression().fit(X, y)
            split_note = "样本量过少（<20），未划分测试集，指标仅供参考"
        importances = model.coef_
    else:
        algorithm = "逻辑回归 LogisticRegression"
        try:
            if n >= 20:
                strat = None
                if len(set(y)) > 1:
                    strat = y
                Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=strat)
                model = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
                pte = model.predict(Xte)
                metrics = {"accuracy": round(float(accuracy_score(yte, pte)), 4)}
            else:
                model = LogisticRegression(max_iter=1000).fit(X, y)
                split_note = "样本量过少（<20），未划分测试集，指标仅供参考"
        except Exception as e:
            return {"code": -1, "msg": f"分类模型训练失败（样本分布不均或类别过少）: {str(e)}"}
        if model.coef_.ndim == 1:
            importances = model.coef_
        else:
            importances = np.mean(np.abs(model.coef_), axis=0)

    # 7. 全量预测
    y_pred_all = model.predict(X)

    # 8. 特征重要性
    imp = sorted(zip(feature_names, importances), key=lambda t: abs(t[1]), reverse=True)
    importance = [{"feature": f, "importance": round(float(v), 4)} for f, v in imp[:15]]

    # 9. 结果表与保存
    result = df2.copy()
    if task == "classification":
        result["预测值"] = [classes[int(i)] for i in y_pred_all]
    else:
        result["预测值"] = [round(float(v), 4) for v in y_pred_all]
    ext = os.path.splitext(file_path)[1].lower()
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    pred_name = f"{base_name}_predict_{datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}"
    pred_path = os.path.abspath(os.path.join(cleaned_folder, pred_name))
    try:
        if ext == ".csv":
            result.to_csv(pred_path, index=False, encoding="utf-8-sig")
        else:
            result.to_excel(pred_path, index=False)
    except Exception as e:
        return {"code": -1, "msg": f"保存预测结果失败: {str(e)}"}

    # 10. 前端渲染图表数据（Chart.js 交互式，替代静态 PNG）
    charts = []
    if task == "regression":
        pts = [[round(float(a), 4), round(float(p), 4)] for a, p in zip(y, y_pred_all)]
        if len(pts) > 1500:  # 采样避免点数过多影响渲染
            step = (len(pts) + 1499) // 1500
            pts = pts[::step]
        charts.append({
            "title": f"实际 vs 预测散点图（目标列：{target}）",
            "type": "scatter",
            "series": {"type": "scatter", "points": pts, "x_label": "实际值", "y_label": "预测值", "col": target,
                       "hint": f"目标列：{target}；特征列：{', '.join(map(str, feature_cols))}；每个点代表一个样本：横轴为实际值，纵轴为模型预测值"},
        })
    else:
        cnt = Counter(classes[int(i)] for i in y_pred_all)
        labels = list(cnt.keys()); vals = [cnt[k] for k in labels]
        charts.append({
            "title": f"预测类别分布（目标列：{target}）",
            "type": "bar",
            "series": {"type": "bar", "labels": [str(l) for l in labels], "counts": vals,
                       "col": f"预测类别（{target}：0/1）", "count_label": "样本数（个）",
                       "hint": f"横轴 0/1 为模型对目标列“{target}”的预测类别取值；柱高=预测为该类别的样本数；特征列：{', '.join(map(str, feature_cols))}"},
        })

    # 11. 预览（前 20 行）
    preview_cols = list(result.columns)
    preview = result.head(20).astype(object).where(result.head(20).notna(), None)
    preview = preview.to_dict(orient="records")

    # 12. 模型解读报告
    top_feat = "，".join([f"{i['feature']}({i['importance']})" for i in importance[:5]]) or "无"
    metric_desc = "；".join([f"{k}={v}" for k, v in metrics.items()]) or "（样本量过少未评估）"
    pred_prompt = f"""
预测任务真实信息：
- 任务类型：{task}；算法：{algorithm}
- 样本数：{n}，特征数：{len(feature_names)}；目标列：{target}
- 类别数：{len(classes) if classes else '-'}
- 评估指标：{metric_desc}
- 训练说明：{split_note or '已按 8:2 划分训练/测试集'}
- 最重要的特征（Top5）：{top_feat}

要求：输出一份中文预测分析报告，说明模型效果如何、哪些特征对目标影响最大、当前预测的可靠程度，严格基于上述数字，禁止编造，控制在200字以内。
"""
    pred_res = call_ollama(pred_prompt, model_name=model_name, stream=False, timeout=300)
    report = pred_res.get("response", "") if "error" not in pred_res else "（模型报告生成失败，可查看下方统计与图表）"

    return {
        "code": 0,
        "msg": "预测分析完成",
        "data": {
            "report": report,
            "task": task,
            "algorithm": algorithm,
            "target": target,
            "n_samples": n,
            "n_features": len(feature_names),
            "n_classes": len(classes),
            "classes": classes[:20],
            "metrics": metrics,
            "split_note": split_note,
            "importance": importance,
            "preview_columns": preview_cols,
            "preview": preview,
            "predicted_file_path": pred_path,
            "predicted_filename": pred_name,
            "charts": charts,
            "model_name": model_name,
        },
    }

async def _predict_with_trained_model(df, model_file, target, model_name):
    """加载训练模块产出的 best_loss / last_loss 模型，对数据集做预测（不重新训练）。"""
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score

    # 1. 加载模型 bundle
    try:
        bundle = _load_trained_bundle(model_file)
    except Exception as e:
        return {"code": -1, "msg": str(e)}
    task = bundle.get("task_type", "regression")
    if task == "cluster":
        return {"code": -1, "msg": "聚类模型不用于目标预测，请训练/选择分类或回归模型"}
    algorithm = str(bundle.get("algorithm", "trained_model"))
    target = target or bundle.get("target_column", "")
    le = bundle.get("label_encoder")
    classes = [str(c) for c in le.classes_] if le is not None else []

    # 目标列类型与模型类型一致性检查（类型不匹配给出明确提示，避免崩溃）
    if target and target in df.columns:
        if task == "regression" and not pd.api.types.is_numeric_dtype(df[target]):
            return {"code": -1, "msg": f"所选模型是回归模型（{algorithm}），但目标列“{target}”不是数值类型（{df[target].dtype}，可能是分类标签）。请选择分类模型，或选择数值型目标列。"}
        if task == "classification" and pd.api.types.is_numeric_dtype(df[target]) and df[target].nunique() > 20:
            return {"code": -1, "msg": f"所选模型是分类模型（{algorithm}），但目标列“{target}”是连续数值（{df[target].nunique()} 个不同值）。请选择回归模型，或选择分类标签目标列。"}

    # 2. 按训练特征工程构建 X 并预测
    try:
        X, feature_names = _build_X_using_bundle(df, bundle)
        scaler = bundle.get("scaler")
        Xs = scaler.transform(X) if scaler is not None else X
        model = bundle["model"]
        y_pred_all = np.asarray(model.predict(Xs))
        y_scaler_b = bundle.get("y_scaler")
        if task == "regression" and y_scaler_b is not None:
            y_pred_all = y_scaler_b.inverse_transform(y_pred_all.reshape(-1, 1)).ravel()
    except Exception as e:
        return {"code": -1, "msg": f"模型预测失败: {str(e)}"}

    # 3. 指标：若数据集含目标列且有效样本足够，对比真实值
    metrics = {}
    split_note = ""
    y_true_arr = None
    if target and target in df.columns:
        mask = np.asarray(df[target].notna())
        y_true_arr = df[target][mask].values
        yp_arr = y_pred_all[mask]
        if len(y_true_arr) >= 5:
            if task == "classification" and le is not None:
                try:
                    metrics = {"accuracy": round(float(accuracy_score(le.transform(y_true_arr.astype(str)), yp_arr)), 4)}
                except Exception:
                    metrics = {}
            else:
                metrics = {
                    "r2": round(float(r2_score(y_true_arr, yp_arr)), 4),
                    "mae": round(float(mean_absolute_error(y_true_arr, yp_arr)), 4),
                    "rmse": round(float(np.sqrt(mean_squared_error(y_true_arr, yp_arr))), 4),
                }
        else:
            split_note = "目标列有效样本过少，未评估指标"
    else:
        split_note = "数据集中无目标列，仅输出预测结果"

    # 4. 特征重要性（线性模型 coef_ / 树模型 feature_importances_）
    importance = []
    try:
        if hasattr(model, "coef_"):
            coef = np.asarray(model.coef_)
            if coef.ndim == 2:
                coef = np.mean(np.abs(coef), axis=0)
            else:
                coef = np.abs(coef)
            imp = sorted(zip(feature_names, coef), key=lambda t: t[1], reverse=True)
            importance = [{"feature": f, "importance": round(float(v), 4)} for f, v in imp[:15]]
        elif hasattr(model, "feature_importances_"):
            imp = sorted(zip(feature_names, model.feature_importances_), key=lambda t: t[1], reverse=True)
            importance = [{"feature": f, "importance": round(float(v), 4)} for f, v in imp[:15]]
    except Exception:
        pass

    # 5. 结果表
    result = df.copy()
    if task == "classification" and le is not None:
        result["预测值"] = [classes[int(i)] if 0 <= int(i) < len(classes) else str(i) for i in y_pred_all]
    else:
        result["预测值"] = [round(float(v), 4) for v in y_pred_all]
    ext = os.path.splitext(os.path.basename(getattr(df, "_src_path", "")))[0] or "data"
    pred_name = f"{ext}_predict_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    pred_path = os.path.abspath(os.path.join(cleaned_folder, pred_name))
    try:
        result.to_csv(pred_path, index=False, encoding="utf-8-sig")
    except Exception as e:
        return {"code": -1, "msg": f"保存预测结果失败: {str(e)}"}

    # 6. 前端渲染图表数据（Chart.js 交互式）：大模型分析已训练模型后给出不同的图像呈现方式，后端确定性绘制
    #    SGD 回归模型输出四图：真实值vs预测值、残差分布直方图、学习曲线、残差-预测值散点图
    charts = []
    is_sgd_reg = algorithm == "linear_regression" and task == "regression"
    has_true = y_true_arr is not None and len(y_true_arr) >= 5
    if task == "classification" and le is not None:
        cnt = Counter(classes[int(i)] for i in y_pred_all)
        labels = list(cnt.keys()); vals = [cnt[k] for k in labels]
        charts.append({
            "title": f"已训练模型（{algorithm}）预测类别分布（目标列：{target}）",
            "type": "bar",
            "series": {"type": "bar", "labels": [str(l) for l in labels], "counts": vals,
                       "col": f"预测类别（{target}：0/1）", "count_label": "样本数（个）",
                       "hint": f"横轴 0/1 为模型对目标列“{target}”的预测类别取值；柱高=预测为该类别的样本数；模型特征：{', '.join(map(str, feature_names[:8]))}{'…' if len(feature_names) > 8 else ''}"},
        })
        # 分类模型诊断：混淆矩阵、ROC 曲线 + AUC、特征重要性 + 学习曲线（SGD 分类）
        if has_true:
            y_enc = le.transform(y_true_arr.astype(str))
            y_pred_enc = np.asarray(y_pred_all[mask], dtype=int)
            # 1) 混淆矩阵
            try:
                from sklearn.metrics import confusion_matrix
                cm = confusion_matrix(y_enc, y_pred_enc, labels=list(range(len(classes))))
                mx = max([int(v) for row in cm for v in row] + [1])
                charts.append({
                    "title": f"混淆矩阵（{algorithm}，目标列：{target}）",
                    "type": "confusion",
                    "confusion": {"labels": [str(c) for c in classes],
                                  "matrix": [[int(v) for v in row] for row in cm],
                                  "pred_label": "预测类别", "true_label": "真实类别",
                                  "max": mx,
                                  "hint": f"行=真实类别，列=预测类别；对角线为预测正确的样本数。数值越大单元格颜色越深；混淆集中在对角线说明分类准确。"},
                })
            except Exception:
                pass
            # 2) ROC 曲线 + AUC（需 predict_proba；二分类直接画，多类按 OvR 逐类画并插值对齐）
            try:
                from sklearn.metrics import roc_curve, roc_auc_score
                if hasattr(model, "predict_proba"):
                    proba = np.asarray(model.predict_proba(Xs[mask]))
                    if len(classes) == 2:
                        fpr, tpr, _ = roc_curve(y_enc, proba[:, 1])
                        auc = float(roc_auc_score(y_enc, proba[:, 1]))
                        charts.append({
                            "title": f"ROC 曲线 + AUC（{algorithm}，AUC={auc:.3f}）",
                            "type": "line",
                            "series": {"type": "line", "roc": True,
                                       "labels": [round(float(v), 4) for v in fpr],
                                       "datasets": [{"label": f"ROC 曲线（AUC={auc:.3f}）", "data": [round(float(v), 4) for v in tpr], "color": "#3B82F6"}],
                                       "diag": [0.0, 1.0],
                                       "x_label": "假阳性率（FPR）", "y_label": "真阳性率（TPR）",
                                       "hint": "ROC 曲线越贴近左上角，分类性能越好；AUC=0.5 表示随机猜测，越接近 1 越优。横轴 FPR=误判为真的比例，纵轴 TPR=正确识别真的比例。"},
                        })
                    else:
                        grid = np.linspace(0, 1, 80)
                        dss = []
                        colors = ["#3B82F6", "#EC4899", "#F59E0B", "#10B981", "#8B5CF6", "#06B6D4"]
                        macro = []
                        for ci, cname in enumerate(classes):
                            yb = (y_enc == ci).astype(int)
                            fpr, tpr, _ = roc_curve(yb, proba[:, ci])
                            a = float(roc_auc_score(yb, proba[:, ci])); macro.append(a)
                            tpr_i = np.interp(grid, fpr, tpr)
                            dss.append({"label": f"类 {cname}（AUC={a:.3f}）", "data": [round(float(v), 4) for v in tpr_i], "color": colors[ci % len(colors)]})
                        ma = float(np.mean(macro))
                        charts.append({
                            "title": f"ROC 曲线 + AUC（{algorithm}，macro AUC={ma:.3f}）",
                            "type": "line",
                            "series": {"type": "line", "roc": True,
                                       "labels": [round(float(v), 3) for v in grid],
                                       "datasets": dss, "diag": [0.0, 1.0],
                                       "x_label": "假阳性率（FPR）", "y_label": "真阳性率（TPR）",
                                       "hint": "每个类别一条 ROC 曲线（一对多 OvR）；macro AUC 为各类 AUC 的均值，越接近 1 分类性能越好。"},
                        })
            except Exception:
                pass
            # 3) 特征重要性
            if importance:
                charts.append({
                    "title": f"特征重要性（{algorithm}，|系数|）",
                    "type": "bar",
                    "series": {"type": "bar", "labels": [i["feature"] for i in importance[:12]],
                               "counts": [i["importance"] for i in importance[:12]],
                               "col": "特征", "count_label": "重要性（|系数|）",
                               "hint": "柱高为模型系数绝对值（多类取各类均值），值越大该特征对分类结果的影响越大。"},
                })
            # 4) 学习曲线（SGD 分类）
            if algorithm == "logistic_regression":
                try:
                    lc = _build_learning_curve(bundle, X[mask], y_enc, "classification")
                    if lc: charts.append(lc)
                except Exception:
                    pass
    elif has_true:
        mask = np.asarray(df[target].notna())
        yp = y_pred_all[mask]
        # 图1：真实值 vs 预测值
        pts = [[round(float(a), 4), round(float(p), 4)] for a, p in zip(y_true_arr, yp)]
        if len(pts) > 1500:
            step = (len(pts) + 1499) // 1500
            pts = pts[::step]
        charts.append({
            "title": f"已训练模型（{algorithm}）实际 vs 预测（目标列：{target}）",
            "type": "scatter",
            "series": {"type": "scatter", "points": pts, "x_label": "实际值", "y_label": "预测值", "col": target,
                   "hint": f"目标列：{target}；模型特征：{', '.join(map(str, feature_names[:8]))}{'…' if len(feature_names) > 8 else ''}；每个点代表一个样本：横轴为实际值，纵轴为模型预测值。点越贴近 y=x 对角线，预测越准。"},
        })
        if is_sgd_reg:
            resid = np.asarray(y_true_arr, dtype=float) - np.asarray(yp, dtype=float)
            # 图2：残差分布直方图
            hc, be = np.histogram(resid, bins=20)
            hl = [f"{be[i]:.2g}~{be[i + 1]:.2g}" for i in range(len(be) - 1)]
            charts.append({
                "title": f"残差分布直方图（{algorithm}，目标列：{target}）",
                "type": "hist",
                "series": {"type": "hist", "labels": hl, "counts": [int(v) for v in hc],
                       "col": "残差区间", "count_label": "样本数（个）",
                       "hint": "残差 = 真实值 − 预测值；柱高=残差落在该区间的样本数。残差集中于 0 附近且近似对称，说明模型误差小、无明显系统性偏差。"},
            })
            # 图3：学习曲线
            try:
                lc = _build_learning_curve(bundle, X[mask], y_true_arr.astype(float), "regression")
                if lc: charts.append(lc)
            except Exception:
                pass
            # 图4：残差-预测值散点图
            rpts = [[round(float(p), 4), round(float(r), 4)] for p, r in zip(yp, resid)]
            if len(rpts) > 1500:
                step = (len(rpts) + 1499) // 1500
                rpts = rpts[::step]
            charts.append({
                "title": f"残差-预测值散点图（{algorithm}，目标列：{target}）",
                "type": "scatter",
                "series": {"type": "scatter", "points": rpts, "x_label": "预测值", "y_label": "残差", "col": target,
                       "hint": "横轴为模型预测值，纵轴为残差（真实−预测）。散点随机分布在 y=0 两侧且无明显喇叭形，说明方差齐性良好；若呈漏斗形扩散说明存在异方差。"},
            })
    else:
        hist_counts, bin_edges = np.histogram([float(v) for v in y_pred_all], bins=20)
        hlabels = [f"{bin_edges[i]:.2g}~{bin_edges[i + 1]:.2g}" for i in range(len(bin_edges) - 1)]
        charts.append({
            "title": f"已训练模型（{algorithm}）预测值分布（目标列：{target}）",
            "type": "hist",
            "series": {"type": "hist", "labels": hlabels, "counts": [int(v) for v in hist_counts],
                   "col": "预测值区间", "count_label": "频数（个）",
                   "hint": f"目标列：{target}；柱高表示预测值落在该数值区间的样本数量"},
        })

    # 7. 预览
    preview_cols = list(result.columns)
    preview = result.head(20).astype(object).where(result.head(20).notna(), None).to_dict(orient="records")

    # 8. 大模型报告
    top_feat = "，".join([f"{i['feature']}({i['importance']})" for i in importance[:5]]) or "无"
    metric_desc = "；".join([f"{k}={v}" for k, v in metrics.items()]) or "（未评估）"
    chart_titles = "；".join(c.get("title", "") for c in charts) or "无"
    pred_prompt = f"""
预测任务真实信息（使用训练模块产出的已训练模型）：
- 算法：{algorithm}；任务类型：{task}
- 模型文件：{model_file}
- 样本数：{len(df)}；特征数：{len(feature_names)}；目标列：{target}
- 评估指标：{metric_desc}
- 说明：{split_note or "已训练模型直接预测，未重新训练"}
- 最重要的特征（Top5）：{top_feat}
- 本次已展示图像：{chart_titles}

要求：输出一份中文预测分析报告，说明该已训练模型的预测效果、对目标影响最大的特征、预测的可靠程度，严格基于上述数字，禁止编造，200字以内。
报告末尾另起一段，以「【图像呈现说明】」为标题，简要说明本次展示了哪些图像、每张图应如何解读（例如：残差是否集中于0、学习曲线是否收敛、散点是否随机分布、预测值与实际值是否贴近对角线），能从中发现什么问题，150字以内。
"""
    try:
        pred_res = call_ollama(pred_prompt, model_name=model_name, stream=False, timeout=300)
        report = pred_res.get("response", "") if "error" not in pred_res else "（模型报告生成失败，可查看下方统计与图表）"
    except Exception:
        report = "（模型报告生成失败，可查看下方统计与图表）"

    return {
        "code": 0,
        "msg": "预测分析完成（使用已训练模型）",
        "data": {
            "report": report,
            "task": task,
            "algorithm": algorithm,
            "target": target,
            "n_samples": len(df),
            "n_features": len(feature_names),
            "n_classes": len(classes),
            "classes": classes[:20],
            "metrics": metrics,
            "split_note": split_note,
            "importance": importance,
            "preview_columns": preview_cols,
            "preview": preview,
            "predicted_file_path": pred_path,
            "predicted_filename": pred_name,
            "charts": charts,
            "model_name": model_name,
            "used_model_file": model_file,
        },
    }

# ================= 加权投票集成（软投票） =================
@app.post("/upload_model")
async def upload_model(file: UploadFile = File(...)):
    """上传分类模型文件（.joblib）到 models 目录，供加权投票集成使用。"""
    import re as _re
    name = os.path.basename(file.filename or "")
    safe = _re.sub(r'[^a-zA-Z0-9._-]', '_', name)
    if not safe.lower().endswith(".joblib"):
        return {"code": -1, "msg": "仅支持 .joblib 模型文件"}
    os.makedirs(model_folder, exist_ok=True)
    save_path = os.path.abspath(os.path.join(model_folder, safe))
    if os.path.exists(save_path):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base, ext = os.path.splitext(safe)
        safe = f"{base}_{ts}{ext}"
        save_path = os.path.abspath(os.path.join(model_folder, safe))
    with open(save_path, "wb") as f:
        f.write(await file.read())
    try:
        b = joblib.load(save_path)
        if "model" not in b:
            os.remove(save_path)
            return {"code": -1, "msg": "文件不是有效的模型 bundle"}
        task = str(b.get("task_type", "unknown"))
        algo = str(b.get("algorithm", "unknown"))
    except Exception as e:
        try:
            os.remove(save_path)
        except Exception:
            pass
        return {"code": -1, "msg": f"模型解析失败: {e}"}
    return {"code": 0, "msg": "模型上传成功", "data": {"filename": safe, "task_type": task, "algorithm": algo}}

def _model_classes(bundle):
    """返回模型 bundle 的类别名列表（由 label_encoder 还原）。"""
    le = bundle.get("label_encoder")
    return [str(c) for c in le.classes_] if le is not None else []

def _bundle_transform(df, bundle):
    """按 bundle 特征工程构建 X 并做标准化（与训练时一致）。"""
    X, _ = _build_X_using_bundle(df, bundle)
    scaler = bundle.get("scaler")
    return scaler.transform(X) if scaler is not None else X

def _macro_f1_on_df(bundle, df, target):
    """计算单个模型在数据集上的 Macro-F1（自动过滤模型未见过的类别）。返回 (f1, 有效样本数)。"""
    from sklearn.metrics import f1_score
    classes = _model_classes(bundle)
    le = bundle.get("label_encoder")
    if not classes or le is None:
        return 0.0, 0
    mask = df[target].notna()
    y_true = df[target][mask].astype(str).values
    try:
        X = _bundle_transform(df[mask], bundle)
        y_pred = np.asarray(bundle["model"].predict(X))
    except Exception:
        return 0.0, 0
    keep = np.array([t in classes for t in y_true])
    if keep.sum() == 0:
        return 0.0, 0
    yt = le.transform(y_true[keep])
    yp = y_pred[keep].astype(int)
    ok = (yp >= 0) & (yp < len(classes))
    if ok.sum() == 0:
        return 0.0, 0
    f1 = f1_score(yt[ok], yp[ok], average="macro", labels=list(range(len(classes))), zero_division=0)
    return round(float(f1), 4), int(ok.sum())

@app.post("/ensemble_predict")
async def ensemble_predict(
    n: int = Query(),
    model_files: str = Query(),
    val_file: str = Query(default=""),
    val_target: str = Query(default=""),
    pred_file: str = Query(),
    pred_target: str = Query(default=""),
    features: str = Query(default=""),
    model_name: str = Query(default=DEFAULT_MODEL),
):
    """分类模型加权投票集成（软投票）：
    1) 加载 n 个分类模型 bundle；
    2) 若提供验证集：计算各模型 Macro-F1，权重 = F1_i / ΣF1_j（线性归一化），否则等权；
    3) 对预测数据各模型输出类别概率，按权重加权平均 → argmax 得集成预测；
    4) 若预测数据含目标列，输出集成 Macro-F1 / Accuracy。"""
    from sklearn.metrics import f1_score, accuracy_score
    try:
        if n < 1 or n > 20:
            return {"code": -1, "msg": "n 值需在 1~20 之间"}
        files = [f.strip() for f in (model_files or "").split(",") if f.strip()]
        if not files:
            return {"code": -1, "msg": "请先上传分类模型"}
        if len(files) < n:
            return {"code": -1, "msg": f"已上传模型数（{len(files)}）少于 n 值（{n}）"}
        files = files[:n]
        bundles = []
        for fn in files:
            try:
                b = _load_trained_bundle(fn)
            except Exception as e:
                return {"code": -1, "msg": f"模型 {fn} 加载失败: {e}"}
            if b.get("task_type") != "classification":
                return {"code": -1, "msg": f"模型 {fn} 不是分类模型（task_type={b.get('task_type')}），软投票集成仅支持分类模型"}
            bundles.append(b)
    except Exception as e:
        return {"code": -1, "msg": f"参数错误: {e}"}

    # ---- 1. 权重：验证集 Macro-F1 线性归一化 ----
    f1_scores, f1_note = [], ""
    if val_file and val_target:
        try:
            vdf = load_df_checked(val_file)
            if val_target not in vdf.columns:
                return {"code": -1, "msg": f"验证集目标列不存在: {val_target}（可用列：{list(vdf.columns)}）"}
            for b in bundles:
                f1, _n = _macro_f1_on_df(b, vdf, val_target)
                f1_scores.append(f1)
            if not any(s > 0 for s in f1_scores):
                f1_scores = []
            else:
                f1_note = "基于验证集 Macro-F1 线性归一化"
        except Exception as e:
            return {"code": -1, "msg": f"验证集计算失败: {e}"}
    if f1_scores:
        ssum = sum(f1_scores)
        weights = [round(s / ssum, 6) for s in f1_scores]
    else:
        weights = [round(1.0 / len(bundles), 6)] * len(bundles)
        f1_note = f1_note or "未提供验证集，采用等权"

    # ---- 2. 预测数据 ----
    try:
        pdf = load_df_checked(pred_file)
    except Exception as e:
        return {"code": -1, "msg": str(e)}
    if len(pdf) == 0:
        return {"code": -1, "msg": "预测数据集为空"}

    # 类别空间：所有模型类别的并集（保序）
    all_classes = []
    for b in bundles:
        for c in _model_classes(b):
            if c not in all_classes:
                all_classes.append(c)
    if not all_classes:
        return {"code": -1, "msg": "模型缺少类别信息（label_encoder 缺失）"}

    probas, preds_each = [], []
    try:
        for b in bundles:
            Xp = _bundle_transform(pdf, b)
            model = b["model"]
            if not hasattr(model, "predict_proba"):
                return {"code": -1, "msg": f"模型 {b.get('algorithm')} 不支持概率输出（predict_proba），无法软投票"}
            proba = np.asarray(model.predict_proba(Xp))
            cls = _model_classes(b)
            P = np.zeros((len(pdf), len(all_classes)))
            for i, c in enumerate(cls):
                if c in all_classes:
                    P[:, all_classes.index(c)] = proba[:, i]
            probas.append(P)
            preds_each.append([all_classes[int(np.argmax(row))] for row in P])
    except Exception as e:
        return {"code": -1, "msg": f"模型预测失败: {e}"}

    P_ens = np.zeros_like(probas[0])
    for w, P in zip(weights, probas):
        P_ens += w * P
    y_ens = [all_classes[int(np.argmax(row))] for row in P_ens]
    conf = [round(float(max(row)), 4) for row in P_ens]

    # ---- 3. 集成指标（若预测数据含真实目标列） ----
    metrics = {}
    ens_f1, ens_acc, yt, yp = None, None, None, None
    if pred_target and pred_target in pdf.columns:
        y_true = pdf[pred_target].astype(str).values
        keep = np.array([t in all_classes for t in y_true])
        if keep.sum() >= 5:
            yt = np.array([all_classes.index(t) for t in y_true[keep]])
            yp = np.array([all_classes.index(p) for p in np.asarray(y_ens)[keep]])
            ens_f1 = round(float(f1_score(yt, yp, average="macro", labels=list(range(len(all_classes))), zero_division=0)), 4)
            ens_acc = round(float(accuracy_score(yt, yp)), 4)
            metrics = {"macro_f1": ens_f1, "accuracy": ens_acc}

    # 各模型信息（验证集 F1 优先；否则若预测数据含目标列则用预测集 F1）
    per_model = []
    for i, b in enumerate(bundles):
        if i < len(f1_scores):
            mf1, _n = f1_scores[i], 0
        elif pred_target and pred_target in pdf.columns:
            mf1, _n = _macro_f1_on_df(b, pdf, pred_target)
        else:
            mf1 = 0.0
        per_model.append({
            "filename": files[i],
            "algorithm": str(b.get("algorithm", "unknown")),
            "classes": _model_classes(b),
            "macro_f1": mf1,
            "weight": weights[i],
        })

    # ---- 4. 结果表 ----
    result = pdf.copy()
    result["集成预测"] = y_ens
    result["集成置信度"] = conf
    for i, fn in enumerate(files):
        result[f"模型{i + 1}_{os.path.splitext(os.path.basename(fn))[0]}"] = preds_each[i]
    base_name = os.path.splitext(os.path.basename(pred_file))[0]
    ens_name = f"{base_name}_ensemble_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    ens_path = os.path.abspath(os.path.join(cleaned_folder, ens_name))
    try:
        result.to_csv(ens_path, index=False, encoding="utf-8-sig")
    except Exception as e:
        return {"code": -1, "msg": f"保存集成预测结果失败: {e}"}

    # ---- 5. 图表 ----
    charts = []
    charts.append({
        "title": f"模型集成权重（{f1_note}）",
        "type": "bar",
        "series": {"type": "bar",
                   "labels": [f"M{i + 1} {os.path.splitext(os.path.basename(f))[0][:16]}" for i, f in enumerate(files)],
                   "counts": weights, "col": "模型", "count_label": "权重",
                   "hint": f"权重由 Macro-F1 线性归一化得到（w_i = F1_i / ΣF1_j）；{f1_note}。权重越大，该模型在软投票中的话语权越高。"},
    })
    cnt = Counter(y_ens)
    labels = list(cnt.keys()); vals = [cnt[k] for k in labels]
    charts.append({
        "title": f"集成预测类别分布（n={n}）",
        "type": "bar",
        "series": {"type": "bar", "labels": [str(l) for l in labels], "counts": vals,
                   "col": "集成预测类别", "count_label": "样本数（个）",
                   "hint": f"软投票集成：n={n} 个模型预测概率按权重加权平均后取最大概率类别。"},
    })
    if ens_f1 is not None:
        try:
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(yt, yp, labels=list(range(len(all_classes))))
            mx = max([int(v) for row in cm for v in row] + [1])
            charts.append({
                "title": f"集成模型混淆矩阵（Macro-F1={ens_f1}）",
                "type": "confusion",
                "confusion": {"labels": [str(c) for c in all_classes],
                              "matrix": [[int(v) for v in row] for row in cm],
                              "pred_label": "集成预测类别", "true_label": "真实类别", "max": mx,
                              "hint": "行=真实类别，列=集成预测类别；对角线为预测正确的样本数。数值越大颜色越深。"},
            })
        except Exception:
            pass

    # ---- 6. 预览 ----
    preview_cols = list(result.columns)
    preview = result.head(20).astype(object).where(result.head(20).notna(), None).to_dict(orient="records")

    # ---- 7. 大模型总结（失败不影响主流程） ----
    report = ""
    try:
        model_lines = "；".join([f"{os.path.basename(m['filename'])}(F1={m['macro_f1']},权重={m['weight']})" for m in per_model])
        _metrics_str = "；".join([f"{_k}={_v}" for _k, _v in metrics.items()]) if metrics else "（预测数据未含目标列，未评估）"
        prompt_ens = (
            "加权投票集成预测任务真实信息：\n"
            "- n={n}，模型：{model_lines}\n"
            "- 权重方式：{f1_note}\n"
            "- 集成指标：{_metrics_str}\n"
            "- 预测样本数：{_n}，类别数：{_nc}，类别：{_classes}\n\n"
            "要求：输出中文集成预测分析报告，说明各模型权重是否合理、集成效果如何，严格基于上述数字，禁止编造，200字以内。"
        ).format(n=n, model_lines=model_lines, f1_note=f1_note, _metrics_str=_metrics_str,
                 _n=len(pdf), _nc=len(all_classes), _classes=all_classes)
        pr = call_ollama(prompt_ens,
            model_name=model_name, stream=False, timeout=120)
        report = pr.get("response", "") if "error" not in pr else "（模型报告生成失败）"
    except Exception:
        report = "（模型报告生成失败）"

    return {
        "code": 0,
        "msg": "加权投票集成预测完成",
        "data": {
            "report": report,
            "n": n,
            "models": per_model,
            "weights": weights,
            "ensemble_metrics": metrics,
            "weight_note": f1_note,
            "classes": all_classes,
            "n_samples": len(pdf),
            "preview_columns": preview_cols,
            "preview": preview,
            "predicted_file_path": ens_path,
            "predicted_filename": ens_name,
            "charts": charts,
        },
    }

def _rmse_on_df(bundle, df, target):
    """计算单个回归模型在数据集上的 RMSE。返回 (rmse, 有效样本数) 或 (None, 0)。"""
    if target not in df.columns:
        return None, 0
    mask = df[target].notna()
    y_true = pd.to_numeric(df[target][mask], errors="coerce")
    ok = y_true.notna()
    if int(ok.sum()) < 3:
        return None, 0
    try:
        X = _bundle_transform(df[mask][ok], bundle)
        y_pred = np.asarray(bundle["model"].predict(X)).astype(float)
        ys_b = bundle.get("y_scaler")
        if ys_b is not None:
            y_pred = ys_b.inverse_transform(y_pred.reshape(-1, 1)).ravel()
    except Exception:
        return None, 0
    yt = y_true[ok].values.astype(float)
    rmse = float(np.sqrt(np.mean((yt - y_pred) ** 2)))
    return rmse, int(ok.sum())

@app.post("/ensemble_predict_reg")
async def ensemble_predict_reg(
    n: int = Query(),
    model_files: str = Query(),
    val_file: str = Query(default=""),
    val_target: str = Query(default=""),
    pred_file: str = Query(),
    pred_target: str = Query(default=""),
    model_name: str = Query(default=DEFAULT_MODEL),
):
    """回归模型加权平均集成：
    1) 加载 n 个回归模型 bundle；
    2) 若提供验证集：计算各模型 RMSE，权重 = (1/RMSE_i) / Σ(1/RMSE_j)（误差倒数归一化），否则等权；
    3) 对预测数据各模型输出预测值，按权重加权平均得集成预测；
    4) 若预测数据含目标列，输出集成 RMSE / MAE / R²。"""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    try:
        if n < 1 or n > 20:
            return {"code": -1, "msg": "n 值需在 1~20 之间"}
        files = [f.strip() for f in (model_files or "").split(",") if f.strip()]
        if not files:
            return {"code": -1, "msg": "请先上传回归模型"}
        if len(files) < n:
            return {"code": -1, "msg": f"已上传模型数（{len(files)}）少于 n 值（{n}）"}
        files = files[:n]
        bundles = []
        for fn in files:
            try:
                b = _load_trained_bundle(fn)
            except Exception as e:
                return {"code": -1, "msg": f"模型 {fn} 加载失败: {e}"}
            if b.get("task_type") != "regression":
                return {"code": -1, "msg": f"模型 {fn} 不是回归模型（task_type={b.get('task_type')}），加权平均集成仅支持回归模型"}
            bundles.append(b)
    except Exception as e:
        return {"code": -1, "msg": f"参数错误: {e}"}

    # ---- 1. 权重：验证集 RMSE 误差倒数归一化 ----
    rmse_scores, w_note = [], ""
    if val_file and val_target:
        try:
            vdf = load_df_checked(val_file)
            if val_target not in vdf.columns:
                return {"code": -1, "msg": f"验证集目标列不存在: {val_target}（可用列：{list(vdf.columns)}）"}
            for b in bundles:
                rmse, _n = _rmse_on_df(b, vdf, val_target)
                rmse_scores.append(rmse)
            if not any(r is not None for r in rmse_scores):
                rmse_scores = []
            else:
                w_note = "基于验证集 RMSE 误差倒数归一化"
        except Exception as e:
            return {"code": -1, "msg": f"验证集计算失败: {e}"}
    if rmse_scores:
        inv = []
        for r in rmse_scores:
            if r is None or r <= 0:
                inv.append(0.0)
            elif r <= 1e-12:
                inv.append(1e9)
            else:
                inv.append(1.0 / r)
        isum = sum(inv)
        if isum > 0:
            weights = [round(v / isum, 6) for v in inv]
        else:
            weights = [round(1.0 / len(bundles), 6)] * len(bundles)
            w_note = "验证集无法评估，采用等权"
    else:
        weights = [round(1.0 / len(bundles), 6)] * len(bundles)
        w_note = w_note or "未提供验证集，采用等权"

    # ---- 2. 预测数据 ----
    try:
        pdf = load_df_checked(pred_file)
    except Exception as e:
        return {"code": -1, "msg": str(e)}
    if len(pdf) == 0:
        return {"code": -1, "msg": "预测数据集为空"}

    preds_each = []
    try:
        for b in bundles:
            Xp = _bundle_transform(pdf, b)
            yp = np.asarray(b["model"].predict(Xp)).astype(float)
            ys_b = b.get("y_scaler")
            if ys_b is not None:
                yp = ys_b.inverse_transform(yp.reshape(-1, 1)).ravel()
            preds_each.append(yp)
    except Exception as e:
        return {"code": -1, "msg": f"模型预测失败: {e}"}

    y_ens = np.zeros(len(pdf))
    for w, yp in zip(weights, preds_each):
        y_ens += w * yp

    # ---- 3. 集成指标（若预测数据含真实目标列） ----
    metrics = {}
    yt_all, yp_all = None, None
    if pred_target and pred_target in pdf.columns:
        y_true = pd.to_numeric(pdf[pred_target], errors="coerce")
        m = y_true.notna().values
        if m.sum() >= 5:
            yt_all = y_true[m].values.astype(float)
            yp_all = y_ens[m]
            metrics = {
                "rmse": round(float(np.sqrt(mean_squared_error(yt_all, yp_all))), 4),
                "mae": round(float(mean_absolute_error(yt_all, yp_all)), 4),
                "r2": round(float(r2_score(yt_all, yp_all)), 4),
            }

    # 各模型信息（验证集 RMSE 优先；否则若预测数据含目标列则用预测集 RMSE）
    per_model = []
    for i, b in enumerate(bundles):
        if i < len(rmse_scores):
            mrmse, _n = rmse_scores[i], 0
        elif pred_target and pred_target in pdf.columns:
            mrmse, _n = _rmse_on_df(b, pdf, pred_target)
        else:
            mrmse = None
        per_model.append({
            "filename": files[i],
            "algorithm": str(b.get("algorithm", "unknown")),
            "rmse": mrmse,
            "weight": weights[i],
        })

    # ---- 4. 结果表 ----
    result = pdf.copy()
    result["集成预测"] = [round(float(v), 6) for v in y_ens]
    for i, fn in enumerate(files):
        result[f"模型{i + 1}_{os.path.splitext(os.path.basename(fn))[0]}"] = [round(float(v), 6) for v in preds_each[i]]
    base_name = os.path.splitext(os.path.basename(pred_file))[0]
    ens_name = f"{base_name}_ensemble_reg_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    ens_path = os.path.abspath(os.path.join(cleaned_folder, ens_name))
    try:
        result.to_csv(ens_path, index=False, encoding="utf-8-sig")
    except Exception as e:
        return {"code": -1, "msg": f"保存集成预测结果失败: {e}"}

    # ---- 5. 图表 ----
    charts = []
    charts.append({
        "title": f"模型集成权重（{w_note}）",
        "type": "bar",
        "series": {"type": "bar",
                   "labels": [f"M{i + 1} {os.path.splitext(os.path.basename(f))[0][:16]}" for i, f in enumerate(files)],
                   "counts": weights, "col": "模型", "count_label": "权重",
                   "hint": f"权重由 RMSE 误差倒数归一化得到（w_i = (1/RMSE_i) / Σ(1/RMSE_j)）；{w_note}。RMSE 越小权重越大，误差低的模型话语权更高。"},
    })
    if yt_all is not None:
        pts = [[round(float(x), 4), round(float(y), 4)] for x, y in zip(yt_all[:500], yp_all[:500])]
        charts.append({
            "title": "真实值 vs 集成预测值（散点）",
            "type": "scatter",
            "series": {"type": "scatter", "points": pts, "x_label": "真实值", "y_label": "集成预测值",
                       "hint": "点越贴近对角线 y=x，预测越准确；集成 RMSE=" + str(metrics["rmse"]) + "。"},
        })
        resid = (yt_all - yp_all)[:500]
        try:
            hist, edges = np.histogram(resid, bins=min(20, max(5, int(np.sqrt(len(resid))))))
            labels = [f"{edges[i]:.2g}~{edges[i + 1]:.2g}" for i in range(len(hist))]
            charts.append({
                "title": "残差分布直方图",
                "type": "bar",
                "series": {"type": "bar", "labels": labels, "counts": [int(v) for v in hist],
                           "col": "残差区间", "count_label": "样本数（个）",
                           "hint": "残差 = 真实值 - 集成预测值；分布越集中在 0 附近且对称，模型越稳定。"},
            })
        except Exception:
            pass
        pts2 = [[round(float(y), 4), round(float(r), 4)] for y, r in zip(yp_all[:500], resid)]
        charts.append({
            "title": "残差 - 集成预测值（散点）",
            "type": "scatter",
            "series": {"type": "scatter", "points": pts2, "x_label": "集成预测值", "y_label": "残差",
                       "hint": "残差应随机分布在 0 附近、无明显趋势；若呈喇叭状说明存在异方差。"},
        })
    else:
        try:
            hist, edges = np.histogram(y_ens, bins=min(20, max(5, int(np.sqrt(len(y_ens))))))
            labels = [f"{edges[i]:.2g}~{edges[i + 1]:.2g}" for i in range(len(hist))]
            charts.append({
                "title": "集成预测值分布直方图",
                "type": "bar",
                "series": {"type": "bar", "labels": labels, "counts": [int(v) for v in hist],
                           "col": "预测值区间", "count_label": "样本数（个）",
                           "hint": "预测值分布，用于观察整体预测结果的范围与集中趋势。"},
            })
        except Exception:
            pass

    # ---- 6. 预览 ----
    preview_cols = list(result.columns)
    preview = result.head(20).astype(object).where(result.head(20).notna(), None).to_dict(orient="records")

    # ---- 7. 大模型总结（失败不影响主流程） ----
    report = ""
    try:
        model_lines = "；".join([f"{os.path.basename(m['filename'])}(RMSE={m['rmse'] if m['rmse'] is not None else '-'},权重={m['weight']})" for m in per_model])
        _metrics_str = "；".join([f"{_k}={_v}" for _k, _v in metrics.items()]) if metrics else "（预测数据未含目标列，未评估）"
        prompt_ens = (
            "回归模型加权平均集成预测任务真实信息：\n"
            "- n={n}，模型：{model_lines}\n"
            "- 权重方式：{w_note}\n"
            "- 集成指标：{_metrics_str}\n"
            "- 预测样本数：{_n}\n\n"
            "要求：输出中文集成预测分析报告，说明各模型权重是否合理、集成效果如何，严格基于上述数字，禁止编造，200字以内。"
        ).format(n=n, model_lines=model_lines, w_note=w_note, _metrics_str=_metrics_str, _n=len(pdf))
        pr = call_ollama(prompt_ens, model_name=model_name, stream=False, timeout=120)
        report = pr.get("response", "") if "error" not in pr else "（模型报告生成失败）"
    except Exception:
        report = "（模型报告生成失败）"

    return {
        "code": 0,
        "msg": "回归加权平均集成预测完成",
        "data": {
            "report": report,
            "n": n,
            "models": per_model,
            "weights": weights,
            "ensemble_metrics": metrics,
            "weight_note": w_note,
            "n_samples": len(pdf),
            "preview_columns": preview_cols,
            "preview": preview,
            "predicted_file_path": ens_path,
            "predicted_filename": ens_name,
            "charts": charts,
        },
    }
