# -*- coding: utf-8 -*-
"""模型训练模块：数据集切分、大模型推荐算法、模型代码生成、超参数训练与 loss 曲线。
训练完成后自动保存 best_loss / last_loss 两个模型文件，供下载与预测分析模块复用。"""
import os
import json
import copy
import warnings
import threading
import time
import uuid

import joblib
import pandas as pd
import numpy as np
from fastapi import Query
from fastapi.responses import FileResponse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import SGDRegressor, SGDClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.cluster import KMeans
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    silhouette_score,
)

warnings.filterwarnings("ignore")

# 实时训练任务表：task_id -> {"stop": Event, "state": {...}}
TRAIN_TASKS = {}
TRAIN_TASKS_LOCK = threading.Lock()

from config import app, split_folder, model_folder
from utils import load_df_checked

# 算法清单：类型与是否支持逐 epoch loss 曲线
TRAIN_ALGOS = {
    "linear_regression":    {"name": "线性回归 (SGD)",      "type": "regression",     "loss": True},
    "logistic_regression":  {"name": "逻辑回归 (SGD)",      "type": "classification", "loss": True},
    "mlp_regressor":        {"name": "MLP 神经网络-回归",   "type": "regression",     "loss": True},
    "mlp_classifier":       {"name": "MLP 神经网络-分类",   "type": "classification", "loss": True},
    "random_forest":        {"name": "随机森林",             "type": "auto",           "loss": False},
    "decision_tree":        {"name": "决策树",               "type": "auto",           "loss": False},
    "knn":                  {"name": "K 近邻",              "type": "auto",           "loss": False},
    "svm":                  {"name": "支持向量机",           "type": "auto",           "loss": False},
    "kmeans":               {"name": "KMeans 聚类",          "type": "cluster",        "loss": False}
}

def _sleep_for(epochs):
    """训练节奏控制：让 loss 曲线逐步可见且可在中途停止（训练全程约 6 秒，曲线逐点生成）。"""
    return min(0.06, max(0.02, 6.0 / max(int(epochs), 1)))

def _pub_progress(task_id, epoch, train_loss, val_loss, logs):
    task = TRAIN_TASKS.get(task_id)
    if task:
        with TRAIN_TASKS_LOCK:
            task["state"]["epoch"] = int(epoch)
            task["state"]["train_loss"] = [round(float(x), 6) for x in train_loss]
            task["state"]["val_loss"] = [round(float(x), 6) for x in val_loss]
            task["state"]["logs"] = list(logs)

def _stop_requested(task_id):
    task = TRAIN_TASKS.get(task_id)
    return bool(task and task["stop"].is_set())

def _run_train_thread(task_id, kwargs):
    try:
        result = _train_worker(task_id, **kwargs)
        task = TRAIN_TASKS.get(task_id)
        if task:
            task["state"]["result"] = result
            task["state"]["status"] = "done"
    except Exception as e:
        import traceback
        traceback.print_exc()
        task = TRAIN_TASKS.get(task_id)
        if task:
            task["state"]["error"] = str(e)
            task["state"]["status"] = "error"

# 各算法推荐可调超参数（前端滑动条 + 输入框联动渲染）
# type: int/float -> 滑动条+输入框联动；choice -> 下拉选择；text -> 文本框
PARAM_SCHEMAS = {
    "linear_regression": [
        {"key": "learning_rate", "label": "学习率", "type": "float", "min": 0.0, "max": 1.0, "step": 0.001, "default": 0.01},
        {"key": "alpha", "label": "L2 正则系数", "type": "float", "min": 0.0, "max": 0.1, "step": 0.00001, "default": 0.0001},
        {"key": "epochs", "label": "训练轮数", "type": "int", "min": 10, "max": 500, "step": 10, "default": 100},
    ],
    "logistic_regression": [
        {"key": "learning_rate", "label": "学习率", "type": "float", "min": 0.0, "max": 1.0, "step": 0.001, "default": 0.01},
        {"key": "alpha", "label": "L2 正则系数", "type": "float", "min": 0.0, "max": 0.1, "step": 0.00001, "default": 0.0001},
        {"key": "epochs", "label": "训练轮数", "type": "int", "min": 10, "max": 500, "step": 10, "default": 100},
    ],
    "mlp_regressor": [
        {"key": "hidden_layer_sizes", "label": "隐藏层结构", "type": "text", "default": "64,32"},
        {"key": "activation", "label": "激活函数", "type": "choice", "options": ["relu", "tanh", "logistic"], "default": "relu"},
        {"key": "learning_rate", "label": "学习率", "type": "float", "min": 0.0, "max": 0.1, "step": 0.0001, "default": 0.001},
        {"key": "batch_size", "label": "批大小", "type": "int", "min": 8, "max": 256, "step": 8, "default": 32},
        {"key": "epochs", "label": "训练轮数", "type": "int", "min": 10, "max": 1000, "step": 10, "default": 200},
    ],
    "mlp_classifier": [
        {"key": "hidden_layer_sizes", "label": "隐藏层结构", "type": "text", "default": "64,32"},
        {"key": "activation", "label": "激活函数", "type": "choice", "options": ["relu", "tanh", "logistic"], "default": "relu"},
        {"key": "learning_rate", "label": "学习率", "type": "float", "min": 0.0, "max": 0.1, "step": 0.0001, "default": 0.001},
        {"key": "batch_size", "label": "批大小", "type": "int", "min": 8, "max": 256, "step": 8, "default": 32},
        {"key": "epochs", "label": "训练轮数", "type": "int", "min": 10, "max": 1000, "step": 10, "default": 200},
    ],
    "random_forest": [
        {"key": "n_estimators", "label": "树数量", "type": "int", "min": 10, "max": 500, "step": 10, "default": 100},
        {"key": "max_depth", "label": "最大深度", "type": "int", "min": 1, "max": 50, "step": 1, "default": 10},
    ],
    "decision_tree": [
        {"key": "max_depth", "label": "最大深度", "type": "int", "min": 1, "max": 50, "step": 1, "default": 10},
        {"key": "min_samples_split", "label": "最小分裂样本", "type": "int", "min": 2, "max": 20, "step": 1, "default": 2},
    ],
    "knn": [
        {"key": "n_neighbors", "label": "邻居数 K", "type": "int", "min": 1, "max": 50, "step": 1, "default": 5},
        {"key": "weights", "label": "权重方式", "type": "choice", "options": ["uniform", "distance"], "default": "uniform"},
    ],
    "svm": [
        {"key": "C", "label": "正则强度 C", "type": "float", "min": 0.0, "max": 100.0, "step": 0.1, "default": 1.0},
        {"key": "kernel", "label": "核函数", "type": "choice", "options": ["rbf", "linear", "poly", "sigmoid"], "default": "rbf"},
    ],
    "kmeans": [
        {"key": "n_clusters", "label": "聚类数 K", "type": "int", "min": 2, "max": 20, "step": 1, "default": 3},
        {"key": "max_iter", "label": "最大迭代", "type": "int", "min": 100, "max": 1000, "step": 50, "default": 300},
    ],
}

@app.get("/train_algo_params")
async def train_algo_params(algorithm: str = Query(default="")):
    """返回指定算法推荐可调超参数 schema（滑动条/输入框渲染用）。"""
    if not algorithm:
        return {"code": 0, "msg": "ok", "data": PARAM_SCHEMAS}
    return {"code": 0, "msg": "ok", "data": PARAM_SCHEMAS.get(algorithm, [])}

def _prepare_xy(df: pd.DataFrame, target_column: str):
    """构建特征矩阵 X 与目标 y（数值填充 + 类别 one-hot；y 分类时编码为数值）。"""
    df = df.copy()
    num_cols = [str(c) for c in df.select_dtypes(include="number").columns]
    cat_cols = [str(c) for c in df.columns if str(c) not in num_cols]
    feature_cols = [c for c in df.columns if c != target_column]
    # 数值列：填充中位数
    keep_num = [c for c in feature_cols if c in num_cols]
    for c in keep_num:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].fillna(df[c].median())
    # 类别列：低基数 one-hot
    keep_cat = [c for c in feature_cols if c in cat_cols and df[c].nunique() <= 20]
    X_parts = [df[keep_num].values] if keep_num else []
    if keep_cat:
        X_parts.append(pd.get_dummies(df[keep_cat], drop_first=False).values.astype(float))
    if not X_parts:
        raise ValueError("没有可用特征列（数值列或低基数类别列）")
    X = np.hstack(X_parts) if len(X_parts) > 1 else X_parts[0]
    # one-hot 后的列名（预测时需按训练列顺序对齐）
    onehot_columns = list(pd.get_dummies(df[keep_cat], drop_first=False).columns) if keep_cat else []
    # 目标列
    y = df[target_column]
    le = None
    if y.dtype == object or y.nunique() <= 12:
        le = LabelEncoder()
        y = le.fit_transform(y.fillna("NA").astype(str))
    else:
        y = pd.to_numeric(y, errors="coerce").fillna(y.median() if not pd.isna(y.median()) else 0).values
        y = np.asarray(y, dtype=float)
    return X, np.asarray(y), le, keep_num, keep_cat, onehot_columns

def _prepare_xy_aligned(df, target_column, ref):
    """按参考（训练集）的特征定义构建 X/y，保证 val/test 列与训练集严格对齐。
    ref 需含 keep_num / keep_cat / onehot_columns / le / median_map。"""
    df = df.copy()
    keep_num = ref["keep_num"]; keep_cat = ref["keep_cat"]
    median_map = ref.get("median_map") or {}
    for cc in keep_num:
        df[cc] = pd.to_numeric(df[cc], errors="coerce")
        if cc in median_map:
            df[cc] = df[cc].fillna(median_map[cc])
    X_parts = [df[keep_num].values] if keep_num else []
    if keep_cat:
        d = pd.get_dummies(df[keep_cat], drop_first=False)
        for col in ref["onehot_columns"]:
            if col not in d.columns:
                d[col] = 0.0
        d = d[ref["onehot_columns"]].values.astype(float)
        X_parts.append(d)
    X = np.hstack(X_parts) if len(X_parts) > 1 else X_parts[0]
    y_raw = df[target_column]
    le = ref.get("le")
    if le is not None:
        def _safe(s):
            return s if s in le.classes_ else le.classes_[0]
        y = le.transform(y_raw.fillna("NA").astype(str).map(_safe))
    else:
        y = pd.to_numeric(y_raw, errors="coerce")
        y = y.fillna(float(y.median()) if not pd.isna(y.median()) else 0).values.astype(float)
    return X, np.asarray(y)

def _save_model_bundle(model, scaler, le, keep_num, keep_cat, onehot_columns,
                       median_map, target_column, algorithm, task_type, params, base, tag,
                       y_scaler=None):
    """把训练好的模型及配套预处理信息打包保存为 joblib 文件。
    tag 取 best_loss（验证 loss 最低的 epoch 模型）或 last_loss（最终模型）。"""
    os.makedirs(model_folder, exist_ok=True)
    path = os.path.join(model_folder, f"{algorithm}_{tag}.joblib")
    bundle = {
        "model": model,
        "scaler": scaler,
        "y_scaler": y_scaler,
        "label_encoder": le,
        "keep_num": keep_num,
        "keep_cat": keep_cat,
        "onehot_columns": onehot_columns,
        "median_map": median_map,
        "target_column": target_column,
        "algorithm": algorithm,
        "task_type": task_type,
        "params": params,
    }
    try:
        joblib.dump(bundle, path)
    except Exception:
        path = None
    return path

def _make_code(file_path, algorithm, target_column, test_size, params):
    """根据算法与超参数，生成可读的 sklearn 模型构建代码文本（确定性模板）。"""
    algo = TRAIN_ALGOS[algorithm]
    rows = []
    rows.append(f"import pandas as pd, numpy as np")
    rows.append(f"from sklearn.model_selection import train_test_split")
    rows.append(f"from sklearn.preprocessing import StandardScaler")
    rows.append("")
    rows.append(f"df = pd.read_csv(r\"{file_path}\")")
    rows.append(f"X = df.drop(columns=[\"{target_column}\"])")
    rows.append(f"y = df[\"{target_column}\"]")
    rows.append("X = pd.get_dummies(X, drop_first=True)  # 类别列 one-hot")
    rows.append("X = X.fillna(X.median())  # 填充缺失")
    rows.append(f"X_train, X_test, y_train, y_test = train_test_split(X, y, test_size={test_size}, random_state=42)")
    rows.append("scaler = StandardScaler()  # 标准化：SGD 等对特征尺度敏感的算法必需")
    rows.append("X_train = scaler.fit_transform(X_train)")
    rows.append("X_test = scaler.transform(X_test)")
    rows.append("")
    algo_line = ""
    if algorithm == "linear_regression":
        algo_line = f"model = SGDRegressor(loss='squared_error', alpha={params.get('alpha', 0.0001)}, learning_rate='invscaling', eta0={params.get('learning_rate', 0.01)}, power_t=0.25, max_iter={params.get('epochs', 100)}, random_state=42)"
        rows.append("# 回归：线性回归（随机梯度下降）")
        rows.append("y_scaler = StandardScaler()  # 目标标准化：SGD 对目标量级敏感，防止梯度爆炸")
        rows.append("y_train = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()")
        rows.append("y_test = y_scaler.transform(y_test.values.reshape(-1, 1)).ravel()")
        rows.append("model.fit(X_train, y_train)")
        rows.append("pred = y_scaler.inverse_transform(model.predict(X_test).reshape(-1, 1)).ravel()")
        rows.append("from sklearn.metrics import mean_squared_error, r2_score")
        rows.append("print('MSE =', mean_squared_error(y_test, pred), 'R2 =', r2_score(y_test, pred))")
    elif algorithm == "logistic_regression":
        algo_line = f"model = SGDClassifier(loss='log_loss', alpha={params.get('alpha', 0.0001)}, learning_rate='constant', eta0={params.get('learning_rate', 0.01)}, max_iter={params.get('epochs', 100)}, random_state=42)"
        rows.append("# 分类：逻辑回归（随机梯度下降）")
    elif algorithm == "mlp_regressor":
        hls = tuple(int(x) for x in str(params.get("hidden_layer_sizes", "64,32")).split(",") if x.strip())
        algo_line = f"model = MLPRegressor(hidden_layer_sizes={hls}, activation='{params.get('activation', 'relu')}', alpha={params.get('alpha', 0.0001)}, learning_rate_init={params.get('learning_rate', 0.001)}, batch_size={params.get('batch_size', 32)}, max_iter={params.get('epochs', 200)}, random_state=42)"
        rows.insert(2, "from sklearn.neural_network import MLPRegressor")
        rows.append("# 回归：MLP 神经网络")
    elif algorithm == "mlp_classifier":
        hls = tuple(int(x) for x in str(params.get("hidden_layer_sizes", "64,32")).split(",") if x.strip())
        algo_line = f"model = MLPClassifier(hidden_layer_sizes={hls}, activation='{params.get('activation', 'relu')}', alpha={params.get('alpha', 0.0001)}, learning_rate_init={params.get('learning_rate', 0.001)}, batch_size={params.get('batch_size', 32)}, max_iter={params.get('epochs', 200)}, random_state=42)"
        rows.insert(2, "from sklearn.neural_network import MLPClassifier")
        rows.append("# 分类：MLP 神经网络")
    elif algorithm == "random_forest":
        algo_line = f"model = RandomForestClassifier(n_estimators={params.get('n_estimators', 100)}, max_depth={params.get('max_depth', 10)}, random_state=42)"
        rows.insert(2, "from sklearn.ensemble import RandomForestClassifier")
        rows.append("# 分类/回归：随机森林")
    elif algorithm == "decision_tree":
        algo_line = f"model = DecisionTreeClassifier(max_depth={params.get('max_depth', 10)}, min_samples_split={params.get('min_samples_split', 2)}, random_state=42)"
        rows.insert(2, "from sklearn.tree import DecisionTreeClassifier")
        rows.append("# 分类/回归：决策树")
    elif algorithm == "knn":
        algo_line = f"model = KNeighborsClassifier(n_neighbors={params.get('n_neighbors', 5)}, weights='{params.get('weights', 'uniform')}')"
        rows.insert(2, "from sklearn.neighbors import KNeighborsClassifier")
        rows.append("# 分类/回归：K 近邻")
    elif algorithm == "svm":
        algo_line = f"model = SVC(C={params.get('C', 1.0)}, kernel='{params.get('kernel', 'rbf')}')"
        rows.insert(2, "from sklearn.svm import SVC")
        rows.append("# 分类/回归：支持向量机")
    elif algorithm == "kmeans":
        rows = rows[:0]
        rows.append("import pandas as pd, numpy as np")
        rows.append("from sklearn.preprocessing import StandardScaler")
        rows.append("from sklearn.cluster import KMeans")
        rows.append("")
        rows.append(f"df = pd.read_csv(r\"{file_path}\")")
        rows.append("X = df.select_dtypes(include='number').fillna(df.select_dtypes(include='number').median())")
        rows.append("X = StandardScaler().fit_transform(X)")
        rows.append(f"model = KMeans(n_clusters={params.get('n_clusters', 3)}, max_iter={params.get('max_iter', 300)}, random_state=42)")
        rows.append("labels = model.fit_predict(X)")
        rows.append("print('聚类标签分布:', np.bincount(labels))")
        rows.append("print('SSE(inertia):', model.inertia_)")
        rows.append("")
        rows.append("完整可运行代码（展示）如下：")
        rows.append(algo_line or "")
        return "\n".join(rows)
    rows.append(algo_line)
    rows.append("")
    rows.append("model.fit(X_train, y_train)  # 训练模型")
    rows.append("print('R2:', model.score(X_test, y_test))  # 评估")
    return "\n".join(rows)

@app.post("/train_split")
async def train_split(
    file_path: str = Query(),
    test_size: float = Query(default=0.2),
    val_size: float = Query(default=0.15),
):
    """对整个数据集（不依赖目标列）切分为训练集/验证集/测试集，保存切分结果并返回概要。"""
    try:
        df = load_df_checked(file_path)
    except Exception as e:
        return {"code": -1, "msg": str(e)}
    ts = min(max(float(test_size), 0.05), 0.8)
    vs = min(max(float(val_size), 0.0), 0.8)  # 允许 0：不切验证集（两切分）
    if vs > 0 and ts + vs > 0.9:
        vs = max(0.05, round(0.9 - ts, 2))
    base = os.path.splitext(os.path.basename(file_path))[0]
    train_path = os.path.join(split_folder, f"{base}_train.csv")
    val_path = os.path.join(split_folder, f"{base}_val.csv")
    test_path = os.path.join(split_folder, f"{base}_test.csv")
    try:
        if vs > 0:
            train_val_df, test_df = train_test_split(df, test_size=ts, random_state=42)
            val_ratio = vs / (1 - ts) if (1 - ts) > 0 else 0.15
            train_df, val_df = train_test_split(train_val_df, test_size=val_ratio, random_state=42)
        else:
            train_df, test_df = train_test_split(df, test_size=ts, random_state=42)
            val_df = pd.DataFrame()
    except Exception as e:
        return {"code": -1, "msg": f"切分失败: {str(e)}"}
    train_df.to_csv(train_path, index=False, encoding="utf-8-sig")
    if vs > 0:
        val_df.to_csv(val_path, index=False, encoding="utf-8-sig")
    else:
        val_path = ""
    test_df.to_csv(test_path, index=False, encoding="utf-8-sig")
    return {
        "code": 0,
        "msg": "切分完成",
        "data": {
            "total_rows": len(df),
            "train_rows": len(train_df),
            "val_rows": len(val_df) if vs > 0 else 0,
            "test_rows": len(test_df),
            "test_size": ts,
            "val_size": vs,
            "train_path": train_path,
            "val_path": val_path,
            "test_path": test_path,
        },
    }

@app.get("/train_download")
async def train_download(file_path: str = Query()):
    """下载切分后的数据集文件（仅限 split 目录）。"""
    abs_split = os.path.abspath(split_folder)
    abs_file = os.path.abspath(file_path)
    if not abs_file.startswith(abs_split):
        return {"code": -1, "msg": "禁止下载非切分目录文件"}
    if not os.path.exists(abs_file):
        return {"code": -1, "msg": "文件不存在"}
    return FileResponse(abs_file, filename=os.path.basename(abs_file))

@app.post("/train_code")
async def train_code(
    file_path: str = Query(),
    algorithm: str = Query(default="mlp_regressor"),
    target_column: str = Query(default=""),
    test_size: float = Query(default=0.2),
    params_json: str = Query(default="{}"),
):
    """按用户选择的算法与超参数，实时生成模型构建代码（确定性模板，展示用）。"""
    if algorithm not in TRAIN_ALGOS:
        return {"code": -1, "msg": f"不支持的算法: {algorithm}"}
    try:
        df = load_df_checked(file_path)
    except Exception as e:
        return {"code": -1, "msg": str(e)}
    if not target_column or target_column not in df.columns:
        return {"code": -1, "msg": "请先选择目标列"}
    try:
        params = json.loads(params_json) if params_json else {}
    except Exception:
        params = {}
    code = _make_code(file_path, algorithm, target_column, test_size, params)
    return {"code": 0, "msg": "ok", "data": {"code": code, "algorithm": algorithm, "algorithm_name": TRAIN_ALGOS[algorithm]["name"]}}

def _build_cv_model(algorithm, params, task_type):
    """为 K 折交叉验证构造未拟合模型（迭代算法用合理 max_iter，保证各折可收敛）。"""
    if algorithm == "linear_regression":
        from sklearn.compose import TransformedTargetRegressor
        # K 折初始学习率上限 0.001：三集模式下离群样本多的折（加州房价 AveOccup 等）
        # 用 0.005 仍可能发散（该折 MSE 达 6.7e4），限制到 0.001 后 5 折全部收敛；
        # 保留用户学习率但向下截断。主训练不受影响（partial_fit + invscaling 全量收敛）。
        return TransformedTargetRegressor(
            regressor=SGDRegressor(alpha=float(params.get("alpha", 0.0001)), learning_rate="invscaling",
                                   eta0=min(float(params.get("learning_rate", 0.01)), 0.001), power_t=0.25,
                                   max_iter=500, tol=1e-3, random_state=42),
            transformer=StandardScaler())
    if algorithm == "logistic_regression":
        return SGDClassifier(loss="log_loss", alpha=float(params.get("alpha", 0.0001)), learning_rate="constant",
                             eta0=float(params.get("learning_rate", 0.01)), max_iter=500, tol=1e-3, random_state=42)
    if algorithm == "mlp_regressor":
        hls = tuple(int(x) for x in str(params.get("hidden_layer_sizes", "64,32")).split(",") if x.strip()) or (16,)
        return MLPRegressor(hidden_layer_sizes=hls, activation=str(params.get("activation", "relu")),
                            alpha=float(params.get("alpha", 0.0001)), learning_rate_init=float(params.get("learning_rate", 0.001)),
                            batch_size=int(params.get("batch_size", 32)), max_iter=300, random_state=42)
    if algorithm == "mlp_classifier":
        hls = tuple(int(x) for x in str(params.get("hidden_layer_sizes", "64,32")).split(",") if x.strip()) or (16,)
        return MLPClassifier(hidden_layer_sizes=hls, activation=str(params.get("activation", "relu")),
                             alpha=float(params.get("alpha", 0.0001)), learning_rate_init=float(params.get("learning_rate", 0.001)),
                             batch_size=int(params.get("batch_size", 32)), max_iter=300, random_state=42)
    if algorithm == "random_forest":
        if task_type == "regression":
            return RandomForestRegressor(n_estimators=int(params.get("n_estimators", 100)), max_depth=int(params.get("max_depth", 10)), random_state=42)
        return RandomForestClassifier(n_estimators=int(params.get("n_estimators", 100)), max_depth=int(params.get("max_depth", 10)), random_state=42)
    if algorithm == "decision_tree":
        if task_type == "regression":
            return DecisionTreeRegressor(max_depth=int(params.get("max_depth", 10)), min_samples_split=int(params.get("min_samples_split", 2)), random_state=42)
        return DecisionTreeClassifier(max_depth=int(params.get("max_depth", 10)), min_samples_split=int(params.get("min_samples_split", 2)), random_state=42)
    if algorithm == "knn":
        if task_type == "regression":
            return KNeighborsRegressor(n_neighbors=int(params.get("n_neighbors", 5)), weights=str(params.get("weights", "uniform")))
        return KNeighborsClassifier(n_neighbors=int(params.get("n_neighbors", 5)), weights=str(params.get("weights", "uniform")))
    if algorithm == "svm":
        if task_type == "regression":
            return SVR(C=float(params.get("C", 1.0)), kernel=str(params.get("kernel", "rbf")))
        return SVC(C=float(params.get("C", 1.0)), kernel=str(params.get("kernel", "rbf")))
    return None

def _run_cv(algorithm, params, X, y, task_type, k_fold):
    """执行 K 折交叉验证，返回每折得分 + 均值 + 标准差。"""
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    model = _build_cv_model(algorithm, params, task_type)
    if model is None:
        return None, "该算法暂不支持 K 折交叉验证"
    scoring = "accuracy" if task_type == "classification" else "neg_mean_squared_error"
    pipe = make_pipeline(StandardScaler(), model)
    scores = cross_val_score(pipe, X, y, cv=int(k_fold), scoring=scoring, error_score="raise")
    if task_type == "classification":
        per = [round(float(s), 5) for s in scores]
        return {"k": int(k_fold), "metric": "Accuracy", "scores": per,
                "mean": round(float(scores.mean()), 5), "std": round(float(scores.std()), 5)}, ""
    per = [round(float(-s), 5) for s in scores]
    return {"k": int(k_fold), "metric": "MSE", "scores": per,
            "mean": round(float(np.mean(per)), 5), "std": round(float(np.std(per)), 5)}, ""

def _train_worker(
    task_id,
    file_path: str = "",
    algorithm: str = "mlp_regressor",
    target_column: str = "",
    test_size: float = 0.2,
    params_json: str = "{}",
    k_fold: int = 0,
    train_file_path: str = "",
    val_file_path: str = "",
    test_file_path: str = "",
):
    """后台线程执行的训练主体（原 train_run 逻辑）：按用户选择的算法与超参数训练模型，
    逐 epoch 实时写进度到 TRAIN_TASKS（前端轮询绘制 loss 曲线），支持中途停止。"""
    logs = []
    _task = TRAIN_TASKS.get(task_id)
    if _task:
        with TRAIN_TASKS_LOCK:
            _task["state"]["status"] = "running"
            _task["state"]["logs"] = logs
    """按用户选择的算法与超参数训练模型，返回逐 epoch loss 曲线与评估指标。
    有迭代过程的算法（SGD 线性/逻辑、MLP）返回 train_loss + val_loss 曲线；
    其他算法返回评估指标并在 loss_curve 中说明不支持逐 epoch loss。"""
    if algorithm not in TRAIN_ALGOS:
        return {"code": -1, "msg": f"不支持的算法: {algorithm}"}
    use_three = bool(train_file_path)
    if not use_three:
        try:
            df = load_df_checked(file_path)
        except Exception as e:
            return {"code": -1, "msg": str(e)}
    if not target_column:
        return {"code": -1, "msg": "请先选择目标列"}
    check_df = load_df_checked(train_file_path) if use_three else df
    if target_column not in check_df.columns:
        return {"code": -1, "msg": f"目标列不存在: {target_column}"}
    try:
        params = json.loads(params_json) if params_json else {}
    except Exception:
        params = {}
    ts = min(max(float(test_size), 0.05), 0.8)
    scaler = StandardScaler()
    if use_three:
        # ---- 三集模式：训练集训练、验证集验证、测试集测试 ----
        df_train = check_df
        df_val = load_df_checked(val_file_path) if val_file_path else None
        df_test = load_df_checked(test_file_path) if test_file_path else None
        if df_val is None:
            df_train, df_val = train_test_split(df_train, test_size=0.15, random_state=42)
        if df_test is None:
            df_train, df_test = train_test_split(df_train, test_size=ts, random_state=42)
        try:
            X_train, y_train, le, keep_num, keep_cat, onehot_columns = _prepare_xy(df_train, target_column)
            median_map = {cc: float(df_train[cc].median()) for cc in keep_num} if keep_num else {}
            ref = {"keep_num": keep_num, "keep_cat": keep_cat, "onehot_columns": onehot_columns,
                   "le": le, "median_map": median_map}
            X_val, y_val = _prepare_xy_aligned(df_val, target_column, ref)
            X_test, y_test = _prepare_xy_aligned(df_test, target_column, ref)
        except Exception as e:
            return {"code": -1, "msg": f"特征构建失败: {str(e)}"}
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        file_path = train_file_path
        df = df_train
        logs.append(f"[1/8] 加载数据：{os.path.basename(file_path)}（{len(df)} 行 × {len(df.columns)} 列）")
        logs.append(f"[2/8] 特征构建：数值 {len(keep_num)} 个、类别 {len(keep_cat)} 个，one-hot 后 {X_train.shape[1]} 维")
        logs.append(f"[3/8] 三集模式：训练 {X_train.shape[0]} / 验证 {X_val.shape[0]} / 测试 {X_test.shape[0]}")
        logs.append("[4/8] 特征标准化完成（StandardScaler：训练集拟合，验证/测试集变换）")
    else:
        # ---- 单文件模式：内部切分 ----
        if target_column not in df.columns:
            return {"code": -1, "msg": "请先选择目标列"}
        try:
            X, y, le, keep_num, keep_cat, onehot_columns = _prepare_xy(df, target_column)
        except Exception as e:
            return {"code": -1, "msg": f"特征构建失败: {str(e)}"}
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts, random_state=42)
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        X_val, y_val = X_test, y_test
        logs.append(f"[1/8] 加载数据：{os.path.basename(file_path)}（{len(df)} 行 × {len(df.columns)} 列）")
        logs.append(f"[2/8] 特征构建：数值 {len(keep_num)} 个、类别 {len(keep_cat)} 个，one-hot 后 {X.shape[1]} 维")
        logs.append(f"[3/8] 内部切分：训练 {X_train.shape[0]} / 验证 {X_val.shape[0]} / 测试 {X_test.shape[0]}")
        logs.append("[4/8] 特征标准化完成（StandardScaler：训练集拟合，测试集变换）")

    y_scaler = None  # 回归目标标准化器（SGD 等对目标量级敏感的算法必需）
    algo = TRAIN_ALGOS[algorithm]
    task_type = algo["type"]
    if task_type == "auto":
        task_type = "classification" if (le is not None) else "regression"
    logs.append(f"[5/8] 开始训练：{algo['name']}（{task_type}），超参数：{json.dumps(params, ensure_ascii=False)}")

    metrics = {}
    train_loss, val_loss = [], []
    note = ""

    try:
        if algorithm == "linear_regression":
            lr = float(params.get("learning_rate", 0.01)); alpha = float(params.get("alpha", 0.0001))
            epochs = int(params.get("epochs", 100))
            # 目标标准化：SGD 对目标值量级敏感（房价等大数值目标直接训练会梯度爆炸：
            # Loss 达 1e23、MSE 1e22、R² 负巨值，表现为"假过拟合"）。训练用标准化 y，
            # 评估/预测时反变换回原尺度。
            y_scaler = StandardScaler()
            y_train_s = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
            y_val_s = y_scaler.transform(y_val.reshape(-1, 1)).ravel()
            # learning_rate='invscaling'：学习率随轮次衰减（eta0/t^0.25）。
            # 用 'constant' 固定学习率时，大样本+极端特征值（加州房价 AveOccup 等离群）会梯度正反馈爆炸
            # （Loss 达 1e21、R² 负巨值，表现为"假过拟合"）。invscaling 既保留用户学习率，又保证收敛。
            model = SGDRegressor(loss="squared_error", alpha=alpha, learning_rate="invscaling", eta0=lr, power_t=0.25, max_iter=1, tol=None, warm_start=True, random_state=42)
            best_val, best_model = float("inf"), None
            for _e in range(max(epochs, 1)):
                model.partial_fit(X_train, y_train_s)
                train_loss.append(float(mean_squared_error(y_train_s, model.predict(X_train))))
                v = float(mean_squared_error(y_val_s, model.predict(X_val)))
                val_loss.append(v)
                if v < best_val:  # 记录验证 loss 最低的 epoch 模型
                    best_val, best_model = v, copy.deepcopy(model)
                _pub_progress(task_id, _e + 1, train_loss, val_loss, logs)
                if _stop_requested(task_id):
                    logs.append(f"[6/8] 训练被用户手动停止（epoch={_e + 1}），已保存当前模型")
                    break
                time.sleep(_sleep_for(epochs))
            last_model = copy.deepcopy(model)
            pred_orig = y_scaler.inverse_transform(model.predict(X_test).reshape(-1, 1)).ravel()
            metrics = {"MSE": round(float(mean_squared_error(y_test, pred_orig)), 5),
                       "RMSE": round(float(mean_squared_error(y_test, pred_orig) ** 0.5), 5),
                       "MAE": round(float(mean_absolute_error(y_test, pred_orig)), 5),
                       "R2": round(float(r2_score(y_test, pred_orig)), 5)}
        elif algorithm == "logistic_regression":
            lr = float(params.get("learning_rate", 0.01)); alpha = float(params.get("alpha", 0.0001))
            epochs = int(params.get("epochs", 100))
            from sklearn.metrics import log_loss
            classes = np.unique(y_train)
            model = SGDClassifier(loss="log_loss", alpha=alpha, learning_rate="constant", eta0=lr, max_iter=1, tol=None, warm_start=True, random_state=42)
            best_val, best_model = float("inf"), None
            for _e in range(max(epochs, 1)):
                model.partial_fit(X_train, y_train, classes=classes)
                tr_p = model.predict_proba(X_train); te_p = model.predict_proba(X_val)
                train_loss.append(round(float(log_loss(y_train, tr_p, labels=classes)), 5))
                v = round(float(log_loss(y_val, te_p, labels=classes)), 5)
                val_loss.append(v)
                if v < best_val:
                    best_val, best_model = v, copy.deepcopy(model)
                _pub_progress(task_id, _e + 1, train_loss, val_loss, logs)
                if _stop_requested(task_id):
                    logs.append(f"[6/8] 训练被用户手动停止（epoch={_e + 1}），已保存当前模型")
                    break
                time.sleep(_sleep_for(epochs))
            last_model = copy.deepcopy(model)
            pred = model.predict(X_test)
            metrics = {"Accuracy": round(float(accuracy_score(y_test, pred)), 5),
                       "Precision": round(float(precision_score(y_test, pred, average="weighted", zero_division=0)), 5),
                       "Recall": round(float(recall_score(y_test, pred, average="weighted", zero_division=0)), 5),
                       "F1": round(float(f1_score(y_test, pred, average="weighted", zero_division=0)), 5)}
        elif algorithm == "mlp_regressor":
            hls = tuple(int(x) for x in str(params.get("hidden_layer_sizes", "64,32")).split(",") if x.strip()) or (16,)
            activation = str(params.get("activation", "relu")); alpha = float(params.get("alpha", 0.0001))
            lr = float(params.get("learning_rate", 0.001)); bs = int(params.get("batch_size", 32))
            epochs = int(params.get("epochs", 200))
            model = MLPRegressor(hidden_layer_sizes=hls, activation=activation, alpha=alpha, learning_rate_init=lr, batch_size=bs, learning_rate="constant", max_iter=1, warm_start=True, random_state=42, tol=0.0)
            best_val, best_model = float("inf"), None
            for _e in range(max(epochs, 1)):
                model.fit(X_train, y_train)
                train_loss.append(float(mean_squared_error(y_train, model.predict(X_train))))
                v = float(mean_squared_error(y_val, model.predict(X_val)))
                val_loss.append(v)
                if v < best_val:
                    best_val, best_model = v, copy.deepcopy(model)
                _pub_progress(task_id, _e + 1, train_loss, val_loss, logs)
                if _stop_requested(task_id):
                    logs.append(f"[6/8] 训练被用户手动停止（epoch={_e + 1}），已保存当前模型")
                    break
                time.sleep(_sleep_for(epochs))
            last_model = copy.deepcopy(model)
            pred = model.predict(X_test)
            metrics = {"MSE": round(float(mean_squared_error(y_test, pred)), 5),
                       "RMSE": round(float(mean_squared_error(y_test, pred) ** 0.5), 5),
                       "MAE": round(float(mean_absolute_error(y_test, pred)), 5),
                       "R2": round(float(r2_score(y_test, pred)), 5)}
        elif algorithm == "mlp_classifier":
            hls = tuple(int(x) for x in str(params.get("hidden_layer_sizes", "64,32")).split(",") if x.strip()) or (16,)
            activation = str(params.get("activation", "relu")); alpha = float(params.get("alpha", 0.0001))
            lr = float(params.get("learning_rate", 0.001)); bs = int(params.get("batch_size", 32))
            epochs = int(params.get("epochs", 200))
            from sklearn.metrics import log_loss
            model = MLPClassifier(hidden_layer_sizes=hls, activation=activation, alpha=alpha, learning_rate_init=lr, batch_size=bs, learning_rate="constant", max_iter=1, warm_start=True, random_state=42, tol=0.0)
            best_val, best_model = float("inf"), None
            for _e in range(max(epochs, 1)):
                model.fit(X_train, y_train)
                tr_p = model.predict_proba(X_train); te_p = model.predict_proba(X_val)
                train_loss.append(round(float(log_loss(y_train, tr_p, labels=model.classes_)), 5))
                v = round(float(log_loss(y_val, te_p, labels=model.classes_)), 5)
                val_loss.append(v)
                if v < best_val:
                    best_val, best_model = v, copy.deepcopy(model)
                _pub_progress(task_id, _e + 1, train_loss, val_loss, logs)
                if _stop_requested(task_id):
                    logs.append(f"[6/8] 训练被用户手动停止（epoch={_e + 1}），已保存当前模型")
                    break
                time.sleep(_sleep_for(epochs))
            last_model = copy.deepcopy(model)
            pred = model.predict(X_test)
            metrics = {"Accuracy": round(float(accuracy_score(y_test, pred)), 5),
                       "Precision": round(float(precision_score(y_test, pred, average="weighted", zero_division=0)), 5),
                       "Recall": round(float(recall_score(y_test, pred, average="weighted", zero_division=0)), 5),
                       "F1": round(float(f1_score(y_test, pred, average="weighted", zero_division=0)), 5)}
        elif algorithm == "random_forest":
            if task_type == "regression":
                model = RandomForestRegressor(n_estimators=int(params.get("n_estimators", 100)), max_depth=int(params.get("max_depth", 10)), random_state=42)
            else:
                model = RandomForestClassifier(n_estimators=int(params.get("n_estimators", 100)), max_depth=int(params.get("max_depth", 10)), random_state=42)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            if task_type == "regression":
                metrics = {"MSE": round(float(mean_squared_error(y_test, pred)), 5), "RMSE": round(float(mean_squared_error(y_test, pred) ** 0.5), 5), "MAE": round(float(mean_absolute_error(y_test, pred)), 5), "R2": round(float(r2_score(y_test, pred)), 5)}
            else:
                metrics = {"Accuracy": round(float(accuracy_score(y_test, pred)), 5), "F1": round(float(f1_score(y_test, pred, average="weighted", zero_division=0)), 5)}
            note = "随机森林不支持逐 epoch loss 曲线，已输出评估指标"
            best_model = last_model = model  # 非迭代算法：best 与 last 均为最终模型
        elif algorithm == "decision_tree":
            if task_type == "regression":
                model = DecisionTreeRegressor(max_depth=int(params.get("max_depth", 10)), min_samples_split=int(params.get("min_samples_split", 2)), random_state=42)
            else:
                model = DecisionTreeClassifier(max_depth=int(params.get("max_depth", 10)), min_samples_split=int(params.get("min_samples_split", 2)), random_state=42)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            if task_type == "regression":
                metrics = {"MSE": round(float(mean_squared_error(y_test, pred)), 5), "RMSE": round(float(mean_squared_error(y_test, pred) ** 0.5), 5), "MAE": round(float(mean_absolute_error(y_test, pred)), 5), "R2": round(float(r2_score(y_test, pred)), 5)}
            else:
                metrics = {"Accuracy": round(float(accuracy_score(y_test, pred)), 5), "F1": round(float(f1_score(y_test, pred, average="weighted", zero_division=0)), 5)}
            note = "决策树不支持逐 epoch loss 曲线，已输出评估指标"
            best_model = last_model = model  # 非迭代算法：best 与 last 均为最终模型
        elif algorithm == "knn":
            if task_type == "regression":
                model = KNeighborsRegressor(n_neighbors=int(params.get("n_neighbors", 5)), weights=str(params.get("weights", "uniform")))
            else:
                model = KNeighborsClassifier(n_neighbors=int(params.get("n_neighbors", 5)), weights=str(params.get("weights", "uniform")))
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            if task_type == "regression":
                metrics = {"MSE": round(float(mean_squared_error(y_test, pred)), 5), "RMSE": round(float(mean_squared_error(y_test, pred) ** 0.5), 5), "MAE": round(float(mean_absolute_error(y_test, pred)), 5), "R2": round(float(r2_score(y_test, pred)), 5)}
            else:
                metrics = {"Accuracy": round(float(accuracy_score(y_test, pred)), 5), "F1": round(float(f1_score(y_test, pred, average="weighted", zero_division=0)), 5)}
            note = "K近邻不支持逐 epoch loss 曲线，已输出评估指标"
            best_model = last_model = model  # 非迭代算法：best 与 last 均为最终模型
        elif algorithm == "svm":
            if task_type == "regression":
                model = SVR(C=float(params.get("C", 1.0)), kernel=str(params.get("kernel", "rbf")))
            else:
                model = SVC(C=float(params.get("C", 1.0)), kernel=str(params.get("kernel", "rbf")))
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            if task_type == "regression":
                metrics = {"MSE": round(float(mean_squared_error(y_test, pred)), 5), "RMSE": round(float(mean_squared_error(y_test, pred) ** 0.5), 5), "MAE": round(float(mean_absolute_error(y_test, pred)), 5), "R2": round(float(r2_score(y_test, pred)), 5)}
            else:
                metrics = {"Accuracy": round(float(accuracy_score(y_test, pred)), 5), "F1": round(float(f1_score(y_test, pred, average="weighted", zero_division=0)), 5)}
            note = "SVM 不支持逐 epoch loss 曲线，已输出评估指标"
            best_model = last_model = model  # 非迭代算法：best 与 last 均为最终模型
        elif algorithm == "kmeans":
            n_clusters = int(params.get("n_clusters", 3)); mi = int(params.get("max_iter", 300))
            model = KMeans(n_clusters=n_clusters, max_iter=mi, random_state=42, n_init=10)
            labels = model.fit_predict(X)
            metrics = {"SSE(inertia)": round(float(model.inertia_), 4)}
            if len(np.unique(labels)) > 1:
                try:
                    metrics["Silhouette"] = round(float(silhouette_score(X, labels)), 5)
                except Exception:
                    pass
            note = "KMeans 聚类无逐 epoch loss 曲线，已输出聚类指标"
            best_model = last_model = model  # 聚类：best 与 last 均为最终模型
    except Exception as e:
        import traceback
        traceback.print_exc()
        logs.append(f"[6/8] 训练失败：{e}")
        return {"code": -1, "msg": f"训练失败: {str(e)}", "logs": logs}

    logs.append(f"[6/8] 训练完成：epochs={max(len(train_loss), 1)}，评估指标：{json.dumps(metrics, ensure_ascii=False)}")

    # ---- 保存 best_loss / last_loss 两个模型（供下载与预测分析模块复用）----
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    median_map = {c: float(df[c].median()) for c in keep_num} if keep_num else {}
    best_path = _save_model_bundle(best_model, scaler, le, keep_num, keep_cat, onehot_columns,
                                   median_map, target_column, algorithm, task_type, params, base_name, "best_loss", y_scaler)
    last_path = _save_model_bundle(last_model, scaler, le, keep_num, keep_cat, onehot_columns,
                                   median_map, target_column, algorithm, task_type, params, base_name, "last_loss", y_scaler)
    model_saved = best_path is not None and last_path is not None
    logs.append("[7/8] 模型保存：" + ("best_loss / last_loss 成功" if model_saved else "失败"))

    # ---- K 折交叉验证（独立评估模型泛化稳定性）----
    cv_result = None
    if int(k_fold) >= 2:
        if task_type == "cluster":
            cv_result = {"note": "KMeans 聚类为无监督算法，不支持 K 折交叉验证"}
        else:
            try:
                if use_three:
                    X_cv, y_cv, _, _, _, _ = _prepare_xy(df_train, target_column)
                else:
                    X_cv, y_cv = X, y
                cv_result, cv_note = _run_cv(algorithm, params, X_cv, y_cv, task_type, int(k_fold))
                if cv_note:
                    cv_result = {"note": cv_note}
            except Exception as e:
                cv_result = {"error": f"K 折交叉验证失败: {str(e)}"}

    if cv_result:
        if "error" in cv_result:
            logs.append(f"[8/8] K 折交叉验证失败：{cv_result['error']}")
        elif "note" in cv_result:
            logs.append(f"[8/8] K 折交叉验证：{cv_result['note']}")
        elif "scores" in cv_result:
            logs.append(f"[8/8] K 折交叉验证：{cv_result['k']} 折 {cv_result['metric']} 均值 {cv_result['mean']}，标准差 {cv_result['std']}")
    else:
        logs.append("[8/8] K 折交叉验证：未启用")

    return {
        "code": 0,
        "msg": "训练完成",
        "data": {
            "algorithm": algorithm,
            "algorithm_name": TRAIN_ALGOS[algorithm]["name"],
            "problem_type": task_type,
            "train_loss": [round(float(x), 6) for x in train_loss],
            "val_loss": [round(float(x), 6) for x in val_loss],
            "epochs": max(len(train_loss), 1),
            "metrics": metrics,
            "note": note,
            "has_loss": len(train_loss) > 1,
            "feature_cols": keep_num + keep_cat,
            "test_size": ts,
            "cv": cv_result,
            "model_saved": model_saved,
            "best_model_file": os.path.basename(best_path) if best_path else "",
            "last_model_file": os.path.basename(last_path) if last_path else "",
            "logs": logs,
        },
    }

@app.get("/train_list_models")
async def train_list_models():
    """列出已训练并可下载/用于预测的模型文件（model_folder 内 joblib）。"""
    os.makedirs(model_folder, exist_ok=True)
    files = sorted([f for f in os.listdir(model_folder) if f.endswith(".joblib")], reverse=True)
    return {"code": 0, "msg": "ok", "data": {"models": files, "model_folder": model_folder}}

@app.get("/train_download_model")
async def train_download_model(model_path: str = Query(default=""), model_file: str = Query(default="")):
    """下载训练好的模型（best_loss / last_loss）。传 model_path（绝对路径）或 model_file（文件名）均可。"""
    os.makedirs(model_folder, exist_ok=True)
    if model_path:
        target = os.path.basename(model_path)  # 只取文件名，防路径穿越
    elif model_file:
        target = os.path.basename(model_file)
    else:
        return {"code": -1, "msg": "缺少模型文件参数"}
    p = os.path.abspath(os.path.join(model_folder, target))
    if os.path.dirname(p) != os.path.abspath(model_folder) or not os.path.exists(p):
        return {"code": -1, "msg": f"模型文件不存在: {target}"}
    return FileResponse(p, filename=target, media_type="application/octet-stream")

@app.post("/train_run")
async def train_run(
    file_path: str = Query(default=""),
    algorithm: str = Query(default="mlp_regressor"),
    target_column: str = Query(default=""),
    test_size: float = Query(default=0.2),
    params_json: str = Query(default="{}"),
    k_fold: int = Query(default=0),
    train_file_path: str = Query(default=""),
    val_file_path: str = Query(default=""),
    test_file_path: str = Query(default=""),
):
    """启动后台训练线程并立即返回 task_id；前端轮询 /train_progress 实时获取 loss 曲线，
    可通过 /train_stop 中途停止（训练几轮出一个点，曲线逐步生成）。"""
    if algorithm not in TRAIN_ALGOS:
        return {"code": -1, "msg": f"不支持的算法: {algorithm}"}
    if not target_column:
        return {"code": -1, "msg": "请先选择目标列"}
    task_id = uuid.uuid4().hex[:12]
    TRAIN_TASKS[task_id] = {
        "stop": threading.Event(),
        "state": {"status": "starting", "epoch": 0, "train_loss": [], "val_loss": [],
                  "logs": [], "result": None},
    }
    kwargs = dict(file_path=file_path, algorithm=algorithm, target_column=target_column,
                  test_size=test_size, params_json=params_json, k_fold=k_fold,
                  train_file_path=train_file_path, val_file_path=val_file_path,
                  test_file_path=test_file_path)
    threading.Thread(target=_run_train_thread, args=(task_id, kwargs), daemon=True).start()
    return {"code": 0, "msg": "训练已启动", "data": {"task_id": task_id}}

@app.get("/train_progress")
async def train_progress(task_id: str = Query()):
    """查询训练任务实时进度：running 返回当前 epoch + loss 点；done 返回完整结果。"""
    task = TRAIN_TASKS.get(task_id)
    if not task:
        return {"code": -1, "msg": "任务不存在"}
    st = task["state"]
    if st.get("status") == "done":
        return {"code": 0, "data": {"status": "done", "epoch": st.get("epoch", 0), "result": st.get("result")}}
    if st.get("status") == "error":
        return {"code": 0, "data": {"status": "error", "error": st.get("error"), "logs": st.get("logs", [])}}
    return {"code": 0, "data": {"status": "running", "epoch": st.get("epoch", 0),
                                "train_loss": st.get("train_loss", []), "val_loss": st.get("val_loss", []),
                                "logs": st.get("logs", [])}}

@app.post("/train_stop")
async def train_stop(task_id: str = Query()):
    """请求停止训练：设置 stop 标志，训练循环在下一个 epoch 退出并保存当前模型。"""
    task = TRAIN_TASKS.get(task_id)
    if not task:
        return {"code": -1, "msg": "任务不存在"}
    task["stop"].set()
    return {"code": 0, "msg": "已请求停止训练"}

