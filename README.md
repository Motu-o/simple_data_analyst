# 📊 本地数据集分析系统（AI大模型数据分析预测平台 Pro）

基于 **Ollama 本地大模型 + FastAPI 后端 + 单页 HTML 前端** 的本地数据分析平台。

**数据全程不出本地**：所有清洗、EDA 绘图、模型训练与预测均由后端确定性代码（pandas / matplotlib / scikit-learn / torch）执行，大模型（`qwen2.5:7b-instruct-q4_K_M`）仅用于**意图解析、算法推荐、分析建议与问答**，不会直接执行任意代码。

---

## ✨ 功能特性

系统为**单页应用**（`frontend/index.html`）：顶部 4 个功能导航（数据清洗 / EDA 分析 / 模型训练 / 预测分析）同页切换，左侧为数据集管理与"快撩我"聊天，无需页面跳转。

| 模块 | 说明 |
|---|---|
| 🎬 开局视频 | 打开页面先播放全屏视频（`start_up.mp4`），点「进入系统」直接进入；点「进入游戏」弹出云朵提醒"该工作的时候还是要工作的"，停 1 秒后进入；视频播完自动进入，进入过程带渐变过渡 |
| 💬 快撩我 | 左侧常驻大模型聊天框（流式输出），模型先回复"你好呀，我是奶龙~ 快撩我啊"，分析中提问回复"等一下奶龙~"；可清空聊天 |
| 🧹 数据清洗 | 上传数据集 → 选择清洗操作（去重 / 填充缺失 / 删缺失行 / 去空格 / 类型转换 / 重命名 / 替换 / 正则替换 / 数值四则运算新列等）→ **大模型分析意图** → 后端 **pandas 白名单确定性执行**；清洗成功/失败均给出可展开日志；清洗结果**手动下载**（`原数据集名_clean`），下载后自动清除缓存；下方独立**数据集展示模块** |
| 📊 EDA 分析 | **选项式按需绘图**（图表类型下拉 + 特征列多选），大模型先分析意图再由后端 matplotlib 确定性绘制（直方图/折线图/柱状图/散点图/箱线图/小提琴图/相关性热力图等，支持多变量），图像**实时展现绘制过程**、可打包下载；**EDA 分析建议**为大模型问答（流式输出），基于 pandas 计算的**相关性/特征重要性**给出分析报告（折叠默认收起，可下载 `原数据集名_特征分析报告.md`）；绘图失败打印日志 |
| 🧠 模型训练 | **三集切分**（训练/验证/测试，`原数据集名_train/_val/_test`，独立模块、可下载、自动加载到下方模型训练）；大模型判断问题类型（回归/分类/聚类）并**推荐算法** → 生成训练代码（折叠默认收起）→ **超参数人工调整**（滑动条 + 输入框联动，按算法推荐超参，支持小数、限制输入位数）→ **K 折交叉验证** → **实时 loss 曲线**（后台线程训练、前端轮询逐点平滑生长、可随时「停止训练」）→ 产出 **`{算法}_best_loss` / `{算法}_last_loss`** 两个模型可下载，供预测模块使用；训练日志折叠默认合上，失败打印日志；程序结束自动清理模型 |
| 🔮 预测分析 | 训练集训练 / 测试集测试 / 验证集验证，或**加载已训练模型**（best_loss / last_loss）对新数据预测；回归（SGD 四图：真实值 vs 预测值、残差分布直方图、学习曲线、残差-预测值散点）与分类（混淆矩阵、ROC 曲线 + AUC、特征重要性）诊断图，均**前端渲染**并按目标列/特征列标注；**分类加权投票集成（软投票）**：设置 n 值、上传 n 个分类模型（.joblib），上传验证集计算各模型 **Macro-F1** 并按**线性归一化**（w_i = F1_i / ΣF1_j）得到权重，预测概率按权重加权平均后取最大类别；**回归加权平均集成**：n 个回归模型 + 验证集（RMSE **误差倒数归一化**权重 w_i=(1/RMSE_i)/Σ(1/RMSE_j)）；输出各模型指标/权重、集成指标、权重图/类别分布/混淆矩阵/真实vs预测/残差图，结果可下载 |
| 📷 图片数据模块 | **数据表 / 图片数据** 双入口导航。图片 zip 上传（100M 上限，损坏/加密拒绝，路径穿越防护，损坏图片轻量解码标记不中断）→ **数据集分析**（类别分布、类别不平衡提示、分辨率/通道统计、损坏样本清单、三集类别对比、按类缩略图）→ **统一分辨率**（默认 224×224，不改变分类结构）→ **数据集切分**（训练/验证/测试按比例随机划分，独立背景框）→ **模型训练**（ResNet50，可调学习率/批次/轮数/优化器，GPU 加速，实时在线**数据增强**：随机水平翻转 p=0.5、随机旋转 10°、随机裁剪 scale=(0.8,1.0)、色彩抖动、随机高斯模糊，参数默认可改）→ **模型评估**（指标 + 大模型评估报告可下载 md）→ **预测推理**（上传图片推理） |

---

## 🏗️ 技术栈

- **后端**：FastAPI + Uvicorn（Python 3.12）
- **数据处理**：pandas、numpy、openpyxl（读 xlsx）
- **机器学习**：scikit-learn（线性/逻辑回归 SGD、MLP 神经网络、随机森林、决策树、KNN、SVM、KMeans）
- **深度学习**：PyTorch（torch 2.5.1+cu121 / torchvision 0.20.1+cu121，ResNet50 图片分类）
- **可视化**：matplotlib（图表实时动画帧）
- **大模型**：Ollama 本地模型 `qwen2.5:7b-instruct-q4_K_M`（全系统唯一接入模型）
- **前端**：原生 HTML + Tailwind CSS（CDN）+ Chart.js，无构建步骤

---

## 📁 目录结构

```
data_analyst/
├── backend/                       # 后端（FastAPI）
│   ├── config.py                  # 全局配置：存储目录 / 安全配置 / Ollama / FastAPI 实例 / 共享会话
│   ├── utils.py                   # 公共工具：Ollama 调用 / 代码提取修复 / 安全预检 / 数据读取
│   ├── main.py                    # 后端入口（路由注册 + 启动）
│   ├── launcher.py                # 一键启动器（后端 8000 + 前端 8081 + 自动开浏览器）
│   ├── routes/                    # 功能路由包（按功能分文件夹，各自 __init__.py）
│   │   ├── chat/__init__.py       # 通用与聊天：模型列表 / 数据集上传 / 快撩我聊天（流式）/ 数据分析问答
│   │   ├── image/__init__.py      # 图片数据模块：zip 上传 / 概览 / 切分 / 统一分辨率 / ResNet50 训练 / 评估 / 推理
│   │   ├── clean/__init__.py      # 数据清洗：意图解析 + 白名单确定性执行 + 日志
│   │   ├── eda/__init__.py        # EDA：选项式按需绘图（动画帧）+ 特征报告 + 打包下载
│   │   ├── train/__init__.py      # 训练：三集切分 / 推荐 / 代码生成 / 超参 / K折 / 实时 loss / 停止
│   │   └── predict/__init__.py    # 预测：回归/分类诊断 / 加载已训练模型 / 加权投票与加权平均集成
│   └── uploads/                   # 运行期数据（自动生成，可随时清空）
│       ├── datasets/              # 上传的数据集
│       ├── cleaned/               # 清洗结果
│       ├── eda_export/            # EDA 生成图片 / zip
│       ├── split/                 # 训练/验证/测试三集
│       ├── models/                # 训练模型（best_loss / last_loss，退出自动清理）
│       ├── image_sets/            # 图片数据集（zip 解压、切分、缩略图）
│       └── imgs/                  # 预测图表
├── frontend/                      # 前端（单页静态）
│   ├── index.html                 # 唯一入口：4 功能导航同页切换 + 图片数据 + 快撩我 + 数据集管理
│   └── start_up.mp4               # 开局视频
├── requirements.txt               # Python 依赖
└── README.md
```

---

## ⚙️ 环境要求

- **Windows / macOS / Linux**（本项目在 Windows 10+ 上开发验证）
- **Python 3.12+**（推荐 conda 环境，本机使用 `D:\application\Anaconda\envs\data_analyst`）
- **Ollama** 已安装并运行，已拉取模型 `qwen2.5:7b-instruct-q4_K_M`
  ```bash
  ollama pull qwen2.5:7b-instruct-q4_K_M
  ```
- **图片数据模块**（可选）：PyTorch 已安装（见 requirements.txt 注释，含 CUDA 12.1 GPU 版；无 GPU 则自动回退 CPU）

---

## 🚀 快速开始

### 方式一：一键启动（推荐）

```bash
# 1. 确保 Ollama 在运行
ollama serve

# 2. 启动系统（自动启动后端 8000 + 前端 8081，并打开浏览器）
cd backend
python launcher.py
```

浏览器自动打开 `http://127.0.0.1:8081/index.html`，播放开局视频后进入系统。

### 方式二：手动分步启动（调试用）

```bash
# 终端 1：后端
cd backend
python -m uvicorn main:app --host 127.0.0.1 --port 8000

# 终端 2：前端
cd frontend
python -m http.server 8081
```

### 安装依赖

```bash
conda create -n data_analyst python=3.12 -y
conda activate data_analyst
pip install -r requirements.txt
```

---

## 🔌 API 接口清单

统一响应格式：`{ "code": 0 成功 | -1 失败, "msg": 提示, "data": 数据 }`

### 通用 / 聊天（routes/chat/__init__.py）
| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/models` | 模型列表（固定返回 qwen2.5:7b-instruct-q4_K_M） |
| POST | `/upload_dataset` | 上传数据集（csv/xlsx/xls，多文件，异步任务） |
| GET | `/upload_status` | 上传任务状态轮询 |
| POST | `/chat` | 快撩我问答（大模型） |
| POST | `/chat_stream` | 快撩我流式问答（SSE） |
| POST | `/chat_analyze` | 数据分析问答（大模型） |

### 数据清洗（routes/clean/__init__.py）
| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/clean_dataset` | 清洗主接口：大模型解析意图 → pandas 白名单确定性执行（含失败日志） |
| POST | `/clean_fast` | 快捷清洗：直接传白名单操作 JSON（不调用大模型） |
| GET | `/download_cleaned` | 下载清洗结果（`原数据集名_clean`，下载后自动清除缓存） |
| POST | `/delete_cleaned` | 删除（手动）下载后的清洗缓存 |

### EDA 分析（routes/eda/__init__.py）
| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/eda_plot` | 选项式按需绘图：大模型分析意图 → matplotlib 确定性绘制（含动画帧，多变量/小提琴图） |
| POST | `/eda_feature_report` | 特征分析报告：基于相关性/特征重要性生成 md（可下载 `原数据集名_特征分析报告`） |
| GET | `/download_plot_zip` | 打包下载本轮绘制的图像（zip） |

### 模型训练（routes/train/__init__.py）
| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/train_split` | 三集切分（训练/验证/测试，`原数据集名_train/_val/_test`，返回路径自动加载到模型训练） |
| GET | `/train_algo_params` | 各算法推荐超参（滑动条范围） |
| POST | `/train_code` | 按所选算法生成训练代码（可折叠展示） |
| POST | `/train_run` | 启动后台训练任务（返回 `task_id`，实时产出 loss 点） |
| GET | `/train_progress` | 轮询训练进度（epoch / train_loss / val_loss / 状态） |
| POST | `/train_stop` | 停止训练（保留当前 epoch 结果并保存模型） |
| GET | `/train_list_models` | 列出已训练模型 |
| GET | `/train_download_model` | 下载指定模型（`{算法}_best_loss/last_loss.joblib`，带路径穿越防护） |
| GET | `/train_download` | 下载切分数据集（`原数据集名_train/_val/_test`） |

### 预测分析（routes/predict/__init__.py）
| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/predict` | 训练集训练/测试集测试/验证集验证 或加载已训练模型预测；回归四图、分类混淆矩阵/ROC/特征重要性 |
| POST | `/upload_model` | 上传模型文件（.joblib）到 models 目录，供集成使用（分类/回归均可） |
| POST | `/ensemble_predict` | 分类加权投票集成（软投票）：n 个分类模型 + 可选验证集（Macro-F1 线性归一化权重）→ 概率加权平均预测 |
| POST | `/ensemble_predict_reg` | 回归加权平均集成：n 个回归模型 + 可选验证集（RMSE 误差倒数归一化权重）→ 加权平均预测 |

### 图片数据模块（routes/image/__init__.py）
| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/upload_images` | 上传图片 zip（100M 上限、损坏/加密拒绝、路径穿越防护、损坏图片轻量解码标记） |
| GET | `/image_set_info` | 图片集概览（样本总数、各类别数量、损坏样本列表、分辨率/通道统计、三集分布） |
| GET | `/image_thumb` | 按类别查看样本缩略图 |
| GET | `/image_split` | 数据集切分（训练/验证/测试按比例随机划分，`{set}_train/_val/_test`） |
| POST | `/image_resize` | 统一分辨率（默认 224×224，不改变分类结构） |
| POST | `/img_train` | 启动 ResNet50 训练任务（学习率/批次/轮数/优化器可调，数据增强参数可调，GPU 加速） |
| GET | `/img_train_status` | 图片训练进度轮询（epoch / loss / acc / 日志） |
| POST | `/img_eval` | 模型评估（测试集指标 + 混淆矩阵/分类报告） |
| POST | `/img_eval_report` | 大模型评估报告（可下载 `数据集名_模型评估报告.md`） |
| GET | `/img_file_download` | 下载图片集产物（切分 zip / 评估报告 md） |
| POST | `/img_predict` | 图片预测推理（上传图片，返回类别与概率） |
| GET | `/img_model_info` | 图片模型信息（是否有已训练模型） |
| POST | `/img_model_upload` | 上传 .pt 模型文件供评估/推理 |

---

## 🔒 安全设计

- **模型不执行任意代码**：清洗操作走白名单（`CLEAN_OPS`）后端确定性执行；EDA 绘图由规则解析映射到固定绘图函数
- **路径穿越防护**：文件读取接口仅放行 `uploads/datasets` 与 `uploads/cleaned` 目录；模型下载校验文件名，禁止 `../` 穿越
- **zip 上传安全**：大小上限 100M；损坏 / 加密 zip 直接拒绝；遍历 entry 过滤 `../`、绝对路径阻止路径穿越；跳过 0 字节空文件；图片白名单 jpg/jpeg/png/bmp/tiff/webp，损坏图片轻量解码标记、记录错误列表、不中断整体任务
- **代码安全预检**：`utils.safe_code_precheck` 拦截 `os.`/`subprocess`/`eval(` 等危险关键字
- **自动清理**：程序退出时清空 `models/` 下全部 `.joblib` 模型与图片 `.pt` 模型

---

## ❓ 常见问题

| 现象 | 原因与解决 |
|---|---|
| 提示"请求ollama失败: 连接被拒绝" | Ollama 未启动，先运行 `ollama serve` |
| `/models` 返回空 | 未拉取模型，执行 `ollama pull qwen2.5:7b-instruct-q4_K_M` |
| 前端改动不生效 | 浏览器缓存旧版，`Ctrl+F5` 强制刷新 |
| 端口被占用（8000/8081） | 结束占用进程后重试 |
| 清洗/绘图/训练失败 | 展开对应模块的**日志**查看详细原因（系统已打印可排查日志） |
| 视频不自动进入 | 视频播放结束后自动进入，或点「进入系统」按钮 |
| 上传错误文件后想替换 | 重新上传同目录新文件即可覆盖（已修复首错文件不替换的问题） |
| 图片训练提示未找到模型 | 需先在模型训练中完成训练或上传模型文件 |

---

## 📝 备注

- 系统固定只接入 `qwen2.5:7b-instruct-q4_K_M` 一个模型（`config.DEFAULT_MODEL`），如需更换模型改这一处后重启即可
- 前端为静态单页，改动 `frontend/index.html` 后刷新即生效；后端改动需重启 uvicorn
- 数据文件均保存在本地 `backend/uploads/`，不会上传到任何云端，可随时清空重建
- 文件名统一格式：清洗 `原数据集名_clean`、特征报告 `原数据集名_特征分析报告`、三集 `原数据集名_train/_test/_val`、模型 `{算法}_best_loss/_last_loss`、图片评估报告 `数据集名_模型评估报告`
- 后端代码按功能分文件夹组织（`routes/chat`、`routes/image`、`routes/clean`、`routes/eda`、`routes/train`、`routes/predict`），新增功能请放入对应包并在 `main.py` 显式 import
