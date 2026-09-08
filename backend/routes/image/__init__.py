# -*- coding: utf-8 -*-
"""图片数据模块：zip 图片集上传 / 概览 / 切分 / 统一分辨率 / ResNet50 训练 / 评估 / 推理。"""
import os
import re
import threading
import uuid

import json
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from fastapi import UploadFile, File, Query
from fastapi.responses import JSONResponse, FileResponse
from datetime import datetime

from config import app, upload_folder, DEFAULT_MODEL
from utils import (
    call_ollama, call_ollama_stream, extract_python_code, fix_plot_code, neutralize_df_reload,
    safe_code_precheck,
)

# 图片集 zip 大小上限：100MB
MAX_IMAGE_ZIP_SIZE = 100 * 1024 * 1024

@app.post("/upload_images")
async def upload_images(file: UploadFile):
    import zipfile
    original_name = os.path.basename(file.filename or "")
    if not original_name.lower().endswith(".zip"):
        return {"code": -1, "msg": "仅支持 zip 格式的图片集"}
    safe_zip = re.sub(r'[^a-zA-Z0-9._-]', '_', original_name)
    if len(safe_zip) > 100:
        name, ext = os.path.splitext(safe_zip)
        safe_zip = name[:100] + ext
    # 保存 zip 临时文件
    zip_path = os.path.abspath(os.path.join(upload_folder, safe_zip))
    with open(zip_path, "wb") as f:
        f.write(await file.read())
    # 大小限制：zip 上限 100MB
    if os.path.getsize(zip_path) > MAX_IMAGE_ZIP_SIZE:
        try:
            os.remove(zip_path)
        except OSError:
            pass
        return {"code": -1, "msg": "图片集 zip 超过 100MB 大小限制，请压缩后再上传"}
    # 解压到 image_sets/<zip名>/（防路径穿越 + 仅保留图片）
    set_name, _ = os.path.splitext(safe_zip)
    target = os.path.abspath(os.path.join(upload_folder, "image_sets", set_name))
    os.makedirs(target, exist_ok=True)
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    count, names, errors, files = 0, [], [], []
    try:
        with zipfile.ZipFile(zip_path) as z:
            # 加密检测：任一条目带加密标志（flag bit 0）直接拒绝
            encrypted = [i.filename for i in z.infolist() if (i.flag_bits & 0x1)]
            if encrypted:
                raise ValueError("zip 已加密，请取消密码后重新压缩上传")
            # 预扫描：统计哪些一级目录含直接图片文件（用于识别"外层包装目录"，如 cat_dog_1/cat/xxx.jpg 中的 cat_dog_1）
            top_has_file = set()
            for info in z.infolist():
                raw_name = info.filename
                norm = raw_name.replace("\\", "/")
                parts = [p for p in norm.split("/") if p not in ("", ".")]
                if raw_name.startswith(("/", "\\")) or norm.startswith("/") or re.match(r"^[a-zA-Z]:", raw_name) or ".." in parts:
                    continue
                fname = os.path.basename(norm)
                if not fname or info.file_size == 0:
                    continue
                if os.path.splitext(fname)[1].lower() not in img_exts:
                    continue
                if len(parts) == 2:
                    top_has_file.add(parts[0])
            for info in z.infolist():
                # 路径穿越防护：跳过绝对路径（/ 开头、盘符）与含 .. 的条目
                raw_name = info.filename
                norm = raw_name.replace("\\", "/")
                parts = [p for p in norm.split("/") if p not in ("", ".")]
                if raw_name.startswith(("/", "\\")) or norm.startswith("/") or re.match(r"^[a-zA-Z]:", raw_name) or ".." in parts:
                    continue
                fname = os.path.basename(norm)
                if not fname:
                    continue
                # 跳过 0 字节空文件
                if info.file_size == 0:
                    continue
                ext = os.path.splitext(fname)[1].lower()
                if ext not in img_exts:
                    continue
                # 类别标签：一级目录含直接图片 → 取一级（cat/xxx.jpg → cat）；
                # 一级目录只是外层包装（cat_dog_1/cat/xxx.jpg，cat_dog_1 下无直接图片）→ 取二级目录为类别；
                # 平铺 → 类别空串
                if len(parts) >= 2:
                    if parts[0] in top_has_file or len(parts) == 2:
                        cls = parts[0]
                    else:
                        cls = parts[1]
                else:
                    cls = ""
                base, e = os.path.splitext(fname)
                out_dir = os.path.join(target, cls) if cls else target
                os.makedirs(out_dir, exist_ok=True)
                out = os.path.join(out_dir, fname)
                n = 1
                while os.path.exists(out):
                    out = os.path.join(out_dir, f"{base}_{n}{e}")
                    n += 1
                with z.open(info) as src, open(out, "wb") as dst:
                    dst.write(src.read())
                # 轻量解码校验：识别损坏图片（只读头部不加载全图），失败则标记并跳过，不中断整体任务
                try:
                    with Image.open(out) as im:
                        w, h, mode = im.size[0], im.size[1], im.mode
                        im.verify()
                except Exception as dec_err:
                    errors.append({"file": fname, "reason": f"图片损坏或无法解码（{type(dec_err).__name__}）"})
                    try:
                        os.remove(out)
                    except OSError:
                        pass
                    continue
                count += 1
                names.append(os.path.basename(out))
                rel_path = os.path.join(cls, os.path.basename(out)) if cls else os.path.basename(out)
                files.append({"path": rel_path.replace("\\", "/"), "cls": cls, "w": w, "h": h, "mode": mode})
    except zipfile.BadZipFile:
        os.remove(zip_path)
        return {"code": -1, "msg": "zip 文件损坏或格式错误"}
    except ValueError as e:
        os.remove(zip_path)
        return {"code": -1, "msg": str(e)}
    except Exception as e:
        os.remove(zip_path)
        return {"code": -1, "msg": f"解压失败: {str(e)}"}
    finally:
        if os.path.exists(zip_path):
            os.remove(zip_path)
    if count == 0:
        return {"code": -1, "msg": "zip 内未找到有效图片（支持 jpg/jpeg/png/bmp/tiff/webp）"}
    # 数据集元数据入库：样本总数、各类别数量、损坏样本列表、文件清单
    classes = {}
    for it in files:
        classes[it["cls"]] = classes.get(it["cls"], 0) + 1
    meta = {"name": set_name, "total": count, "classes": classes, "errors": errors, "files": files}
    meta_path = os.path.join(target, "meta.json")
    try:
        with open(meta_path, "w", encoding="utf-8") as mf:
            json.dump(meta, mf, ensure_ascii=False, indent=2)
    except Exception as e:
        return {"code": -1, "msg": f"元数据入库失败: {str(e)}"}
    if errors:
        msg = f"图片集上传成功：{count} 张，{len(errors)} 张损坏已跳过"
    else:
        msg = f"图片集上传成功：{count} 张"
    return {"code": 0, "msg": msg,
            "data": {"count": count, "names": names, "dir": target, "errors": errors,
                     "classes": classes, "total": count, "meta_path": meta_path}}

# 图片数据集管理：元数据查询 / 缩略图预览 / 训练验证测试集划分
def _image_set_dir(set_name: str):
    """校验图片集目录路径安全，返回目录绝对路径。"""
    set_dir = os.path.abspath(os.path.join(upload_folder, "image_sets", set_name or ""))
    if not os.path.isdir(set_dir):
        raise ValueError("图片集不存在，请先上传 zip 图片集")
    return set_dir

@app.get("/image_set_info")
async def image_set_info(set_name: str = Query()):
    """返回图片集元数据：样本总数、类别分布、损坏样本列表、文件清单。"""
    try:
        set_dir = _image_set_dir(set_name)
    except ValueError as e:
        return {"code": -1, "msg": str(e)}
    meta_path = os.path.join(set_dir, "meta.json")
    if not os.path.exists(meta_path):
        return {"code": -1, "msg": "该图片集无元数据（可能上传时未生成），请重新上传"}
    try:
        with open(meta_path, "r", encoding="utf-8") as mf:
            meta = json.load(mf)
    except Exception as e:
        return {"code": -1, "msg": f"读取元数据失败: {str(e)}"}
    # 懒扫描补齐：旧版元数据可能缺 w/h/mode，现场读取并写回
    files_meta = meta.get("files") or []
    if any(("w" not in it or "h" not in it) for it in files_meta):
        for it in files_meta:
            p = os.path.join(set_dir, it["path"])
            try:
                with Image.open(p) as im:
                    it["w"], it["h"], it["mode"] = im.size[0], im.size[1], im.mode
            except Exception:
                it.setdefault("w", 0); it.setdefault("h", 0); it.setdefault("mode", "?")
        try:
            with open(meta_path, "w", encoding="utf-8") as mf:
                json.dump(meta, mf, ensure_ascii=False, indent=2)
        except Exception:
            pass
    return {"code": 0, "msg": "ok", "data": meta}

@app.get("/image_thumb")
async def image_thumb(set_name: str = Query(), rel_path: str = Query()):
    """生成并返回图片缩略图（160px，缓存于 _thumbs 目录）。"""
    try:
        set_dir = _image_set_dir(set_name)
    except ValueError as e:
        return JSONResponse({"code": -1, "msg": str(e)}, status_code=404)
    abs_path = os.path.abspath(os.path.join(set_dir, rel_path or ""))
    if not abs_path.startswith(set_dir) or not os.path.isfile(abs_path):
        return JSONResponse({"code": -1, "msg": "图片不存在"}, status_code=404)
    cache_dir = os.path.join(set_dir, "_thumbs")
    os.makedirs(cache_dir, exist_ok=True)
    thumb = os.path.join(cache_dir, rel_path.replace("/", "_").replace("\\", "_"))
    if not os.path.exists(thumb):
        try:
            with Image.open(abs_path) as im:
                im.thumbnail((160, 160))
                im.convert("RGB").save(thumb, "JPEG", quality=75)
        except Exception:
            return JSONResponse({"code": -1, "msg": "缩略图生成失败"}, status_code=500)
    return FileResponse(thumb, media_type="image/jpeg")

@app.get("/image_split")
async def image_split(
    set_name: str = Query(),
    train: float = Query(default=0.7),
    val: float = Query(default=0.15),
    test: float = Query(default=0.15),
):
    """按比例随机划分图片数据集为训练/验证/测试集（按类别分层抽样）。
    划分清单写入 image_sets/<set_name>/split/{train,val,test}.txt。"""
    import random
    try:
        set_dir = _image_set_dir(set_name)
    except ValueError as e:
        return {"code": -1, "msg": str(e)}
    meta_path = os.path.join(set_dir, "meta.json")
    if not os.path.exists(meta_path):
        return {"code": -1, "msg": "该图片集无元数据，请重新上传"}
    try:
        with open(meta_path, "r", encoding="utf-8") as mf:
            meta = json.load(mf)
    except Exception as e:
        return {"code": -1, "msg": f"读取元数据失败: {str(e)}"}
    files = meta.get("files") or []
    if not files:
        return {"code": -1, "msg": "图片集为空，无法划分"}
    t = float(train); v = float(val); te = float(test)
    if t <= 0 or v < 0 or te < 0 or abs((t + v + te) - 1.0) > 0.01:
        return {"code": -1, "msg": "划分比例无效（三个比例之和应为 1，训练集比例需 > 0）"}
    # 按类别分层抽样
    by_cls = {}
    for it in files:
        by_cls.setdefault(it.get("cls", ""), []).append(it["path"])
    train_list, val_list, test_list = [], [], []
    for cls, paths in by_cls.items():
        random.shuffle(paths)
        n = len(paths)
        n_train = int(round(n * t))
        n_val = int(round(n * v))
        if n_train + n_val > n:
            n_val = n - n_train
        n_test = n - n_train - n_val
        train_list.extend(paths[:n_train])
        val_list.extend(paths[n_train:n_train + n_val])
        test_list.extend(paths[n_train + n_val:])
    split_dir = os.path.join(set_dir, "split")
    os.makedirs(split_dir, exist_ok=True)
    for name, lst in (("train", train_list), ("val", val_list), ("test", test_list)):
        with open(os.path.join(split_dir, name + ".txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lst))
    def _cls_of(p):
        return p.split("/")[0] if "/" in p else ""
    def _count_cls(lst):
        d = {}
        for p in lst:
            k = _cls_of(p)
            d[k] = d.get(k, 0) + 1
        return d
    return {
        "code": 0,
        "msg": "数据集划分完成",
        "data": {
            "train": len(train_list), "val": len(val_list), "test": len(test_list),
            "total": len(files),
            "split_dir": split_dir,
            "files": {"train": train_list, "val": val_list, "test": test_list},
            "by_cls": {
                "train": _count_cls(train_list),
                "val": _count_cls(val_list),
                "test": _count_cls(test_list),
            },
        },
    }

@app.post("/image_resize")
async def image_resize(
    set_name: str = Query(),
    width: int = Query(default=224),
    height: int = Query(default=224),
):
    """将图片集全部图片统一缩放为指定分辨率（默认 224x224），覆盖原文件并更新元数据。"""
    import shutil
    try:
        set_dir = _image_set_dir(set_name)
    except ValueError as e:
        return {"code": -1, "msg": str(e)}
    if width < 8 or width > 2048 or height < 8 or height > 2048:
        return {"code": -1, "msg": "目标尺寸需在 8~2048 之间"}
    meta_path = os.path.join(set_dir, "meta.json")
    if not os.path.exists(meta_path):
        return {"code": -1, "msg": "该图片集无元数据，请重新上传"}
    try:
        with open(meta_path, "r", encoding="utf-8") as mf:
            meta = json.load(mf)
    except Exception as e:
        return {"code": -1, "msg": f"读取元数据失败: {str(e)}"}
    files_meta = meta.get("files") or []
    if not files_meta:
        return {"code": -1, "msg": "图片集为空，无法统一分辨率"}
    resized, fail_l = 0, []
    for it in files_meta:
        p = os.path.join(set_dir, it["path"])
        try:
            with Image.open(p) as im:
                mode = im.mode
                im = im.resize((width, height), Image.LANCZOS)
                try:
                    im.save(p)  # 保持原通道模式（L/RGB/RGBA 等）与格式不变
                except Exception:
                    # 少数格式不支持原模式保存（如 CMYK→jpg），回退 RGB
                    im = im.convert("RGB")
                    im.save(p)
                    mode = "RGB"
            it["w"], it["h"] = width, height
            it["mode"] = mode
            resized += 1
        except Exception as e:
            fail_l.append({"file": it["path"], "reason": str(e)})
    try:
        with open(meta_path, "w", encoding="utf-8") as mf:
            json.dump(meta, mf, ensure_ascii=False, indent=2)
    except Exception as e:
        return {"code": -1, "msg": f"元数据写回失败: {str(e)}"}
    # 清缩略图缓存，强制重新生成
    thumb_dir = os.path.join(set_dir, "_thumbs")
    if os.path.isdir(thumb_dir):
        shutil.rmtree(thumb_dir, ignore_errors=True)
    msg = f"已统一为 {width}x{height}：{resized} 张"
    if fail_l:
        msg += f"，{len(fail_l)} 张失败"
    return {"code": 0, "msg": msg, "data": {"resized": resized, "errors": fail_l}}

# 图片数据集训练任务表：task_id -> {"status": "running"|"done"|"error", ...}
IMG_TRAIN_TASKS = {}

def _img_train_worker(task_id: str, set_name: str, lr: float, batch_size: int, epochs: int, optimizer_name: str,
                      aug_enabled: int, aug_flip_p: float, aug_rotation: float, aug_crop_min: float,
                      aug_brightness: float, aug_contrast: float, aug_saturation: float, aug_blur_sigma: float):
    """ResNet50 图片分类训练（异步后台线程，实时写回进度供前端轮询）。"""
    try:
        import torch
        from torch import nn
        from torchvision import models, transforms

        set_dir = _image_set_dir(set_name)
        split_dir = os.path.join(set_dir, "split")
        for n in ("train", "val", "test"):
            if not os.path.exists(os.path.join(split_dir, n + ".txt")):
                raise ValueError(f"缺少 {n}.txt 划分文件，请先在数据集分析中完成数据集划分")

        def _read_lst(n):
            with open(os.path.join(split_dir, n + ".txt"), encoding="utf-8") as f:
                return [ln.strip() for ln in f if ln.strip()]

        train_paths = _read_lst("train")
        val_paths = _read_lst("val")
        test_paths = _read_lst("test")
        if not train_paths or not val_paths:
            raise ValueError("训练集/验证集为空，请重新划分")

        def _cls_of(p):
            return p.split("/")[0] if "/" in p else ""

        cls_names = sorted({_cls_of(p) for p in train_paths + val_paths + test_paths})
        if len(cls_names) < 2:
            raise ValueError("类别数不足 2，无法分类训练，请确认数据集按类别目录组织")
        cls2idx = {c: i for i, c in enumerate(cls_names)}
        num_classes = len(cls_names)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 验证/测试集仅缩放不做增强，保证评估一致性
        eval_tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        # 训练集在线数据增强：每个 epoch 读取训练图片时实时动态应用（参数默认，用户可改，可关闭）
        if aug_enabled:
            train_tf = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(aug_crop_min, 1.0)),
                transforms.RandomHorizontalFlip(p=aug_flip_p),
                transforms.RandomRotation(aug_rotation),
                transforms.ColorJitter(brightness=aug_brightness, contrast=aug_contrast, saturation=aug_saturation),
                transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, aug_blur_sigma)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        else:
            train_tf = eval_tf

        class _ImgDS(torch.utils.data.Dataset):
            def __init__(self, paths, tfm):
                self.paths = paths
                self.tfm = tfm
            def __len__(self):
                return len(self.paths)
            def __getitem__(self, i):
                p = self.paths[i]
                img = Image.open(os.path.join(set_dir, p)).convert("RGB")
                return self.tfm(img), cls2idx[_cls_of(p)]

        dl_tr = torch.utils.data.DataLoader(_ImgDS(train_paths, train_tf), batch_size=batch_size, shuffle=True)
        dl_va = torch.utils.data.DataLoader(_ImgDS(val_paths, eval_tf), batch_size=batch_size, shuffle=False)
        dl_te = torch.utils.data.DataLoader(_ImgDS(test_paths, eval_tf), batch_size=batch_size, shuffle=False) if test_paths else None

        # ResNet50（ImageNet 结构，随机初始化；输出层改为类别数）
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.to(device)

        opt_map = {"adamw": torch.optim.AdamW, "adam": torch.optim.Adam, "sgd": torch.optim.SGD}
        opt = opt_map[optimizer_name](model.parameters(), lr=lr)
        crit = nn.CrossEntropyLoss()

        def _eval(dl):
            model.eval()
            tot, nb, correct, total = 0.0, 0, 0, 0
            with torch.no_grad():
                for xb, yb in dl:
                    out = model(xb.to(device))
                    tot += crit(out, yb.to(device)).item()
                    correct += (out.argmax(1).cpu() == yb).sum().item()
                    total += yb.size(0)
                    nb += 1
            return tot / max(1, nb), correct / max(1, total)

        history = {"epoch": [], "train_loss": [], "val_loss": [], "val_acc": []}
        if aug_enabled:
            logs = [f"数据增强：已开启（水平翻转p={aug_flip_p} / 旋转{aug_rotation}° / 裁剪scale=({aug_crop_min},1.0) / 色彩抖动b={aug_brightness},c={aug_contrast},s={aug_saturation} / 高斯模糊sigma=(0.1,{aug_blur_sigma})）"]
        else:
            logs = ["数据增强：已关闭（训练集仅缩放）"]
        best_acc = 0.0
        for ep in range(1, epochs + 1):
            model.train()
            tot, nb = 0.0, 0
            for xb, yb in dl_tr:
                opt.zero_grad()
                out = model(xb.to(device))
                loss = crit(out, yb.to(device))
                loss.backward()
                opt.step()
                tot += loss.item()
                nb += 1
            tl = round(tot / max(1, nb), 4)
            vl, va = _eval(dl_va)
            vl, va = round(vl, 4), round(va, 4)
            history["epoch"].append(ep)
            history["train_loss"].append(tl)
            history["val_loss"].append(vl)
            history["val_acc"].append(va)
            if ep == 1 or ep % 5 == 0 or ep == epochs:
                logs.append(f"Epoch {ep}/{epochs} | train_loss {tl:.4f} | val_loss {vl:.4f} | val_acc {va:.4f}")
            tsk = IMG_TRAIN_TASKS.get(task_id)
            if tsk is not None:
                tsk["current_epoch"] = ep
                tsk["train_loss"] = tl
                tsk["val_loss"] = vl
                tsk["val_acc"] = va
                tsk["history"] = dict(history)
                tsk["log"] = list(logs)
            if va > best_acc:
                best_acc = va
                torch.save({"model": model.state_dict(), "classes": cls_names},
                           os.path.join(set_dir, "best_model.pt"))
            torch.save({"model": model.state_dict(), "classes": cls_names},
                       os.path.join(set_dir, "last_model.pt"))
        test_metrics = None
        if dl_te is not None:
            t_loss, t_acc = _eval(dl_te)
            test_metrics = {"test_loss": round(t_loss, 4), "test_acc": round(t_acc, 4)}
        tsk = IMG_TRAIN_TASKS.get(task_id)
        if tsk is not None:
            tsk["status"] = "done"
            tsk["done"] = True
            tsk["best_val_acc"] = round(best_acc, 4)
            tsk["test_metrics"] = test_metrics
            tsk["classes"] = cls_names
            tsk["model_dir"] = set_dir
            tsk["device"] = str(device)
    except Exception as e:
        tsk = IMG_TRAIN_TASKS.get(task_id)
        if tsk is not None:
            tsk["status"] = "error"
            tsk["error"] = str(e)
            tsk["done"] = True

@app.post("/img_train")
async def img_train(
    set_name: str = Query(),
    learning_rate: float = Query(default=1e-4),
    batch_size: int = Query(default=8),
    epochs: int = Query(default=20),
    optimizer: str = Query(default="adamw"),
    aug_enabled: int = Query(default=1),
    aug_flip_p: float = Query(default=0.5),
    aug_rotation: float = Query(default=10.0),
    aug_crop_min: float = Query(default=0.8),
    aug_brightness: float = Query(default=0.2),
    aug_contrast: float = Query(default=0.2),
    aug_saturation: float = Query(default=0.1),
    aug_blur_sigma: float = Query(default=1.0),
):
    """启动 ResNet50 图片分类训练（异步任务）。数据来自数据集分析模块划分好的 train/val/test。"""
    try:
        set_dir = _image_set_dir(set_name)
    except ValueError as e:
        return {"code": -1, "msg": str(e)}
    split_dir = os.path.join(set_dir, "split")
    if not os.path.isdir(split_dir):
        return {"code": -1, "msg": "尚未划分数据集，请先在数据集分析中完成训练/验证/测试集划分"}
    if optimizer.lower() not in ("adamw", "adam", "sgd"):
        return {"code": -1, "msg": "优化器仅支持 adamw / adam / sgd"}
    if batch_size < 1 or epochs < 1 or learning_rate <= 0:
        return {"code": -1, "msg": "超参数不合法"}
    task_id = "imgtrain_" + uuid.uuid4().hex[:12]
    IMG_TRAIN_TASKS[task_id] = {
        "status": "running", "done": False, "current_epoch": 0, "total_epochs": epochs,
        "set_name": set_name, "learning_rate": learning_rate, "batch_size": batch_size,
        "optimizer": optimizer.lower(), "history": {},
        "augmentation": bool(aug_enabled),
        "aug_params": {"flip_p": aug_flip_p, "rotation": aug_rotation, "crop_min": aug_crop_min,
                       "brightness": aug_brightness, "contrast": aug_contrast, "saturation": aug_saturation,
                       "blur_sigma": aug_blur_sigma},
    }
    threading.Thread(target=_img_train_worker,
                     args=(task_id, set_name, learning_rate, batch_size, epochs, optimizer.lower(),
                           aug_enabled, aug_flip_p, aug_rotation, aug_crop_min,
                           aug_brightness, aug_contrast, aug_saturation, aug_blur_sigma),
                     daemon=True).start()
    return {"code": 0, "msg": "训练任务已启动", "data": {"task_id": task_id, "total_epochs": epochs}}

@app.get("/img_train_status")
async def img_train_status(task_id: str = Query()):
    """查询图片训练任务进度（前端轮询）。"""
    tsk = IMG_TRAIN_TASKS.get(task_id)
    if tsk is None:
        return {"code": -1, "msg": "任务不存在"}
    return {"code": 0, "data": tsk}

# 图片分类模型评估：加载训练产出的 best/last 模型，在测试集上评估
def _img_load_model(set_dir: str, model_name: str = None):
    import torch
    from torch import nn
    from torchvision import models
    names = [model_name] if model_name else ("best_model.pt", "last_model.pt")
    for name in names:
        p = os.path.join(set_dir, name)
        if os.path.exists(p):
            ckpt = torch.load(p, map_location="cpu")
            num_classes = len(ckpt.get("classes", []))
            model = models.resnet50(weights=None)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            model.load_state_dict(ckpt["model"])
            return model, ckpt.get("classes", [])
    raise ValueError("未找到已训练模型，请先在模型训练中完成训练")

@app.post("/img_eval")
async def img_eval(set_name: str = Query(), model: str = Query(None)):
    """在测试集上评估图片分类模型：准确率 / 每类 P·R·F1 / 混淆矩阵。"""
    try:
        import torch
        from torch import nn
        from torchvision import transforms
        set_dir = _image_set_dir(set_name)
    except ValueError as e:
        return {"code": -1, "msg": str(e)}
    test_txt = os.path.join(set_dir, "split", "test.txt")
    if not os.path.exists(test_txt):
        return {"code": -1, "msg": "缺少测试集划分文件，请先在数据集分析中划分数据集"}
    try:
        with open(test_txt, encoding="utf-8") as f:
            test_paths = [ln.strip() for ln in f if ln.strip()]
    except Exception as e:
        return {"code": -1, "msg": f"读取测试集失败: {str(e)}"}
    if not test_paths:
        return {"code": -1, "msg": "测试集为空"}
    try:
        model, cls_names = _img_load_model(set_dir, model)
    except Exception as e:
        return {"code": -1, "msg": str(e)}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    cls2idx = {c: i for i, c in enumerate(cls_names)}
    try:
        X, y = [], []
        for p in test_paths:
            cls = p.replace("\\", "/").split("/")[0] if ("/" in p or "\\" in p) else ""
            if cls not in cls2idx:
                continue
            img = Image.open(os.path.join(set_dir, p)).convert("RGB")
            X.append(tf(img))
            y.append(cls2idx[cls])
        if not X:
            return {"code": -1, "msg": "测试集样本无法解析"}
        Xt = torch.stack(X)
        yt = torch.tensor(y, dtype=torch.long)
        model.eval()
        all_p = []
        with torch.no_grad():
            for i in range(0, Xt.size(0), 16):
                out = model(Xt[i:i + 16].to(device))
                all_p.extend(out.argmax(1).cpu().tolist())
        n = len(cls_names)
        cm = [[0] * n for _ in range(n)]
        for yy, pp in zip(y, all_p):
            cm[yy][pp] += 1
        total = len(y)
        acc = sum(cm[i][i] for i in range(n)) / max(1, total)
        per = []
        for i in range(n):
            tp = cm[i][i]
            fp = sum(cm[j][i] for j in range(n)) - tp
            fn = sum(cm[i]) - tp
            prec = tp / max(1, tp + fp)
            rec = tp / max(1, tp + fn)
            f1 = 2 * prec * rec / max(1e-9, prec + rec)
            per.append({"class": cls_names[i], "support": sum(cm[i]),
                        "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)})
        return {"code": 0, "msg": f"评估完成：准确率 {acc * 100:.2f}%",
                "data": {"accuracy": round(acc, 4), "confusion_matrix": cm, "per_class": per,
                         "classes": cls_names, "total": total, "device": str(device)}}
    except Exception as e:
        return {"code": -1, "msg": f"评估失败: {str(e)}"}

@app.post("/img_eval_report")
async def img_eval_report(set_name: str = Query(), model: str = Query(None)):
    """模型评估报告：大模型基于评估指标生成 Markdown 分析报告，并保存为可下载 md 文件。"""
    try:
        import torch
        from torch import nn
        from torchvision import transforms
        set_dir = _image_set_dir(set_name)
    except ValueError as e:
        return {"code": -1, "msg": str(e)}
    test_txt = os.path.join(set_dir, "split", "test.txt")
    if not os.path.exists(test_txt):
        return {"code": -1, "msg": "缺少测试集划分文件，请先划分数据集"}
    try:
        with open(test_txt, encoding="utf-8") as f:
            test_paths = [ln.strip() for ln in f if ln.strip()]
    except Exception as e:
        return {"code": -1, "msg": f"读取测试集失败: {str(e)}"}
    if not test_paths:
        return {"code": -1, "msg": "测试集为空"}
    try:
        model_obj, cls_names = _img_load_model(set_dir, model)
    except Exception as e:
        return {"code": -1, "msg": str(e)}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_obj.to(device)
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    cls2idx = {c: i for i, c in enumerate(cls_names)}
    try:
        X, y = [], []
        for p in test_paths:
            cls = p.replace("\\", "/").split("/")[0] if ("/" in p or "\\" in p) else ""
            if cls not in cls2idx:
                continue
            img = Image.open(os.path.join(set_dir, p)).convert("RGB")
            X.append(tf(img))
            y.append(cls2idx[cls])
        if not X:
            return {"code": -1, "msg": "测试集样本无法解析"}
        Xt = torch.stack(X)
        yt = torch.tensor(y, dtype=torch.long)
        model_obj.eval()
        all_p = []
        with torch.no_grad():
            for i in range(0, Xt.size(0), 16):
                out = model_obj(Xt[i:i + 16].to(device))
                all_p.extend(out.argmax(1).cpu().tolist())
        n = len(cls_names)
        cm = [[0] * n for _ in range(n)]
        for yy, pp in zip(y, all_p):
            cm[yy][pp] += 1
        total = len(y)
        acc = sum(cm[i][i] for i in range(n)) / max(1, total)
        per = []
        for i in range(n):
            tp = cm[i][i]
            fp = sum(cm[j][i] for j in range(n)) - tp
            fn = sum(cm[i]) - tp
            prec = tp / max(1, tp + fp)
            rec = tp / max(1, tp + fn)
            f1 = 2 * prec * rec / max(1e-9, prec + rec)
            per.append({"class": cls_names[i], "support": sum(cm[i]),
                        "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)})
    except Exception as e:
        return {"code": -1, "msg": f"评估失败: {str(e)}"}
    # 组装评估摘要，交给大模型生成报告
    cm_txt = "\n".join(["| " + " | ".join(map(str, row)) + " |" for row in cm])
    per_txt = "\n".join([
        f"- {p['class']}：样本数 {p['support']}，精确率 {p['precision']:.4f}，召回率 {p['recall']:.4f}，F1 {p['f1']:.4f}"
        for p in per])
    prompt = f"""你是资深机器学习工程师，请基于以下图片分类模型在测试集上的真实评估结果，生成一份专业的中文评估分析报告（Markdown 格式）。

数据集名称：{set_name}
类别：{", ".join(cls_names)}（共 {n} 类）
测试集样本总数：{total}
整体准确率：{acc * 100:.2f}%

每类指标：
{per_txt}

混淆矩阵（行=真实类别，列=预测类别）：
|类别|{'|'.join(cls_names)}|
|---|---|
{cm_txt}

请输出 Markdown 报告，包含以下章节：
1. 评估概览（整体准确率、测试集规模、类别数）
2. 分类结果分析（混淆矩阵解读：哪些类别容易混淆）
3. 每类指标分析（精确率/召回率/F1 解读，指出表现最差与最好的类别）
4. 潜在问题（类别不平衡、样本不足、模型泛化等）
5. 改进建议（数据、模型、训练策略三个方向）

只输出报告正文，不要额外解释。"""
    try:
        res = call_ollama(prompt, model_name=DEFAULT_MODEL, stream=False, temperature=0.3, num_predict=2000)
    except Exception as e:
        return {"code": -1, "msg": f"大模型生成失败: {str(e)}"}
    if "error" in res:
        return {"code": -1, "msg": res["error"]}
    report = res.get("response", "").strip()
    if not report:
        return {"code": -1, "msg": "大模型未返回报告内容"}
    # 保存 md 文件：数据集名_模型评估报告.md
    report_dir = os.path.join(set_dir, "reports")
    os.makedirs(report_dir, exist_ok=True)
    fname = f"{set_name}模型评估报告.md"
    fpath = os.path.join(report_dir, fname)
    try:
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(f"# {set_name} 模型评估报告\n\n")
            f.write(f"> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ｜ 模型：{model or 'best_model.pt'} ｜ 设备：{device}\n\n")
            f.write(report)
    except Exception as e:
        return {"code": -1, "msg": f"报告保存失败: {str(e)}"}
    return {"code": 0, "msg": "评估报告生成完成",
            "data": {"report": report, "filename": fname}}

@app.get("/img_file_download")
async def img_file_download(set_name: str = Query(), file: str = Query()):
    """下载图片数据集目录内的文件（报告 / 模型等），限制在数据集目录内，防路径穿越。"""
    try:
        set_dir = os.path.realpath(_image_set_dir(set_name))
    except ValueError as e:
        return {"code": -1, "msg": str(e)}
    fp = os.path.realpath(os.path.join(set_dir, file))
    if not fp.startswith(set_dir + os.sep):
        return {"code": -1, "msg": "非法文件路径"}
    if not os.path.isfile(fp):
        return {"code": -1, "msg": "文件不存在"}
    return FileResponse(fp, filename=os.path.basename(fp))

@app.post("/img_predict")
async def img_predict(set_name: str = Query(), model: str = Query(None), files: list[UploadFile] = File(...)):
    """用已训练模型批量预测上传图片：返回每张图 Top-3 类别与置信度。"""
    try:
        import torch
        from torch import nn
        from torchvision import transforms
        set_dir = _image_set_dir(set_name)
    except ValueError as e:
        return {"code": -1, "msg": str(e)}
    if not files:
        return {"code": -1, "msg": "请上传待预测图片"}
    try:
        model, cls_names = _img_load_model(set_dir, model)
    except Exception as e:
        return {"code": -1, "msg": str(e)}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    results = []
    for f in files:
        fname = os.path.basename(f.filename or "unknown")
        try:
            data = await f.read()
            from io import BytesIO
            img = Image.open(BytesIO(data)).convert("RGB")
            x = tf(img).unsqueeze(0)
            with torch.no_grad():
                prob = torch.softmax(model(x.to(device))[0], dim=0).cpu().tolist()
            idx_sorted = sorted(range(len(prob)), key=lambda k: prob[k], reverse=True)
            top3 = [{"class": cls_names[i], "conf": round(prob[i], 4)} for i in idx_sorted[:3]]
            results.append({"file": fname, "top1": cls_names[idx_sorted[0]],
                            "conf": round(prob[idx_sorted[0]], 4), "top3": top3})
        except Exception as e:
            results.append({"file": fname, "error": f"预测失败（{type(e).__name__}）"})
    return {"code": 0, "msg": f"预测完成 {len(results)} 张", "data": {"results": results, "classes": cls_names}}

# 图片模型信息查询：列出当前图片集已保存的模型
@app.get("/img_model_info")
def img_model_info(set_name: str = Query()):
    try:
        set_dir = _image_set_dir(set_name)
    except ValueError as e:
        return {"code": -1, "msg": str(e)}
    models = sorted([f for f in os.listdir(set_dir) if f.endswith(".pt")])
    default = "best_model.pt" if "best_model.pt" in models else (models[0] if models else "")
    test_count = 0
    test_txt = os.path.join(set_dir, "split", "test.txt")
    if os.path.exists(test_txt):
        with open(test_txt, encoding="utf-8") as f:
            test_count = len([ln for ln in f if ln.strip()])
    return {"code": 0, "data": {"models": models, "default": default, "has_model": bool(models), "test_count": test_count}}

# 上传用户模型文件（.pt）到当前图片集，用于评估 / 预测
@app.post("/img_model_upload")
async def img_model_upload(set_name: str = Query(), model_file: UploadFile = File(...)):
    try:
        set_dir = _image_set_dir(set_name)
    except ValueError as e:
        return {"code": -1, "msg": str(e)}
    fn = os.path.basename(model_file.filename or "user_model.pt")
    if not fn.endswith(".pt"):
        return {"code": -1, "msg": "仅支持 .pt 模型文件"}
    data = await model_file.read()
    if not data:
        return {"code": -1, "msg": "模型文件为空"}
    target = os.path.join(set_dir, "user_model.pt")
    with open(target, "wb") as f:
        f.write(data)
    try:
        _img_load_model(set_dir, "user_model.pt")
    except Exception as e:
        try:
            os.remove(target)
        except OSError:
            pass
        return {"code": -1, "msg": f"模型文件无效：{e}"}
    return {"code": 0, "msg": f"模型已加载：{fn}", "data": {"model": "user_model.pt"}}

# 数据分析问答接口
