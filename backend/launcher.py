# -*- coding: utf-8 -*-
"""一键启动器：同时启动后端服务(8000)与前端页面(8081)，并自动打开浏览器。

双击 exe / 运行本脚本即可，无需手动敲命令。
按 Ctrl+C 停止服务，程序结束后自动清理 best_loss / last_loss 模型。
"""
import os
import sys
import time
import threading
import webbrowser

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def get_frontend_dir():
    """前端页面目录：打包后从 exe 资源(_MEIPASS)读取；源码运行时读项目 frontend。"""
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            fe = os.path.join(meipass, "frontend")
            if os.path.isdir(fe):
                return fe
        fe = os.path.join(os.path.dirname(sys.executable), "frontend")
        if os.path.isdir(fe):
            return fe
    for cand in (os.path.join(_HERE, "..", "frontend"),
                 os.path.join(_HERE, "frontend"),
                 os.path.join(os.getcwd(), "frontend")):
        if os.path.isdir(cand):
            return os.path.abspath(cand)
    return os.path.join(_HERE, "frontend")


def serve_frontend(port=8081):
    """在后台线程提供前端静态页面服务。"""
    import functools
    import http.server
    fe = get_frontend_dir()
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=fe)
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    httpd.serve_forever()


def open_browser(url, delay=2.5):
    time.sleep(delay)
    try:
        webbrowser.open(url)
    except Exception:
        pass


def main():
    import uvicorn
    from config import app
    # 显式导入路由模块，完成接口注册
    import routes.chat      # noqa: F401
    import routes.image     # noqa: F401
    import routes.clean     # noqa: F401
    import routes.eda       # noqa: F401
    import routes.predict   # noqa: F401
    import routes.train     # noqa: F401

    # 前端静态服务线程
    fe_dir = get_frontend_dir()
    threading.Thread(target=serve_frontend, daemon=True).start()

    # 自动打开浏览器
    threading.Thread(target=open_browser, args=("http://127.0.0.1:8081/index.html",), daemon=True).start()

    print("=" * 56)
    print("  数据集分析系统 已启动")
    print(f"  前端页面 : http://127.0.0.1:8081  （{fe_dir}）")
    print("  后端接口 : http://127.0.0.1:8000")
    print("  浏览器将自动打开；按 Ctrl+C 停止服务")
    print("=" * 56)

    try:
        uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
    except KeyboardInterrupt:
        pass
    finally:
        print("\n服务已停止。")


if __name__ == "__main__":
    main()
