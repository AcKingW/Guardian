"""
Guardian 监控管理 GUI（Tkinter）。
提供添加/删除监控视频源、一键启动/停止多路检测的图形界面。
"""
import fileinput
import os
import sys
import tkinter as tk
from tkinter import messagebox

from config import VIDEO_INFO_DIR, URL_ADDRESS_FILE, ID_NAME_FILE
from detector.fall_detector import FallDetector


class MonitorApp:
    """
    监控管理桌面应用。

    布局:
      - 输入区：ID 名称 / URL 地址 / 删除名称
      - 操作按钮：添加 / 运行 / 删除 / 停止
      - 显示区：ID 列表 / URL 列表
    """

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title('Guardian - 乘客跌倒报警系统')
        self.root.geometry('580x520')

        self.detector = FallDetector()

        # 确保配置目录存在
        os.makedirs(VIDEO_INFO_DIR, exist_ok=True)

        self._build_ui()
        self._load_existing_sources()

    # ─────────────────────────────────────────
    # UI 构建
    # ─────────────────────────────────────────

    def _build_ui(self):
        """构建所有界面控件。"""
        pad = {'padx': 8, 'pady': 4}

        # ── 输入标签 ──────────────────────────
        tk.Label(self.root, text='监控 ID 名称:').place(x=10, y=10)
        tk.Label(self.root, text='视频流 URL:').place(x=10, y=40)
        tk.Label(self.root, text='待删除 ID:').place(x=10, y=70)
        tk.Label(self.root, text='ID 列表').place(x=20, y=160)
        tk.Label(self.root, text='URL 列表').place(x=260, y=160)

        # ── 输入框 ────────────────────────────
        self.var_name    = tk.StringVar()
        self.var_url     = tk.StringVar()
        self.var_del     = tk.StringVar()
        self.ent_name    = tk.Entry(self.root, textvariable=self.var_name,  width=30)
        self.ent_url     = tk.Entry(self.root, textvariable=self.var_url,   width=30)
        self.ent_del     = tk.Entry(self.root, textvariable=self.var_del,   width=30)
        self.ent_name.place(x=200, y=10)
        self.ent_url.place(x=200, y=40)
        self.ent_del.place(x=200, y=70)

        # ── 操作按钮 ──────────────────────────
        tk.Button(self.root, text='添加',   width=10, command=self._add).place(x=60,  y=120)
        tk.Button(self.root, text='启动',   width=10, command=self._run).place(x=195, y=120)
        tk.Button(self.root, text='删除',   width=10, command=self._delete).place(x=330, y=120)
        tk.Button(self.root, text='停止',   width=10,
                  command=self._stop, background='red').place(x=460, y=120)

        # ── 显示列表 ──────────────────────────
        self.t_name    = tk.Text(self.root, width=18, height=15, font=14)
        self.t_address = tk.Text(self.root, width=38, height=15, font=14)
        self.t_name.place(x=20,  y=180)
        self.t_address.place(x=180, y=180)

    def _load_existing_sources(self):
        """启动时从文件加载已有的视频源记录。"""
        if os.path.exists(ID_NAME_FILE):
            with open(ID_NAME_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    self.t_name.insert('insert', line)
        if os.path.exists(URL_ADDRESS_FILE):
            with open(URL_ADDRESS_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    self.t_address.insert('insert', line)

    # ─────────────────────────────────────────
    # 按钮回调
    # ─────────────────────────────────────────

    def _add(self):
        """添加一条监控源记录（追加到文件和显示框）。"""
        name = self.var_name.get().strip()
        url  = self.var_url.get().strip()
        if not name or not url:
            messagebox.showwarning('输入错误', '请填写完整的 ID 名称和 URL 地址')
            return

        self.t_name.insert('end', name + '\n')
        self.t_address.insert('end', url + '\n')

        with open(ID_NAME_FILE,     'a+', encoding='utf-8') as f:
            f.write(name + '\n')
        with open(URL_ADDRESS_FILE, 'a+', encoding='utf-8') as f:
            f.write(url + '\n')

        self.var_name.set('')
        self.var_url.set('')

    def _run(self):
        """从配置文件读取所有视频源并启动多线程检测。"""
        sources = FallDetector.load_sources_from_files()
        if not sources:
            messagebox.showinfo('提示', '当前没有视频源，请先添加。')
            return
        self.detector.detect_multi(sources)

    def _delete(self):
        """根据 ID 名称从文件和显示框中删除对应记录。"""
        del_name = self.var_del.get().strip()
        if not del_name:
            messagebox.showwarning('输入错误', '请输入要删除的 ID 名称')
            return

        # 查找行号
        line_num = None
        with open(ID_NAME_FILE, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, start=1):
                if line.strip() == del_name:
                    line_num = i
                    break

        if line_num is None:
            messagebox.showerror('错误', f'未找到 ID: {del_name}')
            return

        # 删除两个文件中的对应行
        for filepath in (URL_ADDRESS_FILE, ID_NAME_FILE):
            with fileinput.input(filepath, inplace=True, mode='r') as f:
                for line in f:
                    if f.filelineno() != line_num:
                        print(line, end='')

        # 刷新显示框
        self.t_name.delete('1.0', 'end')
        self.t_address.delete('1.0', 'end')
        self._load_existing_sources()
        self.var_del.set('')

    @staticmethod
    def _stop():
        """退出程序。"""
        sys.exit(0)


def launch():
    """启动 GUI 监控管理界面。"""
    root = tk.Tk()
    app = MonitorApp(root)
    root.mainloop()


if __name__ == '__main__':
    launch()
