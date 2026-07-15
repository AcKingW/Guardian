"""
misc/sd.py — 独立工具脚本：樱花动画 + 烟花粒子效果
注意：此脚本与安防监控核心功能无关，是独立的 Tkinter 演示程序。

使用方式:
    python misc/sd.py
"""
import random
import time
import turtle
from math import cos, radians, sin
from random import choice, randint, uniform

import cv2
import tkinter as tk
from PIL import Image, ImageTk

# ─────────────────────────────────────────────
# 物理常数
# ─────────────────────────────────────────────
GRAVITY = 0.05
COLORS = [
    'red', 'blue', 'yellow', 'white', 'green',
    'orange', 'purple', 'seagreen', 'indigo', 'cornflowerblue',
]


# ─────────────────────────────────────────────
# 樱花绘制
# ─────────────────────────────────────────────

def draw_tree(branch: int, t: turtle.RawTurtle):
    """递归绘制樱花树枝。"""
    time.sleep(0.0005)
    if branch <= 3:
        return
    if 8 <= branch <= 12:
        t.color('snow' if random.randint(0, 2) == 0 else 'lightcoral')
        t.pensize(branch / 3)
    elif branch < 8:
        t.color('snow' if random.randint(0, 1) == 0 else 'lightcoral')
        t.pensize(branch / 2)
    else:
        t.color('sienna')
        t.pensize(branch / 10)

    t.forward(branch)
    a = 1.5 * random.random()
    b = 1.5 * random.random()
    t.right(20 * a)
    draw_tree(branch - 10 * b, t)
    t.left(40 * a)
    draw_tree(branch - 10 * b, t)
    t.right(20 * a)
    t.up()
    t.backward(branch)
    t.down()


def draw_petals(m: int, t: turtle.RawTurtle):
    """绘制随机散落花瓣。"""
    for _ in range(m):
        a = 200 - 400 * random.random()
        b = 10 - 20 * random.random()
        t.up()
        t.forward(b)
        t.left(90)
        t.forward(a)
        t.down()
        t.color('lightcoral')
        t.circle(1)
        t.up()
        t.backward(a)
        t.right(90)
        t.backward(b)


# ─────────────────────────────────────────────
# 烟花粒子系统
# ─────────────────────────────────────────────

class Particle:
    """单个烟花粒子，模拟爆炸扩散和自由落体。"""

    def __init__(self, canvas: tk.Canvas, idx: int, total: int,
                 explosion_speed: float, x: float = 0., y: float = 0.,
                 vx: float = 0., vy: float = 0., size: float = 2.,
                 color: str = 'red', lifespan: float = 2., **kwargs):
        self.id       = idx
        self.x, self.y = x, y
        self.initial_speed = explosion_speed
        self.vx, self.vy = vx, vy
        self.total    = total
        self.age      = 0.0
        self.color    = color
        self.canvas   = canvas
        self.lifespan = lifespan
        self.cid = canvas.create_oval(
            x - size, y - size, x + size, y + size, fill=color
        )

    def update(self, dt: float):
        self.age += dt
        if self.alive() and self.is_expanding():
            move_x = cos(radians(self.id * 360 / self.total)) * self.initial_speed
            move_y = sin(radians(self.id * 360 / self.total)) * self.initial_speed
            self.canvas.move(self.cid, move_x, move_y)
            self.vx = move_x / (float(dt) * 1000)
        elif self.alive():
            move_x = cos(radians(self.id * 360 / self.total))
            self.canvas.move(self.cid, self.vx + move_x, self.vy + GRAVITY * dt)
            self.vy += GRAVITY * dt
        elif self.cid is not None:
            self.canvas.delete(self.cid)
            self.cid = None

    def is_expanding(self) -> bool:
        return self.age <= 1.2

    def alive(self) -> bool:
        return self.age <= self.lifespan


def simulate(canvas: tk.Canvas):
    """主循环：每次调用创建并动画化一批烟花爆炸点。"""
    t1 = time.time()
    explode_points = []
    wait_time = randint(10, 100)

    for _ in range(randint(6, 10)):
        objects = []
        x = randint(50, 550)
        y = randint(50, 150)
        speed = uniform(0.5, 1.5)
        size  = uniform(0.5, 3)
        color = choice(COLORS)
        exp_speed = uniform(0.2, 1)
        total = randint(10, 50)
        for i in range(1, total):
            objects.append(Particle(
                canvas, idx=i, total=total, explosion_speed=exp_speed,
                x=x, y=y, vx=speed, vy=speed, color=color,
                size=size, lifespan=uniform(0.6, 1.75),
            ))
        explode_points.append(objects)

    total_time = 0.0
    while total_time < 1.8:
        time.sleep(0.01)
        t_new = time.time()
        dt = t_new - t1
        t1 = t_new
        for group in explode_points:
            for particle in group:
                particle.update(dt)
        canvas.update()
        total_time += dt

    canvas.after(wait_time, simulate, canvas)


def close(root: tk.Tk, *ignore):
    """关闭窗口并退出。"""
    root.quit()


def main():
    """启动樱花动画 + 烟花效果演示窗口。"""
    root = tk.Tk()
    root.title('樱花烟花演示')
    canvas = tk.Canvas(root, height=520, width=750)
    canvas.pack()

    w = turtle.TurtleScreen(canvas)
    t = turtle.RawTurtle(w)
    w.screensize(bg='black')
    t.hideturtle()
    t.getscreen().tracer(5, 0)
    t.left(90)
    t.up()
    t.backward(150)
    t.down()
    t.color('sienna')

    draw_tree(60, t)
    draw_petals(200, t)

    root.protocol('WM_DELETE_WINDOW', lambda: close(root))
    root.after(100, simulate, canvas)
    root.mainloop()


if __name__ == '__main__':
    main()
