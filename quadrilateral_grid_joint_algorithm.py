# -*- coding: utf-8 -*-
"""
基于四角格网的交互式 A* 可视化：
- 保留：障碍物密度热图（YlOrBr）、半透明障碍叠加、信息面板、路径密度对比曲线、Chaikin 平滑、
        传统 A* / Sigmoid 递增 A*、转弯惩罚滑块、拐弯次数显示等功能。
- 改动：六边形网格 -> 四角格网（4 邻接），距离改为曼哈顿距离，KDE 和局部密度也在方格上计算。
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import heapq
from math import exp, sqrt

S = "Local obstacle density (radius=1)"

# ===== 全局风格 =====
mpl.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 600,
    "axes.linewidth": 0.6,
    "axes.edgecolor": "black",
    "font.family": "Times New Roman",
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})
CMAP = mpl.cm.YlOrBr
COL_GRID   = "0.82"
COL_OBS    = (0.80, 0.20, 0.20, 0.55)

COL_START  = "green"
COL_GOAL   = "#F44336"
COL_PATH_TRAD   = "#90CAF9"
COL_EXPLORE_TRAD= "#FFC107"
COL_PATH_SIG    = "#9C27B0"
COL_EXPLORE_SIG = "#4CAF50"

COL_TRAD_MEAN = "#1E88E5"
COL_SIG_MEAN  = "#6A1B9A"
COL_DIFF_FILL_POS = "#43A047"
COL_DIFF_FILL_NEG = "#E53935"


# ===== 四角格网距离 / 邻接 / 范围 =====
def manhattan_distance(a, b):
    ax, ay = a
    bx, by = b
    return abs(ax - bx) + abs(ay - by)


def grid_neighbors(p, W, H):
    """4 邻接：上、下、左、右"""
    x, y = p
    nbrs = []
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < W and 0 <= ny < H:
            nbrs.append((nx, ny))
    return nbrs


def grid_range(center, radius, W, H):
    """以中心为 (cx,cy)，方形窗口 [-r,r] x [-r,r] 内所有格子"""
    cx, cy = center
    out = []
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            x, y = cx + dx, cy + dy
            if 0 <= x < W and 0 <= y < H:
                out.append((x, y))
    return out


# ===== KDE 障碍密度（在方格栅格上） =====
def kde_density_grid(W, H, obstacles, beta=1.6, R=3):
    D = np.zeros((H, W), dtype=float)
    if not obstacles:
        return D
    for o in obstacles:
        ox, oy = o
        for p in grid_range(o, R, W, H):
            px, py = p
            d = sqrt((px - ox) ** 2 + (py - oy) ** 2)
            D[py, px] += np.exp(- (d * d) / (2.0 * beta * beta))
    vmax = D.max()
    if vmax > 1e-12:
        D /= vmax
    return D


# ===== Sigmoid 递增权重 / 滤波 / Chaikin 平滑 =====
BETA_SIG = 1.0
H0       = 0.15
ALPHA_MIN= 0.5
ALPHA_MAX= 1.5


def sigmoid_increasing(Kn):
    return ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) / (1 + exp(-BETA_SIG * (Kn - H0)))


def moving_average(y, k=5):
    if y.size == 0:
        return y
    k = max(1, int(k))
    if k == 1:
        return y
    pad = k // 2
    y_pad = np.pad(y, (pad, pad), mode='edge')
    kernel = np.ones(k) / k
    return np.convolve(y_pad, kernel, mode='valid')


def chaikin_smooth(points, iters=2):
    """Chaikin 平滑（保端点，Q/R 交错插入）。仅用于可视化。"""
    pts = np.asarray(points, dtype=float)
    if len(pts) < 3 or iters <= 0:
        return pts
    for _ in range(iters):
        Q = 0.75 * pts[:-1] + 0.25 * pts[1:]
        R = 0.25 * pts[:-1] + 0.75 * pts[1:]
        out = np.empty((2 * len(pts) - 2, 2), dtype=float)
        out[0::2] = Q
        out[1::2] = R
        pts = np.vstack([pts[0:1], out, pts[-1:]])
    return pts


# ===== 主类：四角格网版本 =====
class GridAStarSigmoidIncVisualizer:
    def __init__(self):
        # 方格网大小
        self.W = 100
        self.H = 75
        self.cell_size = 1.0

        self.fig = plt.figure(figsize=(14, 10))
        self.ax = plt.axes([0.1, 0.3, 0.6, 0.65])
        self.ax.set_aspect('equal')

        # 起终点
        self.start = (0, 0)
        self.end = (self.W - 1, self.H - 1)

        # 网格：0 空闲 / 1 障碍
        self.grid = np.zeros((self.H, self.W), dtype=int)

        # 路径缓存
        self.path_trad, self.exp_trad = [], set()
        self.path_sig, self.exp_sig = [], set()

        # 绘图句柄
        self.path_line = None
        self.explore_scatter = None
        self.start_goal_patches = []
        self.cbar_ax = None

        # 每个格子的中心坐标
        self.cell_centers = {
            (x, y): (x + 0.5, y + 0.5)
            for y in range(self.H) for x in range(self.W)
        }

        # 转弯惩罚权重
        self.turn_weight = 0.0

        self.render_all()
        self.add_controls()
        self.add_info_panel()
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

    # ===== 转弯惩罚 =====
    def compute_turn(self, pp, prev, cur, to_goal_dist):
        """
        pp: 前前节点, prev: 前节点, cur: 当前候选
        to_goal_dist: 当前到终点的距离（用于归一化）
        """
        if pp is None or prev is None:
            return 0.0
        v1 = (prev[0] - pp[0],   prev[1] - pp[1])
        v2 = (cur[0]  - prev[0], cur[1]  - prev[1])
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        det = v1[0] * v2[1] - v1[1] * v2[0]
        angle = abs(np.arctan2(det, dot))  # 弧度
        base = max(to_goal_dist, 1)
        return float(self.turn_weight) * (angle / np.pi) / base

    # ===== 障碍集合 & 局部密度 =====
    def current_obstacles(self):
        obs = set()
        for y in range(self.H):
            for x in range(self.W):
                if self.grid[y, x] == 1:
                    obs.add((x, y))
        obs.discard(self.start)
        obs.discard(self.end)
        return obs

    def local_density_ratio(self, obstacles, node, radius=1):
        """方格内局部障碍密度：窗口大小 (2r+1)×(2r+1)"""
        W, H = self.W, self.H
        cx, cy = node
        tot, cnt = 0, 0
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                x, y = cx + dx, cy + dy
                if 0 <= x < W and 0 <= y < H:
                    tot += 1
                    if (x, y) in obstacles:
                        cnt += 1
        return (cnt / tot) if tot > 0 else 0.0

    # ===== A* + 转弯惩罚 + 可选密度启发 =====
    def astar_with_turn(self, improved=False):
        W, H = self.W, self.H
        start, goal = self.start, self.end
        obstacles = self.current_obstacles()

        obstacles.discard(start)
        obstacles.discard(goal)

        open_heap = []
        heapq.heappush(open_heap, (0.0, start, None, None))  # (priority, cur, prev, pprev)
        came_from = {}
        cost_so_far = {start: 0.0}
        explored = set([start])
        turn_counts = {start: 0}

        while open_heap:
            _, current, prev, pprev = heapq.heappop(open_heap)
            if current == goal:
                break

            to_goal_dist = manhattan_distance(current, goal)
            for nb in grid_neighbors(current, W, H):
                if nb in obstacles:
                    continue

                # 转弯惩罚
                turn_penalty = self.compute_turn(pprev, prev, nb, to_goal_dist)

                new_cost = cost_so_far[current] + 1.0 + turn_penalty
                if nb not in cost_so_far or new_cost < cost_so_far[nb]:
                    cost_so_far[nb] = new_cost

                    # 启发：传统 vs Sigmoid 递增
                    h = manhattan_distance(nb, goal)
                    if improved:
                        Kn = self.local_density_ratio(obstacles, nb, radius=1)
                        alpha = sigmoid_increasing(Kn)
                        h = alpha * h

                    priority = new_cost + h
                    heapq.heappush(open_heap, (priority, nb, current, prev))
                    came_from[nb] = (current, prev)
                    explored.add(nb)

                    # 统计拐弯次数
                    if prev is not None and turn_penalty > 1e-6:
                        turn_counts[nb] = turn_counts.get(current, 0) + 1
                    else:
                        turn_counts[nb] = turn_counts.get(current, 0)

        # 回溯路径
        current = goal
        path = []
        while current != start:
            path.append(current)
            prev_tuple = came_from.get(current, None)
            if prev_tuple is None:
                return [], explored, 0  # 无解
            current = prev_tuple[0]
        path.append(start)
        path.reverse()
        return path, explored, turn_counts.get(goal, 0)

    # ===== UI 控件 =====
    def add_controls(self):
        ax_clear = plt.axes([0.1, 0.2, 0.12, 0.05])
        self.btn_clear = Button(ax_clear, '清除所有障碍')
        self.btn_clear.on_clicked(self.clear_obstacles)

        ax_random = plt.axes([0.24, 0.2, 0.12, 0.05])
        self.btn_random = Button(ax_random, '随机生成障碍')
        self.btn_random.on_clicked(self.random_obstacles)

        ax_astar1 = plt.axes([0.38, 0.2, 0.12, 0.05])
        self.btn_astar1 = Button(ax_astar1, '传统A*算法')
        self.btn_astar1.on_clicked(lambda event: self.run_and_show(improved=False))

        ax_astar2 = plt.axes([0.52, 0.2, 0.12, 0.05])
        self.btn_astar2 = Button(ax_astar2, 'Sigmoid递增A*')
        self.btn_astar2.on_clicked(lambda event: self.run_and_show(improved=True))

        ax_clear_path = plt.axes([0.66, 0.2, 0.12, 0.05])
        self.btn_clear_path = Button(ax_clear_path, '清除路径')
        self.btn_clear_path.on_clicked(self.clear_path)

        ax_compare = plt.axes([0.80, 0.2, 0.14, 0.05])
        self.btn_compare = Button(ax_compare, '路径密度对比曲线')
        self.btn_compare.on_clicked(lambda event: self.plot_density_comparison())

        # 转弯惩罚滑块
        ax_tw = plt.axes([0.70, 0.15, 0.18, 0.03])
        self.slider_tw = Slider(ax_tw, '转弯惩罚', 0.0, 2.0, valinit=self.turn_weight)
        self.slider_tw.on_changed(self.update_turn_weight)

    def add_info_panel(self):
        self.info_ax = plt.axes([0.72, 0.4, 0.25, 0.55])
        self.info_ax.axis('off')
        self.info_ax.text(0.1, 0.9, "路径规划统计信息", fontsize=14, fontweight='bold')
        self.length_text   = self.info_ax.text(0.15, 0.75, "路径长度: 0", fontsize=10)
        self.explored_text = self.info_ax.text(0.15, 0.70, "遍历格子数: 0", fontsize=10)
        self.density_text  = self.info_ax.text(0.15, 0.65, "路径障碍密度: 0.00", fontsize=10)
        self.turns_text    = self.info_ax.text(0.15, 0.60, "拐弯次数: 0", fontsize=10)
        self.add_legend()

    def add_legend(self):
        legend_elements = [
            Rectangle((0, 0), 1, 1, fc='white', ec='black', label='空闲'),
            Rectangle((0, 0), 1, 1, fc=COL_OBS, ec='black', label='障碍(叠加)'),
            Rectangle((0, 0), 1, 1, fc=COL_START, ec='black', label='起点'),
            Rectangle((0, 0), 1, 1, fc=COL_GOAL,  ec='black', label='终点'),
            Rectangle((0, 0), 1, 1, fc=COL_PATH_TRAD, label='传统路径'),
            Rectangle((0, 0), 1, 1, fc=COL_EXPLORE_TRAD, label='传统A*探索节点'),
            Rectangle((0, 0), 1, 1, fc=COL_PATH_SIG, label='Sigmoid路径'),
            Rectangle((0, 0), 1, 1, fc=COL_EXPLORE_SIG, label='Sigmoid探索节点'),
        ]
        self.info_ax.legend(handles=legend_elements, loc='lower left',
                            bbox_to_anchor=(0.1, 0.05), fontsize=10)

    # ===== 鼠标点选加/减障碍 =====
    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        mx, my = event.xdata, event.ydata
        if mx is None or my is None:
            return
        if 0 <= mx < self.W and 0 <= my < self.H:
            x = int(mx)
            y = int(my)
            if (x, y) in [self.start, self.end]:
                return
            self.grid[y, x] = 0 if self.grid[y, x] == 1 else 1
            self.clear_path()
            self.render_all()

    def clear_obstacles(self, event=None):
        self.grid.fill(0)
        self.clear_path()
        self.render_all()

    def random_obstacles(self, event=None):
        self.grid.fill(0)
        for y in range(self.H):
            for x in range(self.W):
                if (x, y) not in [self.start, self.end] and np.random.random() < 0.3:
                    self.grid[y, x] = 1
        self.clear_path()
        self.render_all()

    # ===== 统计数值 =====
    def calculate_obstacle_density(self, node, radius=1):
        obstacles = self.current_obstacles()
        return self.local_density_ratio(obstacles, node, radius)

    def calculate_path_density(self, path, radius=1):
        if not path:
            return 0.0
        return float(np.mean([self.calculate_obstacle_density(p, radius) for p in path]))

    def update_stats_panel(self, path, explored, turns=0):
        length = max(0, len(path) - 1)
        explored_n = len(explored)
        density_val = self.calculate_path_density(path, radius=1) if path else 0.0
        self.length_text.set_text(f"路径长度: {length}")
        self.explored_text.set_text(f"遍历格子数: {explored_n}")
        self.density_text.set_text(f"路径障碍密度: {density_val:.2f}")
        self.turns_text.set_text(f"拐弯次数: {int(turns)}")
        self.fig.canvas.draw_idle()

    def update_stats(self):
        self.length_text.set_text(f"路径长度: 0")
        self.explored_text.set_text(f"遍历格子数: 0")
        self.density_text.set_text(f"路径障碍密度: 0.00")
        self.turns_text.set_text(f"拐弯次数: 0")
        self.fig.canvas.draw_idle()

    # ===== 绘制：密度底图 + 障碍叠加 + colorbar =====
    def render_all(self):
        self.ax.clear()
        self.ax.set_xlim(0, self.W)
        self.ax.set_ylim(0, self.H)
        self.ax.set_aspect('equal')
        self.ax.invert_yaxis()  # 让 y=0 在上方，看起来更像“地图”
        self.ax.axis('off')

        W, H = self.W, self.H
        cell = self.cell_size

        obstacles = self.current_obstacles()
        density = kde_density_grid(W, H, obstacles, beta=1.6, R=3)

        # 背景方格 + 密度填色
        patches = []
        faces = []
        for y in range(H):
            for x in range(W):
                rect = Rectangle((x, y), cell, cell,
                                 edgecolor=COL_GRID, linewidth=0.6)
                patches.append(rect)
                v = density[y, x]
                if v > 0:
                    c = CMAP(v)
                    faces.append((c[0], c[1], c[2], 0.22 + 0.25 * v))
                else:
                    faces.append((1, 1, 1, 1))

        coll = PatchCollection(patches, match_original=True)
        coll.set_facecolor(faces)
        self.ax.add_collection(coll)

        # 障碍覆盖（小一圈的红色矩形）
        for (x, y) in obstacles:
            self.ax.add_patch(Rectangle((x + 0.05, y + 0.05),
                                        cell * 0.9, cell * 0.9,
                                        facecolor=COL_OBS, edgecolor=None, linewidth=0))

        # 起终点
        for p, col in [(self.start, COL_START), (self.end, COL_GOAL)]:
            x, y = p
            patch = Rectangle((x + 0.1, y + 0.1),
                              cell * 0.8, cell * 0.8,
                              facecolor=col, edgecolor="black", linewidth=0.6)
            self.ax.add_patch(patch)

        # 右侧 colorbar（先删旧的）
        if self.cbar_ax is not None:
            try:
                self.cbar_ax.remove()
            except Exception:
                pass
            self.cbar_ax = None
        self.cbar_ax = self.fig.add_axes([0.705, 0.3, 0.012, 0.60])
        norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
        cb = mpl.colorbar.ColorbarBase(self.cbar_ax, cmap=CMAP, norm=norm)
        cb.set_label("Obstacle density (normalized)", rotation=90, labelpad=6)

        self.ax.set_title("四角格网：障碍密度底图（YlOrBr）", pad=6)
        self.fig.canvas.draw_idle()

    # ===== 跑并显示某一种算法 =====
    def run_and_show(self, improved=False):
        # 清掉旧路径叠加
        if self.path_line is not None:
            try:
                self.path_line.remove()
            except Exception:
                pass
            self.path_line = None
        if self.explore_scatter is not None:
            try:
                self.explore_scatter.remove()
            except Exception:
                pass
            self.explore_scatter = None
        for p in self.start_goal_patches:
            try:
                p.remove()
            except Exception:
                pass
        self.start_goal_patches = []

        # A* 计算
        path, explored, turns = self.astar_with_turn(improved=improved)

        # 缓存路径
        if improved:
            self.path_sig, self.exp_sig = path, explored
        else:
            self.path_trad, self.exp_trad = path, explored

        if path:
            # 探索节点散点
            ex_x, ex_y = [], []
            for (x, y) in explored:
                cx, cy = self.cell_centers[(x, y)]
                ex_x.append(cx)
                ex_y.append(cy)
            ex_color = COL_EXPLORE_SIG if improved else COL_EXPLORE_TRAD
            self.explore_scatter = self.ax.scatter(
                ex_x, ex_y, s=18, c=ex_color, alpha=0.75,
                linewidths=0, zorder=9
            )

            # 路径线 + Chaikin 平滑
            pts = np.array([self.cell_centers[(x, y)] for (x, y) in path], dtype=float)
            smooth = chaikin_smooth(pts, iters=2)
            pcolor = COL_PATH_SIG if improved else COL_PATH_TRAD
            self.path_line, = self.ax.plot(
                smooth[:, 0], smooth[:, 1],
                color=pcolor, linewidth=2.0, zorder=10,
                solid_joinstyle='round'
            )

            # 重新压一遍起终点
            for p, col in [(self.start, COL_START), (self.end, COL_GOAL)]:
                x, y = p
                patch = Rectangle(
                    (x + 0.1, y + 0.1),
                    self.cell_size * 0.8, self.cell_size * 0.8,
                    facecolor=col, edgecolor="black", linewidth=0.6, zorder=11
                )
                self.ax.add_patch(patch)
                self.start_goal_patches.append(patch)

            self.update_stats_panel(path, explored, turns=turns)
            self.fig.canvas.draw_idle()
            print(f"{'Sigmoid递增A*' if improved else '传统A*'} 完成! "
                  f"平均局部密度: {self.calculate_path_density(path):.2f}，拐弯次数: {turns}")
        else:
            print("找不到路径!")
            self.update_stats()

    # ===== 清空路径叠加 =====
    def clear_path(self, *args):
        self.path_trad, self.exp_trad = [], set()
        self.path_sig, self.exp_sig = [], set()

        if self.path_line is not None:
            try:
                self.path_line.remove()
            except Exception:
                pass
            self.path_line = None
        if self.explore_scatter is not None:
            try:
                self.explore_scatter.remove()
            except Exception:
                pass
            self.explore_scatter = None
        for p in self.start_goal_patches:
            try:
                p.remove()
            except Exception:
                pass
        self.start_goal_patches = []
        self.update_stats()
        self.fig.canvas.draw_idle()

    # ===== 路径局部密度对比曲线 =====
    def plot_density_comparison(self, radius=1, smooth_k=7):
        obstacles = self.current_obstacles()

        # 若路径还未计算，则先跑一遍
        if not self.path_trad:
            self.path_trad, self.exp_trad, _ = self.astar_with_turn(improved=False)
        if not self.path_sig:
            self.path_sig, self.exp_sig, _ = self.astar_with_turn(improved=True)

        if not self.path_trad or not self.path_sig:
            print("至少有一条路径不存在，无法绘制对比曲线。请先确保两种算法都能找到路径。")
            return

        def path_density_series(path):
            return np.array(
                [self.local_density_ratio(obstacles, p, radius) for p in path],
                dtype=float
            )

        y_trad = path_density_series(self.path_trad)
        y_sig = path_density_series(self.path_sig)

        # 平滑
        y_trad_s = moving_average(y_trad, k=smooth_k)
        y_sig_s = moving_average(y_sig, k=smooth_k)

        x_trad = np.arange(len(y_trad))
        x_sig = np.arange(len(y_sig))

        m_trad = float(np.mean(y_trad))
        m_sig = float(np.mean(y_sig))

        fig = plt.figure(figsize=(11, 7.5))
        gs = fig.add_gridspec(2, 1, height_ratios=[2.2, 1], hspace=0.25)

        # 顶部：两条路径的局部密度
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(x_trad, y_trad, color=COL_PATH_TRAD, alpha=0.35, linewidth=1.2,
                 label="Traditional A* (Local Density)")
        ax1.plot(x_sig, y_sig, color=COL_PATH_SIG, alpha=0.35, linewidth=1.2,
                 label="Sigmoid Incremental A* (Local Density)")

        ax1.plot(x_trad, y_trad_s, color=COL_PATH_TRAD, linewidth=2.2,
                 label="Traditional A* (Smoothing)")
        ax1.plot(x_sig, y_sig_s, color=COL_PATH_SIG, linewidth=2.2,
                 label="Sigmoid Incremental A* (Smoothing)")

        ax1.fill_between(x_trad, y_trad_s, m_trad, color=COL_PATH_TRAD,
                         alpha=0.10, linewidth=0)
        ax1.fill_between(x_sig, y_sig_s, m_sig, color=COL_PATH_SIG,
                         alpha=0.10, linewidth=0)

        ax1.axhline(m_trad, color=COL_TRAD_MEAN, linestyle="--", linewidth=1.2,
                    label=f"Traditional A* average: {m_trad:.2f}")
        ax1.axhline(m_sig, color=COL_SIG_MEAN, linestyle="--", linewidth=1.2,
                    label=f"Sigmoid average: {m_sig:.2f}")

        # 极值标注
        t_max_i, t_min_i = int(np.argmax(y_trad_s)), int(np.argmin(y_trad_s))
        s_max_i, s_min_i = int(np.argmax(y_sig_s)), int(np.argmin(y_sig_s))
        ax1.scatter([t_max_i, t_min_i],
                    [y_trad_s[t_max_i], y_trad_s[t_min_i]],
                    color=COL_PATH_TRAD, s=28, zorder=5)
        ax1.scatter([s_max_i, s_min_i],
                    [y_sig_s[s_max_i], y_sig_s[s_min_i]],
                    color=COL_PATH_SIG, s=28, zorder=5)
        ax1.text(t_max_i, y_trad_s[t_max_i] + 0.02,
                 f"max {y_trad_s[t_max_i]:.2f}",
                 color=COL_PATH_TRAD, fontsize=9, ha='center')
        ax1.text(t_min_i, y_trad_s[t_min_i] - 0.04,
                 f"min {y_trad_s[t_min_i]:.2f}",
                 color=COL_PATH_TRAD, fontsize=9, ha='center')
        ax1.text(s_max_i, y_sig_s[s_max_i] + 0.02,
                 f"max {y_sig_s[s_max_i]:.2f}",
                 color=COL_PATH_SIG, fontsize=9, ha='center')
        ax1.text(s_min_i, y_sig_s[s_min_i] - 0.04,
                 f"min {y_sig_s[s_min_i]:.2f}",
                 color=COL_PATH_SIG, fontsize=9, ha='center')

        ax1.set_title(
            "Comparison of path local obstacle density (traditional A* vs Sigmoid incremental A*)",
            pad=6
        )
        ax1.set_xlabel("Path step sequence (index)")
        ax1.set_ylabel(f"{S}")
        ax1.set_ylim(0, 1.02)
        ax1.grid(alpha=0.25, linewidth=0.6)
        ax1.legend(ncol=2, frameon=False)

        # 底部：差值曲线（Sigmoid - Traditional）
        ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
        L = max(len(y_trad_s), len(y_sig_s))
        yT = np.full(L, np.nan)
        yS = np.full(L, np.nan)
        yT[:len(y_trad_s)] = y_trad_s
        yS[:len(y_sig_s)] = y_sig_s
        diff = yS - yT

        ax2.axhline(0.0, color="0.25", linewidth=1.0)
        x = np.arange(L)
        ax2.fill_between(x, diff, 0, where=(diff >= 0),
                         color=COL_DIFF_FILL_POS, alpha=0.25, linewidth=0)
        ax2.fill_between(x, diff, 0, where=(diff < 0),
                         color=COL_DIFF_FILL_NEG, alpha=0.25, linewidth=0)
        ax2.plot(x, diff, color="0.2", linewidth=1.2)

        pos_adv = np.nanmean(diff[diff >= 0]) if np.any(diff >= 0) else 0.0
        neg_adv = np.nanmean(diff[diff < 0]) if np.any(diff < 0) else 0.0
        ax2.text(0.01, 1.02,
                 f"Average positive advantage: {pos_adv:.2f}",
                 transform=ax2.transAxes, fontsize=10,
                 color=COL_DIFF_FILL_POS)
        ax2.text(0.31, 1.02,
                 f"Average negative disadvantage: {neg_adv:.2f}",
                 transform=ax2.transAxes, fontsize=10,
                 color=COL_DIFF_FILL_NEG)

        ax2.set_ylabel("Density difference (Sig - Trad)")
        ax2.set_xlabel("Path step sequence (index)")
        ax2.grid(alpha=0.25, linewidth=0.6)

        fig.tight_layout()
        out_path = "../path_local_density_comparison_grid.png"
        plt.savefig(out_path, bbox_inches="tight")
        plt.show()
        print("Exported:", out_path)

    # ===== 其他 =====
    def update_turn_weight(self, value):
        self.turn_weight = float(value)


if __name__ == "__main__":
    mpl.rcParams['font.family'] = 'SimSun'  # 你本地如果没有宋体，可以改掉这一行
    plt.rcParams['axes.unicode_minus'] = False
    vis = GridAStarSigmoidIncVisualizer()
    plt.show()
