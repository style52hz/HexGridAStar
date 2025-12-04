# -*- coding: utf-8 -*-
"""
在原始交互版上融合第二段代码的配色/样式/布局 + 路径局部密度对比曲线 + 路径平滑(Chaikin)
- 背景：障碍物密度热图（YlOrBr），六角边框线宽0.6、edgecolor=0.82灰
- 障碍：半透明红色小六边形覆盖
- 交互：点击加/减障碍；传统A*；Sigmoid递增A*；清路径；随机障碍；路径密度对比曲线
- 新增：对两种算法最终生成的可视化路径使用 Chaikin 平滑（仅影响绘图，不影响统计与密度计算）
- 网格大小：W=25, H=15（列×行）
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.patches import RegularPolygon, Rectangle
from matplotlib.collections import PatchCollection
import heapq
from math import exp, sqrt

S = "Local obstacle density (radius=1)"

# ===== 全局风格：融合第二段样式 =====
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
CMAP = mpl.cm.YlOrBr         # 第二段的色带
COL_GRID   = "0.82"          # 六边形边框灰
COL_OBS    = (0.80, 0.20, 0.20, 0.55)  # 障碍覆盖(半透明红)

# 叠加层(沿用第一段语义色)
COL_START  = "green"
COL_GOAL   = "#F44336"
COL_PATH_TRAD   = "#90CAF9"
COL_EXPLORE_TRAD= "#FFC107"
COL_PATH_SIG    = "#9C27B0"
COL_EXPLORE_SIG = "#4CAF50"

# 曲线配色（论文级：主线用路径色，均值线为同色深阶，填充为低透明）
COL_TRAD_MEAN = "#1E88E5"    # 传统路径均值线（蓝系）
COL_SIG_MEAN  = "#6A1B9A"    # Sigmoid路径均值线（紫系）
COL_DIFF_FILL_POS = "#43A047" # 差值>0（Sig>Trad）填充
COL_DIFF_FILL_NEG = "#E53935" # 差值<0（Sig<Trad）填充

# ===== 正确的 odd-r 偏移 <-> cube 转换 =====
def offset_to_cube_odd_r(col, row):
    x = col - (row - (row & 1)) // 2
    z = row
    y = -x - z
    return x, y, z

def cube_to_offset_odd_r(x, y, z):
    col = x + (z - (z & 1)) // 2
    row = z
    return int(col), int(row)

CUBE_DIRS = [
    (+1, -1, 0), (+1, 0, -1), (0, +1, -1),
    (-1, +1, 0), (-1, 0, +1), (0, -1, +1)
]

def hex_distance(a, b):
    ax, ay = a
    bx, by = b
    ax_c, ay_c, az_c = offset_to_cube_odd_r(ax, ay)
    bx_c, by_c, bz_c = offset_to_cube_odd_r(bx, by)
    return max(abs(ax_c - bx_c), abs(ay_c - by_c), abs(az_c - bz_c))

def hex_neighbors(p, W, H):
    col, row = p
    x, y, z = offset_to_cube_odd_r(col, row)
    out = []
    for dx, dy, dz in CUBE_DIRS:
        nx, ny, nz = x + dx, y + dy, z + dz
        oc, orow = cube_to_offset_odd_r(nx, ny, nz)
        if 0 <= oc < W and 0 <= orow < H:
            out.append((oc, orow))
    return out

def hex_center(col, row, r, dx, dy):
    x = col * dx + (dx / 2 if (row & 1) else 0.0)
    y = row * dy
    return x, y

def hex_range(center, radius, W, H):
    cx, cy = center
    ccx, ccy, ccz = offset_to_cube_odd_r(cx, cy)
    out = []
    for dx in range(-radius, radius+1):
        for dy in range(-radius, radius+1):
            dz = -dx - dy
            if max(abs(dx), abs(dy), abs(dz)) <= radius:
                x, y, z = ccx + dx, ccy + dy, ccz + dz
                oc, orow = cube_to_offset_odd_r(x, y, z)
                if 0 <= oc < W and 0 <= orow < H:
                    out.append((oc, orow))
    return out

# ===== KDE 障碍密度(用于底图渲染) =====
def kde_density_hex(W, H, obstacles, beta=1.6, R=3):
    D = np.zeros((H, W), dtype=float)
    if not obstacles:
        return D
    for o in obstacles:
        for p in hex_range(o, R, W, H):
            d = hex_distance(p, o)
            D[p[1], p[0]] += np.exp(- (d*d) / (2.0 * beta * beta))
    vmax = D.max()
    if vmax > 1e-12:
        D /= vmax
    return D

# ===== 两种启发：传统 & Sigmoid递增 =====
BETA_SIG = 1.0    # 与第一段一致
H0       = 0.15
ALPHA_MIN= 0.5
ALPHA_MAX= 1.5

def sigmoid_increasing(Kn):
    return ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) / (1 + exp(-BETA_SIG * (Kn - H0)))

def local_density_ratio(obstacles, node, W, H, radius=1):
    tot = 0
    cnt = 0
    for q in hex_range(node, radius, W, H):
        tot += 1
        if q in obstacles:
            cnt += 1
    return (cnt / tot) if tot > 0 else 0.0

def heuristic(a, b, improved, obstacles, W, H):
    Hbase = hex_distance(a, b)
    if not improved:
        return Hbase
    Kn = local_density_ratio(obstacles, a, W, H, radius=1)
    alpha = sigmoid_increasing(Kn)
    return alpha * Hbase

def astar_hex(W, H, start, goal, obstacles, improved=False):
    # 确保起终点可通行
    obstacles = set(obstacles)
    obstacles.discard(start)
    obstacles.discard(goal)

    open_heap = []
    heapq.heappush(open_heap, (0, start))
    g = {start: 0}
    came_from = {}
    open_set = {start}
    closed = set()

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current not in open_set:
            continue
        open_set.remove(current)
        closed.add(current)

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path, closed

        for nb in hex_neighbors(current, W, H):
            if nb in obstacles or nb in closed:
                continue
            ng = g[current] + 1
            if nb not in g or ng < g[nb]:
                g[nb] = ng
                f = ng + heuristic(nb, goal, improved, obstacles, W, H)
                came_from[nb] = current
                heapq.heappush(open_heap, (f, nb))
                open_set.add(nb)

    return [], closed  # 无解

# ===== 平滑工具：Chaikin（仅用于绘图，不更改统计路径） =====
def chaikin_smooth(points, iters=3):
    """
    Chaikin smoothing with:
    - interleaved Q/R order (Q0,R0,Q1,R1,...)
    - endpoint preservation (keep first and last points)
    - iters in {1,2} is recommended; too many leads to huge point counts
    """
    pts = np.asarray(points, dtype=float)
    if len(pts) < 3 or iters <= 0:
        return pts
    for _ in range(iters):
        n = len(pts)
        Q = 0.75 * pts[:-1] + 0.25 * pts[1:]
        R = 0.25 * pts[:-1] + 0.75 * pts[1:]
        out = np.empty((2*n - 2, 2), dtype=float)
        out[0::2] = Q
        out[1::2] = R
        pts = np.vstack([pts[0:1], out, pts[-1:]])  # preserve endpoints
    return pts

# ===== 平滑工具（移动平均，不依赖第三方库） =====
def moving_average(y, k=5):
    if y.size == 0: return y
    k = max(1, int(k))
    if k == 1: return y
    pad = k // 2
    y_pad = np.pad(y, (pad, pad), mode='edge')
    kernel = np.ones(k) / k
    return np.convolve(y_pad, kernel, mode='valid')

# ===== 主类：保留布局/按钮/信息面板 + 密度对比曲线 + 平滑绘制 =====
class HexAStarSigmoidIncVisualizer:
    def __init__(self):
        # 不等长六角格网
        self.W = 25  # 列数
        self.H = 15  # 行数

        self.fig = plt.figure(figsize=(14, 10))
        self.ax = plt.axes([0.1, 0.3, 0.6, 0.65])  # 主绘图区(保留第一段布局)
        self.ax.set_aspect('equal')

        self.hex_radius = 0.5
        self.hex_dx = sqrt(3) * self.hex_radius
        self.hex_dy = 1.5 * self.hex_radius

        # 起终点
        self.start = (0, 0)
        self.end   = (self.W - 1, self.H - 1)

        # 网格：0空闲/1障碍
        self.grid = np.zeros((self.H, self.W), dtype=int)

        # 运行结果（各一次）
        self.path_trad, self.exp_trad = [], set()
        self.path_sig,  self.exp_sig  = [], set()

        # 最新一次 run 的展示句柄
        self.path_line = None
        self.explore_scatter = None
        self.start_goal_patches = []

        # 预计算中心坐标（odd-r）
        self.hex_centers = {
            (x, y): ((x + 0.5) * self.hex_dx if (y & 1) else x * self.hex_dx,
                     y * self.hex_dy)
            for y in range(self.H) for x in range(self.W)
        }

        self.render_all()    # 底图(密度) + 障碍叠加
        self.add_controls()
        self.add_info_panel()
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

    # ===== UI =====
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

    def add_info_panel(self):
        self.info_ax = plt.axes([0.72, 0.4, 0.25, 0.55])  # 保持布局
        self.info_ax.axis('off')
        self.info_ax.text(0.1, 0.9, "路径规划统计信息", fontsize=14, fontweight='bold')
        self.length_text   = self.info_ax.text(0.15, 0.75, "路径长度: 0", fontsize=10)
        self.explored_text = self.info_ax.text(0.15, 0.70, "遍历格子数: 0", fontsize=10)
        self.density_text  = self.info_ax.text(0.15, 0.65, "路径障碍密度: 0.00", fontsize=10)
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

    # ===== 交互 =====
    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        mx, my = event.xdata, event.ydata
        if mx is None or my is None:
            return
        for (x, y), (cx, cy) in self.hex_centers.items():
            if (mx - cx)**2 + (my - cy)**2 <= (self.hex_radius * 0.9)**2:
                if (x, y) in [self.start, self.end]:
                    return
                self.grid[y, x] = 0 if self.grid[y, x] == 1 else 1
                self.clear_path()      # 修改障碍后清路径
                self.render_all()      # 重绘底图与障碍
                return

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

    # ===== 统计 =====
    def calculate_obstacle_density(self, node, radius=1):
        obstacles = self.current_obstacles()
        return local_density_ratio(obstacles, node, self.W, self.H, radius)

    def calculate_path_density(self, path, radius=1):
        if not path: return 0.0
        return float(np.mean([self.calculate_obstacle_density(p, radius) for p in path]))

    def update_stats_panel(self, path, explored):
        # 用于 run_and_show 显示最新一次运行结果
        length = max(0, len(path) - 1)
        explored_n = len(explored)
        density_val = self.calculate_path_density(path, radius=1) if path else 0.0
        self.length_text.set_text(f"路径长度: {length}")
        self.explored_text.set_text(f"遍历格子数: {explored_n}")
        self.density_text.set_text(f"路径障碍密度: {density_val:.2f}")
        self.fig.canvas.draw_idle()

    def update_stats(self):
        # 清空显示
        self.length_text.set_text(f"路径长度: 0")
        self.explored_text.set_text(f"遍历格子数: 0")
        self.density_text.set_text(f"路径障碍密度: 0.00")
        self.fig.canvas.draw_idle()

    # ===== 绘制：底图(密度) + 障碍叠加 + colorbar =====
    def render_all(self):
        self.ax.clear()
        self.ax.set_xlim(-1, self.W * self.hex_dx + 1)
        self.ax.set_ylim(-1, self.H * self.hex_dy + 1)
        self.ax.axis('off')

        W, H = self.W, self.H
        r, dx, dy = self.hex_radius, self.hex_dx, self.hex_dy

        obstacles = self.current_obstacles()
        density = kde_density_hex(W, H, obstacles, beta=1.6, R=3)

        # 背景密度六边形
        patches, faces = [], []
        for y in range(H):
            for x in range(W):
                cx, cy = self.hex_centers[(x, y)]
                poly = RegularPolygon((cx, cy), 6, r, orientation=0.0,
                                      edgecolor=COL_GRID, linewidth=0.6, antialiased=True)
                patches.append(poly)
                v = density[y, x]
                if v > 0:
                    c = CMAP(v)
                    faces.append((c[0], c[1], c[2], 0.22 + 0.25*v))
                else:
                    faces.append((1, 1, 1, 1))

        coll = PatchCollection(patches, match_original=True)
        coll.set_facecolor(faces)
        self.ax.add_collection(coll)

        # 障碍覆盖(小一圈的红色半透明六边形)
        for (x, y) in obstacles:
            cx, cy = self.hex_centers[(x, y)]
            self.ax.add_patch(RegularPolygon((cx, cy), 6, r*0.96, orientation=0.0,
                                             facecolor=COL_OBS, edgecolor=None, linewidth=0))

        # 右侧紧凑colorbar（不移动信息面板）
        cax = self.fig.add_axes([0.705, 0.3, 0.012, 0.60])
        norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
        cb = mpl.colorbar.ColorbarBase(cax, cmap=CMAP, norm=norm)
        cb.set_label("Obstacle density (normalized)", rotation=90, labelpad=6)

        self.ax.set_title("六角格网：障碍密度底图（YlOrBr）", pad=6)
        self.fig.canvas.draw_idle()

    def current_obstacles(self):
        obs = set()
        for y in range(self.H):
            for x in range(self.W):
                if self.grid[y, x] == 1:
                    obs.add((x, y))
        obs.discard(self.start)
        obs.discard(self.end)
        return obs

    # ===== 跑并显示（单种算法），对可视化路径做Chaikin平滑 =====
    def run_and_show(self, improved=False):
        # 清理旧叠加但保留底图
        if self.path_line is not None:
            try: self.path_line.remove()
            except Exception: pass
            self.path_line = None
        if self.explore_scatter is not None:
            try: self.explore_scatter.remove()
            except Exception: pass
            self.explore_scatter = None
        for p in self.start_goal_patches:
            try: p.remove()
            except Exception: pass
        self.start_goal_patches = []

        W, H = self.W, self.H
        obstacles = self.current_obstacles()
        path, explored = astar_hex(W, H, self.start, self.end, obstacles, improved=improved)

        # 保存两条路径以便对比
        if improved:
            self.path_sig, self.exp_sig = path, explored
        else:
            self.path_trad, self.exp_trad = path, explored

        # 叠加展示
        if path:
            # 探索节点
            ex_x, ex_y = [], []
            for (x, y) in explored:
                cx, cy = self.hex_centers[(x, y)]
                ex_x.append(cx); ex_y.append(cy)
            ex_color = COL_EXPLORE_SIG if improved else COL_EXPLORE_TRAD
            self.explore_scatter = self.ax.scatter(ex_x, ex_y, s=18, c=ex_color, alpha=0.75,
                                                   linewidths=0, zorder=9)

            # 路径中心→Chaikin平滑→绘制
            pts = np.array([self.hex_centers[(x, y)] for (x, y) in path], dtype=float)
            # 推荐 iters=2；若路径太短(<3)将自动跳过平滑
            smooth = chaikin_smooth(pts, iters=2)
            pcolor = COL_PATH_SIG if improved else COL_PATH_TRAD
            self.path_line, = self.ax.plot(smooth[:,0], smooth[:,1],
                                           color=pcolor, linewidth=2.0, zorder=10)

            # 起终点
            for p, col in [(self.start, COL_START), (self.end, COL_GOAL)]:
                cx, cy = self.hex_centers[p]
                patch = self.ax.add_patch(RegularPolygon((cx, cy), 6, self.hex_radius*0.80,
                                                         orientation=0.0, facecolor=col,
                                                         edgecolor="black", linewidth=0.6, zorder=10))
                self.start_goal_patches.append(patch)

            self.update_stats_panel(path, explored)
            self.fig.canvas.draw_idle()
            print(f"{'Sigmoid递增A*' if improved else '传统A*'} 完成! 平均局部密度: {self.calculate_path_density(path):.2f}")
        else:
            print("找不到路径!")
            self.update_stats()

    # ===== 清空路径叠加（不改变底图与障碍） =====
    def clear_path(self, *args):
        # 清空记录
        self.path_trad, self.exp_trad = [], set()
        self.path_sig,  self.exp_sig  = [], set()

        if self.path_line is not None:
            try: self.path_line.remove()
            except Exception: pass
            self.path_line = None
        if self.explore_scatter is not None:
            try: self.explore_scatter.remove()
            except Exception: pass
            self.explore_scatter = None
        for p in self.start_goal_patches:
            try: p.remove()
            except Exception: pass
        self.start_goal_patches = []
        self.update_stats()
        self.fig.canvas.draw_idle()

    # ===== 新增：路径局部密度对比曲线（仍使用离散路径节点计算，不平滑数据本身） =====
    def plot_density_comparison(self, radius=1, smooth_k=7):
        """
        生成独立图窗，对比传统A*与Sigmoid递增A*的路径“局部障碍密度”序列。
        - 顶部子图：两条密度序列 + 移动平均平滑 + 均值线 + 极值标注
        - 底部子图：差值( Sigmoid - 传统 )，绿色为优势、红色为劣势
        注：此处对比使用原始离散路径节点的密度，不受 Chaikin 可视化平滑影响。
        """
        W, H = self.W, self.H
        obstacles = self.current_obstacles()

        # 若尚未计算路径，则自动计算
        if not self.path_trad:
            self.path_trad, self.exp_trad = astar_hex(W, H, self.start, self.end, obstacles, improved=False)
        if not self.path_sig:
            self.path_sig,  self.exp_sig  = astar_hex(W, H, self.start, self.end, obstacles, improved=True)

        if not self.path_trad or not self.path_sig:
            print("至少有一条路径不存在，无法绘制对比曲线。请先确保两种算法都能找到路径。")
            return

        # 计算两条路径的局部密度序列
        def path_density_series(path):
            return np.array([local_density_ratio(obstacles, p, W, H, radius) for p in path], dtype=float)

        y_trad = path_density_series(self.path_trad)
        y_sig  = path_density_series(self.path_sig)

        # 平滑（移动平均）
        y_trad_s = moving_average(y_trad, k=smooth_k)
        y_sig_s  = moving_average(y_sig,  k=smooth_k)

        x_trad = np.arange(len(y_trad))
        x_sig  = np.arange(len(y_sig))

        # 均值
        m_trad = float(np.mean(y_trad))
        m_sig  = float(np.mean(y_sig))

        # 画图
        fig = plt.figure(figsize=(11, 7.5))
        gs = fig.add_gridspec(2, 1, height_ratios=[2.2, 1], hspace=0.25)

        # 顶部：两条路径的局部密度
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(x_trad, y_trad, color=COL_PATH_TRAD, alpha=0.35, linewidth=1.2, label="Traditional A* (local density)")
        ax1.plot(x_sig,  y_sig,  color=COL_PATH_SIG,  alpha=0.35, linewidth=1.2, label="Sigmoid Incremental A* (Local Density)")

        # 平滑主线
        ax1.plot(x_trad, y_trad_s, color=COL_PATH_TRAD, linewidth=2.2, label="Traditional A* (Smoothing)")
        ax1.plot(x_sig,  y_sig_s,  color=COL_PATH_SIG,  linewidth=2.2, label=f"Traditional A* (Smoothing)")

        # 区域填充（提高观感）
        ax1.fill_between(x_trad, y_trad_s, m_trad, color=COL_PATH_TRAD, alpha=0.10, linewidth=0)
        ax1.fill_between(x_sig,  y_sig_s,  m_sig,  color=COL_PATH_SIG,  alpha=0.10, linewidth=0)

        # 均值线
        ax1.axhline(m_trad, color=COL_TRAD_MEAN, linestyle="--", linewidth=1.2, label=f"Traditional A* average: {m_trad:.2f}")
        ax1.axhline(m_sig,  color=COL_SIG_MEAN,  linestyle="--", linewidth=1.2, label=f"igmoid average:: {m_sig:.2f}")

        # 极值标注
        t_max_i, t_min_i = int(np.argmax(y_trad_s)), int(np.argmin(y_trad_s))
        s_max_i, s_min_i = int(np.argmax(y_sig_s)),  int(np.argmin(y_sig_s))
        ax1.scatter([t_max_i, t_min_i], [y_trad_s[t_max_i], y_trad_s[t_min_i]],
                    color=COL_PATH_TRAD, s=28, zorder=5)
        ax1.scatter([s_max_i, s_min_i], [y_sig_s[s_max_i], y_sig_s[s_min_i]],
                    color=COL_PATH_SIG, s=28, zorder=5)
        ax1.text(t_max_i, y_trad_s[t_max_i]+0.02, f"max {y_trad_s[t_max_i]:.2f}", color=COL_PATH_TRAD, fontsize=9, ha='center')
        ax1.text(t_min_i, y_trad_s[t_min_i]-0.04, f"min {y_trad_s[t_min_i]:.2f}", color=COL_PATH_TRAD, fontsize=9, ha='center')
        ax1.text(s_max_i, y_sig_s[s_max_i]+0.02,  f"max {y_sig_s[s_max_i]:.2f}",  color=COL_PATH_SIG,  fontsize=9, ha='center')
        ax1.text(s_min_i, y_sig_s[s_min_i]-0.04,  f"min {y_sig_s[s_min_i]:.2f}",  color=COL_PATH_SIG,  fontsize=9, ha='center')

        ax1.set_title("Comparison of path local obstacle density (traditional A* vs Sigmoid incremental A*)", pad=6)
        ax1.set_xlabel("Path step sequence (index)")
        ax1.set_ylabel("%s）" % S)
        ax1.set_ylim(0, 1.02)
        ax1.grid(alpha=0.25, linewidth=0.6)
        ax1.legend(ncol=2, frameon=False)

        # 底部：差值曲线（Sigmoid - 传统）
        ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
        L = max(len(y_trad_s), len(y_sig_s))
        yT = np.full(L, np.nan); yS = np.full(L, np.nan)
        yT[:len(y_trad_s)] = y_trad_s
        yS[:len(y_sig_s)]  = y_sig_s
        diff = yS - yT

        ax2.axhline(0.0, color="0.25", linewidth=1.0)
        x = np.arange(L)
        ax2.fill_between(x, diff, 0, where=(diff>=0), color=COL_DIFF_FILL_POS, alpha=0.25, linewidth=0)
        ax2.fill_between(x, diff, 0, where=(diff<0),  color=COL_DIFF_FILL_NEG, alpha=0.25, linewidth=0)
        ax2.plot(x, diff, color="0.2", linewidth=1.2)

        # 摘要统计
        pos_adv = np.nanmean(diff[diff>=0]) if np.any(diff>=0) else 0.0
        neg_adv = np.nanmean(diff[diff<0])  if np.any(diff<0) else 0.0
        ax2.text(0.01, 1.02, f"Average positive advantage: {pos_adv:.2f}", transform=ax2.transAxes, fontsize=10, color=COL_DIFF_FILL_POS)
        ax2.text(0.31, 1.02, f"Average negative disadvantage: {neg_adv:.2f}", transform=ax2.transAxes, fontsize=10, color=COL_DIFF_FILL_NEG)

        ax2.set_ylabel("Density difference (Sig - Trad)")
        ax2.set_xlabel("Path step sequence (index)")
        ax2.grid(alpha=0.25, linewidth=0.6)
        ax2.legend(ncol=2, frameon=False, loc="lower right")

        fig.tight_layout()
        out_path = "../path_local_density_comparison.png"
        plt.savefig(out_path, bbox_inches="tight")
        plt.show()
        print("Exported:", out_path)

if __name__ == "__main__":
    mpl.rcParams['font.family'] = 'SimSun'
    plt.rcParams['axes.unicode_minus'] = False
    vis = HexAStarSigmoidIncVisualizer()
    plt.show()
