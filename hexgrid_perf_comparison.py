# -*- coding: utf-8 -*-
"""
Hex-grid interactive A* visualizer with multiple variants:
- Original A*
- Chen 2023 improved A*
- Gao 2019 improved A*
- Proposed HDT-A* (density-aware + turning-penalty)

Structure is analogous to the previous square-grid multi-variant code, but
all geometry, neighbors, distances, and density are defined on a hexagonal grid
(odd-r offset layout).
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from matplotlib.patches import RegularPolygon, Rectangle
from matplotlib.collections import PatchCollection
import heapq
from math import sqrt, exp

# ===== Global style =====
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

COL_PATH_ORIG = "#2196F3"
COL_EXP_ORIG  = "#BBDEFB"

COL_PATH_CHEN = "#9C27B0"
COL_EXP_CHEN  = "#CE93D8"

COL_PATH_GAO  = "#FF9800"
COL_EXP_GAO   = "#FFE0B2"

COL_PATH_HDT  = "#4CAF50"
COL_EXP_HDT   = "#A5D6A7"

COL_TRAD_MEAN = "#1E88E5"
COL_HDT_MEAN  = "#2E7D32"

COL_DIFF_FILL_POS = "#43A047"
COL_DIFF_FILL_NEG = "#E53935"

S = "Local obstacle density (radius=1)"

# Parameters for HDT-A* sigmoid density scaling
BETA_SIG = 1.0
H0       = 0.15
ALPHA_MIN= 0.5
ALPHA_MAX= 1.5

# Turning penalty weight for Gao 2019 (turn-time cost)
GAO_TURN_WEIGHT = 0.8

# ===== Hex geometry: odd-r offset <-> cube coordinates =====
def offset_to_cube_odd_r(col, row):
    """
    Odd-r horizontal layout: rows are horizontal, odd rows are shifted right.
    """
    x = col - (row - (row & 1)) // 2
    z = row
    y = -x - z
    return x, y, z

def cube_to_offset_odd_r(x, y, z):
    col = x + (z - (z & 1)) // 2
    row = z
    return int(col), int(row)

# 6 neighbor directions in cube coordinates
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

def hex_range(center, radius, W, H):
    cx, cy = center
    ccx, ccy, ccz = offset_to_cube_odd_r(cx, cy)
    out = []
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            dz = -dx - dy
            if max(abs(dx), abs(dy), abs(dz)) <= radius:
                x = ccx + dx
                y = ccy + dy
                z = ccz + dz
                oc, orow = cube_to_offset_odd_r(x, y, z)
                if 0 <= oc < W and 0 <= orow < H:
                    out.append((oc, orow))
    return out

# ===== Density and helpers =====
def kde_density_hex(W, H, obstacles, beta=1.6, R=3):
    D = np.zeros((H, W), dtype=float)
    if not obstacles:
        return D
    for o in obstacles:
        for p in hex_range(o, R, W, H):
            d = hex_distance(p, o)
            D[p[1], p[0]] += np.exp(- (d * d) / (2.0 * beta * beta))
    vmax = D.max()
    if vmax > 1e-12:
        D /= vmax
    return D

def sigmoid_increasing(Kn):
    return ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) / (1.0 + exp(-BETA_SIG * (Kn - H0)))

def moving_average(y, k=5):
    y = np.asarray(y, dtype=float)
    if y.size == 0:
        return y
    k = max(1, int(k))
    if k == 1:
        return y
    pad = k // 2
    y_pad = np.pad(y, (pad, pad), mode="edge")
    kernel = np.ones(k) / k
    return np.convolve(y_pad, kernel, mode="valid")

def chaikin_smooth(points, iters=2):
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

def compute_turn_angle(prev2, prev1, cur):
    """
    Compute absolute turning angle between vectors prev1-prev2 and cur-prev1 (in radians).
    Using offset coordinates is sufficient to detect direction change on hex grid.
    """
    if prev2 is None or prev1 is None:
        return 0.0
    v1 = (prev1[0] - prev2[0], prev1[1] - prev2[1])
    v2 = (cur[0]  - prev1[0], cur[1]  - prev1[1])
    if (v1[0] == 0 and v1[1] == 0) or (v2[0] == 0 and v2[1] == 0):
        return 0.0
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    det = v1[0] * v2[1] - v1[1] * v2[0]
    angle = abs(np.arctan2(det, dot))
    return float(angle)

class HexAStarMultiVisualizer:
    def __init__(self):
        # Hex grid size (columns x rows)
        self.W = 75
        self.H = 45

        # Hex geometry
        self.hex_radius = 0.5
        self.hex_dx = sqrt(3) * self.hex_radius
        self.hex_dy = 1.5 * self.hex_radius

        self.fig = plt.figure(figsize=(14, 10))
        self.ax = plt.axes([0.1, 0.3, 0.6, 0.65])
        self.ax.set_aspect('equal')

        self.start = (0, 0)
        self.goal  = (self.W - 1, self.H - 1)

        self.grid = np.zeros((self.H, self.W), dtype=int)

        self.path_orig, self.exp_orig = [], set()
        self.path_chen, self.exp_chen = [], set()
        self.path_gao,  self.exp_gao  = [], set()
        self.path_hdt,  self.exp_hdt  = [], set()

        self.path_line = None
        self.explore_scatter = None
        self.start_goal_patches = []
        self.cbar_ax = None

        self.turn_weight_hdt = 0.0

        # Precompute hex centers for each cell
        self.hex_centers = {
            (x, y): (
                (x + 0.5) * self.hex_dx if (y & 1) else x * self.hex_dx,
                y * self.hex_dy
            )
            for y in range(self.H) for x in range(self.W)
        }

        self.render_all()
        self.add_controls()
        self.add_info_panel()
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

    # ===== environment helpers =====
    def current_obstacles(self):
        obs = set()
        for y in range(self.H):
            for x in range(self.W):
                if self.grid[y, x] == 1:
                    obs.add((x, y))
        obs.discard(self.start)
        obs.discard(self.goal)
        return obs

    def local_density_ratio(self, obstacles, node, radius=1):
        cells = hex_range(node, radius, self.W, self.H)
        tot = len(cells)
        if tot == 0:
            return 0.0
        cnt = sum(1 for c in cells if c in obstacles)
        return cnt / float(tot)

    # Approximate Gao-style directional pruning using center vectors
    def allowed_by_direction_hex(self, current, nb, goal):
        cx, cy = self.hex_centers[current]
        gx, gy = self.hex_centers[goal]
        nx, ny = self.hex_centers[nb]
        vg = (gx - cx, gy - cy)
        vs = (nx - cx, ny - cy)
        dot = vg[0] * vs[0] + vg[1] * vs[1]
        # Forbid moves whose direction has non-positive projection towards the goal
        return dot > 0.0

    # Heuristic for Chen 2023 (hybrid: hex distance + Euclidean in the embedding)
    def heuristic_chen(self, node):
        d_hex = hex_distance(node, self.goal)
        cx, cy = self.hex_centers[node]
        gx, gy = self.hex_centers[self.goal]
        d_euc = sqrt((cx - gx)**2 + (cy - gy)**2)
        return 0.5 * (d_hex + d_euc)

    # ===== algorithms =====
    def astar_original(self):
        start, goal = self.start, self.goal
        obstacles = self.current_obstacles()

        open_heap = []
        heapq.heappush(open_heap, (0.0, start))
        came_from = {}
        g_cost = {start: 0.0}
        explored = set([start])

        while open_heap:
            f, current = heapq.heappop(open_heap)
            if current == goal:
                break
            for nb in hex_neighbors(current, self.W, self.H):
                if nb in obstacles:
                    continue
                new_g = g_cost[current] + 1.0
                if nb not in g_cost or new_g < g_cost[nb]:
                    g_cost[nb] = new_g
                    h_nb = hex_distance(nb, goal)
                    f_nb = new_g + h_nb
                    heapq.heappush(open_heap, (f_nb, nb))
                    came_from[nb] = current
                    explored.add(nb)

        path = []
        cur = goal
        while cur != start:
            path.append(cur)
            if cur not in came_from:
                return [], explored, 0, 0.0
            cur = came_from[cur]
        path.append(start)
        path.reverse()

        turns, total_angle = self.count_turns_and_angle(path)
        return path, explored, turns, total_angle

    def astar_chen2023(self):
        start, goal = self.start, self.goal
        obstacles = self.current_obstacles()

        open_heap = []
        heapq.heappush(open_heap, (0.0, start, None))
        came_from = {}
        g_cost = {start: 0.0}
        explored = set([start])

        while open_heap:
            f, current, parent = heapq.heappop(open_heap)
            if current == goal:
                break
            h_cur = self.heuristic_chen(current)
            h_parent = self.heuristic_chen(parent) if parent is not None else h_cur

            for nb in hex_neighbors(current, self.W, self.H):
                if nb in obstacles:
                    continue
                h_nb = self.heuristic_chen(nb)
                # simple pruning: do not go to nodes with much worse heuristic
                if h_nb > min(h_cur, h_parent) + 1.0:
                    continue

                new_g = g_cost[current] + 1.0
                if nb not in g_cost or new_g < g_cost[nb]:
                    g_cost[nb] = new_g
                    f_nb = new_g + h_nb
                    heapq.heappush(open_heap, (f_nb, nb, current))
                    came_from[nb] = current
                    explored.add(nb)

        path = []
        cur = goal
        while cur != start:
            path.append(cur)
            if cur not in came_from:
                return [], explored, 0, 0.0
            cur = came_from[cur]
        path.append(start)
        path.reverse()

        turns, total_angle = self.count_turns_and_angle(path)
        return path, explored, turns, total_angle

    def astar_gao2019(self):
        start, goal = self.start, self.goal
        obstacles = self.current_obstacles()

        open_heap = []
        heapq.heappush(open_heap, (0.0, start, None, None))  # (f, current, prev, prev2)
        came_from = {}
        g_cost = {start: 0.0}
        explored = set([start])

        while open_heap:
            f, current, prev, prev2 = heapq.heappop(open_heap)
            if current == goal:
                break

            for nb in hex_neighbors(current, self.W, self.H):
                if nb in obstacles:
                    continue
                # Gao-style directional pruning
                if not self.allowed_by_direction_hex(current, nb, goal):
                    continue

                angle = compute_turn_angle(prev2, prev, nb)
                turn_cost = GAO_TURN_WEIGHT * (angle / np.pi)

                new_g = g_cost[current] + 1.0 + turn_cost
                if nb not in g_cost or new_g < g_cost[nb]:
                    g_cost[nb] = new_g
                    h = hex_distance(nb, goal)
                    f_nb = new_g + h
                    heapq.heappush(open_heap, (f_nb, nb, current, prev))
                    came_from[nb] = current
                    explored.add(nb)

        path = []
        cur = goal
        while cur != start:
            path.append(cur)
            if cur not in came_from:
                return [], explored, 0, 0.0
            cur = came_from[cur]
        path.append(start)
        path.reverse()

        turns, total_angle = self.count_turns_and_angle(path)
        return path, explored, turns, total_angle

    def astar_hdt(self):
        start, goal = self.start, self.goal
        obstacles = self.current_obstacles()

        open_heap = []
        heapq.heappush(open_heap, (0.0, start, None, None))
        came_from = {}
        g_cost = {start: 0.0}
        explored = set([start])

        while open_heap:
            f, current, prev, prev2 = heapq.heappop(open_heap)
            if current == goal:
                break

            for nb in hex_neighbors(current, self.W, self.H):
                if nb in obstacles:
                    continue

                angle = compute_turn_angle(prev2, prev, nb)
                base = max(hex_distance(current, goal), 1)
                turn_penalty = float(self.turn_weight_hdt) * (angle / np.pi) / base

                new_g = g_cost[current] + 1.0 + turn_penalty
                if nb not in g_cost or new_g < g_cost[nb]:
                    g_cost[nb] = new_g

                    Kn = self.local_density_ratio(obstacles, nb, radius=1)
                    alpha = sigmoid_increasing(Kn)
                    h = alpha * hex_distance(nb, goal)

                    f_nb = new_g + h
                    heapq.heappush(open_heap, (f_nb, nb, current, prev))
                    came_from[nb] = current
                    explored.add(nb)

        path = []
        cur = goal
        while cur != start:
            path.append(cur)
            if cur not in came_from:
                return [], explored, 0, 0.0
            cur = came_from[cur]
        path.append(start)
        path.reverse()

        turns, total_angle = self.count_turns_and_angle(path)
        return path, explored, turns, total_angle

    # ===== statistics =====
    def count_turns_and_angle(self, path):
        if len(path) < 3:
            return 0, 0.0
        turns = 0
        total_angle = 0.0
        for i in range(2, len(path)):
            prev2 = path[i - 2]
            prev1 = path[i - 1]
            cur   = path[i]
            angle = compute_turn_angle(prev2, prev1, cur)
            if angle > 1e-6:
                turns += 1
                total_angle += angle
        return turns, total_angle

    def calculate_obstacle_density(self, node, radius=1):
        obs = self.current_obstacles()
        return self.local_density_ratio(obs, node, radius)

    def calculate_path_density(self, path, radius=1):
        if not path:
            return 0.0
        return float(np.mean([self.calculate_obstacle_density(p, radius) for p in path]))

    # ===== UI =====
    def add_controls(self):
        ax_clear = plt.axes([0.10, 0.20, 0.12, 0.05])
        self.btn_clear = Button(ax_clear, 'Clear obstacles')
        self.btn_clear.on_clicked(self.clear_obstacles)

        ax_random = plt.axes([0.24, 0.20, 0.12, 0.05])
        self.btn_random = Button(ax_random, 'Random obstacles')
        self.btn_random.on_clicked(self.random_obstacles)

        ax_orig = plt.axes([0.38, 0.20, 0.10, 0.05])
        self.btn_orig = Button(ax_orig, 'Original A*')
        self.btn_orig.on_clicked(lambda evt: self.run_and_show("orig"))

        ax_chen = plt.axes([0.50, 0.20, 0.10, 0.05])
        self.btn_chen = Button(ax_chen, 'Chen 2023')
        self.btn_chen.on_clicked(lambda evt: self.run_and_show("chen"))

        ax_gao = plt.axes([0.62, 0.20, 0.10, 0.05])
        self.btn_gao = Button(ax_gao, 'Gao 2019')
        self.btn_gao.on_clicked(lambda evt: self.run_and_show("gao"))

        ax_hdt = plt.axes([0.74, 0.20, 0.12, 0.05])
        self.btn_hdt = Button(ax_hdt, 'HDT-A* (proposed)')
        self.btn_hdt.on_clicked(lambda evt: self.run_and_show("hdt"))

        ax_clear_path = plt.axes([0.10, 0.14, 0.12, 0.05])
        self.btn_clear_path = Button(ax_clear_path, 'Clear paths')
        self.btn_clear_path.on_clicked(self.clear_path)

        ax_compare = plt.axes([0.24, 0.14, 0.22, 0.05])
        self.btn_compare = Button(ax_compare, 'Density: Gao vs HDT')
        self.btn_compare.on_clicked(lambda evt: self.plot_density_comparison())

        ax_tw = plt.axes([0.50, 0.14, 0.24, 0.03])
        self.slider_tw = Slider(ax_tw, 'Turn weight (HDT)', 0.0, 2.0, valinit=self.turn_weight_hdt)
        self.slider_tw.on_changed(self.update_turn_weight_hdt)

    def add_info_panel(self):
        self.info_ax = plt.axes([0.72, 0.40, 0.25, 0.55])
        self.info_ax.axis('off')
        self.info_ax.text(0.1, 0.90, "Path planning statistics", fontsize=13, fontweight='bold')
        self.alg_text      = self.info_ax.text(0.1, 0.82, "Algorithm: -", fontsize=10)
        self.length_text   = self.info_ax.text(0.1, 0.74, "Path length: 0", fontsize=10)
        self.explored_text = self.info_ax.text(0.1, 0.68, "Expanded nodes: 0", fontsize=10)
        self.density_text  = self.info_ax.text(0.1, 0.62, "Avg obstacle density: 0.00", fontsize=10)
        self.turns_text    = self.info_ax.text(0.1, 0.56, "Turning points: 0", fontsize=10)
        self.angle_text    = self.info_ax.text(0.1, 0.50, "Total turning angle (deg): 0.0", fontsize=10)
        self.add_legend()

    def add_legend(self):
        legend_elements = [
            Rectangle((0, 0), 1, 1, fc='white', ec='black', label='Free cell'),
            Rectangle((0, 0), 1, 1, fc=COL_OBS, ec='black', label='Obstacle'),
            Rectangle((0, 0), 1, 1, fc=COL_START, ec='black', label='Start'),
            Rectangle((0, 0), 1, 1, fc=COL_GOAL,  ec='black', label='Goal'),
            Rectangle((0, 0), 1, 1, fc=COL_PATH_ORIG, label='Original A* path'),
            Rectangle((0, 0), 1, 1, fc=COL_PATH_CHEN, label='Chen 2023 path'),
            Rectangle((0, 0), 1, 1, fc=COL_PATH_GAO,  label='Gao 2019 path'),
            Rectangle((0, 0), 1, 1, fc=COL_PATH_HDT,  label='HDT-A* (proposed) path'),
        ]
        self.info_ax.legend(handles=legend_elements, loc='lower left',
                            bbox_to_anchor=(0.0, 0.02), fontsize=9, frameon=False)

    # ===== interaction =====
    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        mx, my = event.xdata, event.ydata
        if mx is None or my is None:
            return
        for (x, y), (cx, cy) in self.hex_centers.items():
            if (mx - cx)**2 + (my - cy)**2 <= (self.hex_radius * 0.9)**2:
                if (x, y) in [self.start, self.goal]:
                    return
                self.grid[y, x] = 0 if self.grid[y, x] == 1 else 1
                self.clear_path()
                self.render_all()
                return

    def clear_obstacles(self, event=None):
        self.grid.fill(0)
        self.clear_path()
        self.render_all()

    def random_obstacles(self, event=None):
        self.grid.fill(0)
        for y in range(self.H):
            for x in range(self.W):
                if (x, y) not in [self.start, self.goal] and np.random.random() < 0.3:
                    self.grid[y, x] = 1
        self.clear_path()
        self.render_all()

    def update_stats_panel(self, alg_name, path, explored, turns=0, total_angle=0.0):
        length = max(0, len(path) - 1)
        explored_n = len(explored)
        density_val = self.calculate_path_density(path, radius=1) if path else 0.0
        self.alg_text.set_text(f"Algorithm: {alg_name}")
        self.length_text.set_text(f"Path length: {length}")
        self.explored_text.set_text(f"Expanded nodes: {explored_n}")
        self.density_text.set_text(f"Avg obstacle density: {density_val:.2f}")
        self.turns_text.set_text(f"Turning points: {int(turns)}")
        self.angle_text.set_text(f"Total turning angle (deg): {np.degrees(total_angle):.1f}")
        self.fig.canvas.draw_idle()

    def reset_stats(self):
        self.alg_text.set_text("Algorithm: -")
        self.length_text.set_text("Path length: 0")
        self.explored_text.set_text("Expanded nodes: 0")
        self.density_text.set_text("Avg obstacle density: 0.00")
        self.turns_text.set_text("Turning points: 0")
        self.angle_text.set_text("Total turning angle (deg): 0.0")
        self.fig.canvas.draw_idle()

    # ===== drawing =====
    def render_all(self):
        self.ax.clear()
        self.ax.set_xlim(-1, self.W * self.hex_dx + 1)
        self.ax.set_ylim(-1, self.H * self.hex_dy + 1)
        self.ax.axis('off')

        obs = self.current_obstacles()
        density = kde_density_hex(self.W, self.H, obs, beta=1.6, R=3)

        patches = []
        faces   = []
        for y in range(self.H):
            for x in range(self.W):
                cx, cy = self.hex_centers[(x, y)]
                poly = RegularPolygon((cx, cy), 6, self.hex_radius,
                                      orientation=0.0,
                                      edgecolor=COL_GRID, linewidth=0.6,
                                      antialiased=True)
                patches.append(poly)
                v = density[y, x]
                if v > 0:
                    c = CMAP(v)
                    faces.append((c[0], c[1], c[2], 0.22 + 0.25 * v))
                else:
                    faces.append((1, 1, 1, 1))

        coll = PatchCollection(patches, match_original=True)
        coll.set_facecolor(faces)
        self.ax.add_collection(coll)

        for (x, y) in obs:
            cx, cy = self.hex_centers[(x, y)]
            poly = RegularPolygon((cx, cy), 6, self.hex_radius * 0.96,
                                  orientation=0.0,
                                  facecolor=COL_OBS, edgecolor=None,
                                  linewidth=0)
            self.ax.add_patch(poly)

        for p, col in [(self.start, COL_START), (self.goal, COL_GOAL)]:
            cx, cy = self.hex_centers[p]
            poly = RegularPolygon((cx, cy), 6, self.hex_radius * 0.90,
                                  orientation=0.0,
                                  facecolor=col, edgecolor="black",
                                  linewidth=0.8, zorder=8)
            self.ax.add_patch(poly)

        if self.cbar_ax is not None:
            try:
                self.cbar_ax.remove()
            except Exception:
                pass
            self.cbar_ax = None
        self.cbar_ax = self.fig.add_axes([0.705, 0.30, 0.012, 0.60])
        norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
        cb = mpl.colorbar.ColorbarBase(self.cbar_ax, cmap=CMAP, norm=norm)
        cb.set_label("Obstacle density (normalized)", rotation=90, labelpad=6)

        self.ax.set_title("Hex grid: obstacle density background (YlOrBr)", pad=6)
        self.fig.canvas.draw_idle()

    # ===== running & plotting =====
    def run_and_show(self, mode):
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

        if mode == "orig":
            path, explored, turns, total_angle = self.astar_original()
            self.path_orig, self.exp_orig = path, explored
            pcolor = COL_PATH_ORIG
            ecolor = COL_EXP_ORIG
            alg_name = "Original A*"
        elif mode == "chen":
            path, explored, turns, total_angle = self.astar_chen2023()
            self.path_chen, self.exp_chen = path, explored
            pcolor = COL_PATH_CHEN
            ecolor = COL_EXP_CHEN
            alg_name = "Chen 2023"
        elif mode == "gao":
            path, explored, turns, total_angle = self.astar_gao2019()
            self.path_gao, self.exp_gao = path, explored
            pcolor = COL_PATH_GAO
            ecolor = COL_EXP_GAO
            alg_name = "Gao 2019"
        elif mode == "hdt":
            path, explored, turns, total_angle = self.astar_hdt()
            self.path_hdt, self.exp_hdt = path, explored
            pcolor = COL_PATH_HDT
            ecolor = COL_EXP_HDT
            alg_name = "HDT-A* (proposed)"
        else:
            return

        if path:
            ex_x, ex_y = [], []
            for (x, y) in explored:
                cx, cy = self.hex_centers[(x, y)]
                ex_x.append(cx)
                ex_y.append(cy)
            self.explore_scatter = self.ax.scatter(
                ex_x, ex_y, s=18, c=ecolor, alpha=0.75,
                linewidths=0, zorder=9
            )

            pts = np.array([self.hex_centers[(x, y)] for (x, y) in path], dtype=float)
            smooth = chaikin_smooth(pts, iters=2)
            self.path_line, = self.ax.plot(
                smooth[:, 0], smooth[:, 1],
                color=pcolor, linewidth=2.0, zorder=10,
                solid_joinstyle='round'
            )

            for p, col in [(self.start, COL_START), (self.goal, COL_GOAL)]:
                cx, cy = self.hex_centers[p]
                poly = RegularPolygon((cx, cy), 6, self.hex_radius * 0.90,
                                      orientation=0.0,
                                      facecolor=col, edgecolor="black",
                                      linewidth=0.8, zorder=11)
                self.start_goal_patches.append(self.ax.add_patch(poly))

            self.update_stats_panel(alg_name, path, explored, turns, total_angle)
            self.fig.canvas.draw_idle()
            print(f"{alg_name} done. Avg local density: {self.calculate_path_density(path):.2f}, "
                  f"Turns: {turns}, Total angle(deg): {np.degrees(total_angle):.1f}")
        else:
            print("No path found.")
            self.reset_stats()

    def clear_path(self, *args):
        self.path_orig, self.exp_orig = [], set()
        self.path_chen, self.exp_chen = [], set()
        self.path_gao,  self.exp_gao  = [], set()
        self.path_hdt,  self.exp_hdt  = [], set()

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
        self.reset_stats()
        self.fig.canvas.draw_idle()

    def plot_density_comparison(self, radius=1, smooth_k=7):
        obs = self.current_obstacles()

        if not self.path_gao:
            self.path_gao, self.exp_gao, _, _ = self.astar_gao2019()
        if not self.path_hdt:
            self.path_hdt, self.exp_hdt, _, _ = self.astar_hdt()

        if not self.path_gao or not self.path_hdt:
            print("Both Gao 2019 and HDT-A* paths are required for comparison.")
            return

        def path_density_series(path):
            return np.array(
                [self.local_density_ratio(obs, p, radius) for p in path],
                dtype=float
            )

        y_gao = path_density_series(self.path_gao)
        y_hdt = path_density_series(self.path_hdt)

        y_gao_s = moving_average(y_gao, k=smooth_k)
        y_hdt_s = moving_average(y_hdt, k=smooth_k)

        x_gao = np.arange(len(y_gao))
        x_hdt = np.arange(len(y_hdt))

        m_gao = float(np.mean(y_gao))
        m_hdt = float(np.mean(y_hdt))

        fig = plt.figure(figsize=(11, 7.5))
        gs = fig.add_gridspec(2, 1, height_ratios=[2.2, 1.0], hspace=0.25)

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(x_gao, y_gao, color=COL_PATH_GAO, alpha=0.35, linewidth=1.2,
                 label="Gao 2019 (Local Density)")
        ax1.plot(x_hdt, y_hdt, color=COL_PATH_HDT, alpha=0.35, linewidth=1.2,
                 label="HDT-A* (Local Density)")

        ax1.plot(x_gao, y_gao_s, color=COL_PATH_GAO, linewidth=2.2,
                 label="Gao 2019 (Smoothed)")
        ax1.plot(x_hdt, y_hdt_s, color=COL_PATH_HDT, linewidth=2.2,
                 label="HDT-A* (Smoothed)")

        ax1.fill_between(x_gao, y_gao_s, m_gao, color=COL_PATH_GAO, alpha=0.10, linewidth=0)
        ax1.fill_between(x_hdt, y_hdt_s, m_hdt, color=COL_PATH_HDT, alpha=0.10, linewidth=0)

        ax1.axhline(m_gao, color=COL_TRAD_MEAN, linestyle="--", linewidth=1.2,
                    label=f"Gao 2019 avg: {m_gao:.2f}")
        ax1.axhline(m_hdt, color=COL_HDT_MEAN, linestyle="--", linewidth=1.2,
                    label=f"HDT-A* avg: {m_hdt:.2f}")

        g_max_i, g_min_i = int(np.argmax(y_gao_s)), int(np.argmin(y_gao_s))
        h_max_i, h_min_i = int(np.argmax(y_hdt_s)), int(np.argmin(y_hdt_s))
        ax1.scatter([g_max_i, g_min_i], [y_gao_s[g_max_i], y_gao_s[g_min_i]],
                    color=COL_PATH_GAO, s=28, zorder=5)
        ax1.scatter([h_max_i, h_min_i], [y_hdt_s[h_max_i], y_hdt_s[h_min_i]],
                    color=COL_PATH_HDT, s=28, zorder=5)
        ax1.text(g_max_i, y_gao_s[g_max_i] + 0.02, f"max {y_gao_s[g_max_i]:.2f}",
                 color=COL_PATH_GAO, fontsize=9, ha='center')
        ax1.text(g_min_i, y_gao_s[g_min_i] - 0.04, f"min {y_gao_s[g_min_i]:.2f}",
                 color=COL_PATH_GAO, fontsize=9, ha='center')
        ax1.text(h_max_i, y_hdt_s[h_max_i] + 0.02, f"max {y_hdt_s[h_max_i]:.2f}",
                 color=COL_PATH_HDT, fontsize=9, ha='center')
        ax1.text(h_min_i, y_hdt_s[h_min_i] - 0.04, f"min {y_hdt_s[h_min_i]:.2f}",
                 color=COL_PATH_HDT, fontsize=9, ha='center')

        ax1.set_title("Comparison of local obstacle density along path (Gao 2019 vs HDT-A*)", pad=6)
        ax1.set_xlabel("Path step index")
        ax1.set_ylabel(S)
        ax1.set_ylim(0, 1.02)
        ax1.grid(alpha=0.25, linewidth=0.6)
        ax1.legend(ncol=2, frameon=False)

        ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
        L = max(len(y_gao_s), len(y_hdt_s))
        yG = np.full(L, np.nan)
        yH = np.full(L, np.nan)
        yG[:len(y_gao_s)] = y_gao_s
        yH[:len(y_hdt_s)] = y_hdt_s
        diff = yH - yG

        ax2.axhline(0.0, color="0.25", linewidth=1.0)
        x = np.arange(L)
        ax2.fill_between(x, diff, 0, where=(diff >= 0), color=COL_DIFF_FILL_POS,
                         alpha=0.25, linewidth=0)
        ax2.fill_between(x, diff, 0, where=(diff < 0), color=COL_DIFF_FILL_NEG,
                         alpha=0.25, linewidth=0)
        ax2.plot(x, diff, color="0.2", linewidth=1.2)

        pos_adv = np.nanmean(diff[diff >= 0]) if np.any(diff >= 0) else 0.0
        neg_adv = np.nanmean(diff[diff < 0])  if np.any(diff < 0) else 0.0
        ax2.text(0.01, 1.02, f"Avg positive advantage: {pos_adv:.2f}",
                 transform=ax2.transAxes, fontsize=10, color=COL_DIFF_FILL_POS)
        ax2.text(0.31, 1.02, f"Avg negative disadvantage: {neg_adv:.2f}",
                 transform=ax2.transAxes, fontsize=10, color=COL_DIFF_FILL_NEG)

        ax2.set_ylabel("Density difference (HDT - Gao)")
        ax2.set_xlabel("Path step index")
        ax2.grid(alpha=0.25, linewidth=0.6)

        fig.tight_layout()
        out_path = "hex_gao_vs_hdt_local_density_comparison.png"
        plt.savefig(out_path, bbox_inches="tight")
        plt.show()
        print("Exported:", out_path)

    # ===== callbacks =====
    def update_turn_weight_hdt(self, value):
        self.turn_weight_hdt = float(value)


if __name__ == "__main__":
    mpl.rcParams["font.family"] = "SimSun"  # if you need Chinese labels somewhere
    plt.rcParams["axes.unicode_minus"] = False
    vis = HexAStarMultiVisualizer()
    plt.show()
