import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import matplotlib as mpl
import heapq
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle

mpl.rcParams['font.family'] = 'SimSun'
plt.rcParams['axes.unicode_minus'] = False


class AStarVisualizer:
    def __init__(self):
        self.grid_size = 100
        self.fig = plt.figure(figsize=(14, 10))
        self.ax = plt.axes([0.1, 0.3, 0.6, 0.65])

        # 优化后的颜色映射 - 更适合学术论文
        self.cmap = ListedColormap([
            'white',  # 0=空闲 - 白色
            'black',  # 1=障碍 - 深灰色
            'green',  # 2=起点 - 绿色
            '#F44336',  # 3=终点 - 红色
            '#90CAF9',  # 4=路径1 - 蓝色
            '#FFC107',  # 5=已探索1 - 琥珀色
            '#9C27B0'  # 6=路径2 - 紫色
        ])

        # 初始化网格数据
        self.grid = np.zeros((self.grid_size, self.grid_size))
        self.start = (0, 0)
        self.end = (self.grid_size - 1, self.grid_size - 1)
        self.grid[self.start] = 2
        self.grid[self.end] = 3

        # 绘制初始网格
        self.img = self.ax.imshow(self.grid, cmap=self.cmap, vmin=0, vmax=6)
        self.ax.set_xticks(np.arange(-0.5, self.grid_size, 1), minor=True)
        self.ax.set_yticks(np.arange(-0.5, self.grid_size, 1), minor=True)
        self.ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
        self.ax.set_title("A*路径规划可视化 (带统计信息)")

        # 添加控制元素和信息面板
        self.add_controls()
        self.add_info_panel()

        # 连接事件
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

        # 路径规划相关
        self.path1 = []
        self.path2 = []
        self.explored1 = set()
        self.explored2 = set()
        self.stats = {'path1_length': 0, 'path1_explored': 0,
                      'path2_length': 0, 'path2_explored': 0}

    def add_controls(self):
        # 清除按钮
        ax_clear = plt.axes([0.1, 0.2, 0.12, 0.05])
        self.btn_clear = Button(ax_clear, '清除所有障碍')
        self.btn_clear.on_clicked(self.clear_obstacles)

        # 随机生成按钮
        ax_random = plt.axes([0.24, 0.2, 0.12, 0.05])
        self.btn_random = Button(ax_random, '随机生成障碍')
        self.btn_random.on_clicked(self.random_obstacles)

        # A*算法按钮1
        ax_astar1 = plt.axes([0.38, 0.2, 0.12, 0.05])
        self.btn_astar1 = Button(ax_astar1, '运行A*算法(路径1)')
        self.btn_astar1.on_clicked(lambda event: self.run_astar(4))

        # A*算法按钮2
        ax_astar2 = plt.axes([0.52, 0.2, 0.12, 0.05])
        self.btn_astar2 = Button(ax_astar2, '运行A*算法(路径2)')
        self.btn_astar2.on_clicked(lambda event: self.run_astar(6))

        # 清除路径按钮
        ax_clear_path = plt.axes([0.66, 0.2, 0.12, 0.05])
        self.btn_clear_path = Button(ax_clear_path, '清除路径')
        self.btn_clear_path.on_clicked(self.clear_paths)

    def add_info_panel(self):
        # 信息面板背景
        self.info_ax = plt.axes([0.72, 0.4, 0.25, 0.55])
        self.info_ax.axis('off')

        # 统计信息标题
        self.info_ax.text(0.1, 0.9, "路径规划统计信息",
                          fontsize=14, fontweight='bold')

        # 路径1信息
        self.info_ax.text(0.1, 0.8, "路径1 (蓝色):", fontsize=12)
        self.path1_length_text = self.info_ax.text(0.15, 0.75, "路径长度: 0", fontsize=10)
        self.path1_explored_text = self.info_ax.text(0.15, 0.7, "遍历格子数: 0", fontsize=10)

        # 路径2信息
        self.info_ax.text(0.1, 0.6, "路径2 (紫色):", fontsize=12)
        self.path2_length_text = self.info_ax.text(0.15, 0.55, "路径长度: 0", fontsize=10)
        self.path2_explored_text = self.info_ax.text(0.15, 0.5, "遍历格子数: 0", fontsize=10)

        # 比较信息
        self.comparison_text = self.info_ax.text(0.1, 0.4, "", fontsize=12)

        # 图例
        self.add_legend()

    def add_legend(self):
        # 创建图例
        legend_elements = [
            Rectangle((0, 0), 1, 1, fc='white', ec='black', label='空闲'),
            Rectangle((0, 0), 1, 1, fc='black', label='障碍'),
            Rectangle((0, 0), 1, 1, fc='green', label='起点'),
            Rectangle((0, 0), 1, 1, fc='red', label='终点'),
            Rectangle((0, 0), 1, 1, fc='#90CAF9', label='路径1'),
            Rectangle((0, 0), 1, 1, fc='#FFC107', label='已探索'),
            Rectangle((0, 0), 1, 1, fc='purple', label='路径2')
        ]

        self.info_ax.legend(handles=legend_elements, loc='lower left',
                            bbox_to_anchor=(0.1, 0.1), fontsize=10)

    def update_stats(self):
        # 更新路径1统计
        self.path1_length_text.set_text(f"路径长度: {self.stats['path1_length']}")
        self.path1_explored_text.set_text(f"遍历格子数: {self.stats['path1_explored']}")

        # 更新路径2统计
        self.path2_length_text.set_text(f"路径长度: {self.stats['path2_length']}")
        self.path2_explored_text.set_text(f"遍历格子数: {self.stats['path2_explored']}")

        # 比较信息
        if self.stats['path1_length'] > 0 and self.stats['path2_length'] > 0:
            diff = self.stats['path1_length'] - self.stats['path2_length']
            if diff > 0:
                comp_text = f"路径2比路径1短 {abs(diff)} 格"
            elif diff < 0:
                comp_text = f"路径1比路径2短 {abs(diff)} 格"
            else:
                comp_text = "两条路径长度相同"

            explored_diff = self.stats['path1_explored'] - self.stats['path2_explored']
            if explored_diff > 0:
                comp_text += f"\n路径2少探索 {abs(explored_diff)} 格"
            elif explored_diff < 0:
                comp_text += f"\n路径1少探索 {abs(explored_diff)} 格"
            else:
                comp_text += "\n两条路径探索格子数相同"

            self.comparison_text.set_text(comp_text)

        self.fig.canvas.draw()

    def on_click(self, event):
        if event.inaxes != self.ax:
            return

        x, y = int(round(event.xdata)), int(round(event.ydata))

        # 不改变起点和终点
        if (x, y) == self.start or (x, y) == self.end:
            return

        # 切换障碍物状态
        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
            if self.grid[y, x] == 0:  # 空闲变障碍
                self.grid[y, x] = 1
            elif self.grid[y, x] == 1:  # 障碍变空闲
                self.grid[y, x] = 0
            self.update_display()

    def clear_obstacles(self, event):
        self.grid.fill(0)
        self.grid[self.start] = 2
        self.grid[self.end] = 3
        self.path1 = []
        self.path2 = []
        self.explored1 = set()
        self.explored2 = set()
        self.stats = {'path1_length': 0, 'path1_explored': 0,
                      'path2_length': 0, 'path2_explored': 0}
        self.update_display()
        self.update_stats()
        print("所有障碍物已清除")

    def clear_paths(self, event):
        # 保留障碍物和起点终点，只清除路径和探索区域
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                if self.grid[y, x] in [4, 5, 6]:  # 路径1、已探索、路径2
                    self.grid[y, x] = 0

        # 恢复起点和终点
        self.grid[self.start] = 2
        self.grid[self.end] = 3

        # 重置路径和探索区域
        self.path1 = []
        self.path2 = []
        self.explored1 = set()
        self.explored2 = set()
        self.stats = {'path1_length': 0, 'path1_explored': 0,
                      'path2_length': 0, 'path2_explored': 0}

        self.update_display()
        self.update_stats()
        print("所有路径和探索区域已清除")

    def random_obstacles(self, event):
        # 保留起点和终点
        self.grid.fill(0)
        self.grid[self.start] = 2
        self.grid[self.end] = 3

        # 随机生成约20%的障碍物
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                if (x, y) != self.start and (x, y) != self.end:
                    if np.random.random() < 0.2:
                        self.grid[y, x] = 1

        self.path1 = []
        self.path2 = []
        self.explored1 = set()
        self.explored2 = set()
        self.stats = {'path1_length': 0, 'path1_explored': 0,
                      'path2_length': 0, 'path2_explored': 0}
        self.update_display()
        self.update_stats()
        print("已随机生成障碍物")

    def heuristic(self, a, b):
        # 曼哈顿距离
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def run_astar(self, path_color):
        # 重置之前的路径
        if path_color == 4:  # 路径1
            self.path1 = []
            self.explored1 = set()
            # 清除之前的路径1显示
            for y in range(self.grid_size):
                for x in range(self.grid_size):
                    if self.grid[y, x] == 4 or self.grid[y, x] == 5:
                        self.grid[y, x] = 0
        else:  # 路径2
            self.path2 = []
            self.explored2 = set()
            # 清除之前的路径2显示
            for y in range(self.grid_size):
                for x in range(self.grid_size):
                    if self.grid[y, x] == 6:
                        self.grid[y, x] = 0

        # 恢复起点和终点
        self.grid[self.start] = 2
        self.grid[self.end] = 3

        # 确保起点和终点没有被障碍物阻挡
        if self.grid[self.start[1], self.start[0]] == 1 or self.grid[self.end[1], self.end[0]] == 1:
            print("起点或终点被障碍物阻挡!")
            return

        # A*算法实现
        heap = []
        heapq.heappush(heap, (0, self.start))
        came_from = {}
        cost_so_far = {self.start: 0}
        explored = set()
        explored.add(self.start)

        while heap:
            current = heapq.heappop(heap)[1]

            if current == self.end:
                break

            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:  # 4邻域
                x, y = current[0] + dx, current[1] + dy
                if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                    if self.grid[y, x] == 1:  # 障碍物
                        continue

                    new_cost = cost_so_far[current] + 1
                    if (x, y) not in cost_so_far or new_cost < cost_so_far[(x, y)]:
                        cost_so_far[(x, y)] = new_cost
                        priority = new_cost + self.heuristic(self.end, (x, y))
                        heapq.heappush(heap, (priority, (x, y)))
                        came_from[(x, y)] = current
                        explored.add((x, y))

        # 重建路径
        current = self.end
        path = []
        while current != self.start:
            path.append(current)
            current = came_from.get(current, None)
            if current is None:
                print("找不到路径!")
                return
        path.append(self.start)
        path.reverse()

        # 保存路径和统计信息
        if path_color == 4:
            self.path1 = path
            self.explored1 = explored
            self.stats['path1_length'] = len(path) - 1  # 减去起点
            self.stats['path1_explored'] = len(explored)

            # 绘制已探索区域
            for (x, y) in explored:
                if (x, y) not in path and (x, y) != self.start and (x, y) != self.end:
                    self.grid[y, x] = 5
        else:
            self.path2 = path
            self.explored2 = explored
            self.stats['path2_length'] = len(path) - 1
            self.stats['path2_explored'] = len(explored)

        # 可视化路径
        for (x, y) in path[1:-1]:  # 跳过起点和终点
            if self.grid[y, x] != 4 and self.grid[y, x] != 6:  # 不覆盖已有路径
                self.grid[y, x] = path_color

        self.update_display()
        self.update_stats()
        print("A*算法完成!")

    def update_display(self):
        self.img.set_data(self.grid)
        self.fig.canvas.draw()


if __name__ == "__main__":
    visualizer = AStarVisualizer()
    plt.show()