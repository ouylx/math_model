import matplotlib.pyplot as plt
import numpy as np
import preprocess


# 转换地图数据为数值
def map_to_array(map_data):
    size = len(map_data)
    array = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if map_data[i][j] == '@':
                array[i, j] = 1  # 不可行点
            else:
                array[i, j] = 0  # 可行点
    return array


# 绘制地图
def plot_map(map_data, paths, sts):
    size = len(map_data)
    array = map_to_array(map_data)

    fig, ax = plt.subplots(figsize=(7, 6))


    # 绘制不可行点
    y, x = np.where(array == 1)
    ax.scatter(x, y, c='black', label='Blocked', s=100)
    for st in sts:
        # 标记起点和终点
        ax.scatter(st[0][1], st[0][0], c='orange', s=100)  # 起点
        ax.scatter(st[1][1], st[1][0], c='green', s=100)  # 终点

    colormap = plt.get_cmap('tab20')  # 使用 tab20 colormap
    colors = colormap(np.linspace(0, 1, len(paths)))  # 获取 11 种颜色
    for i, path in enumerate(paths):
        # 绘制路径
        y = [site[0] for site in path]
        x = [site[1] for site in path]
        ax.plot(x, y, marker='o', linestyle='-', markersize=3, label=f'Path{i}',color=colors[i])

    # 设置坐标轴
    #ax.set_xticks(np.arange(size))
    #ax.set_yticks(np.arange(size))
    ax.set_xticks(np.arange(size + 1) - .5, minor=True)
    ax.set_yticks(np.arange(size + 1) - .5, minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=2)

    # 设置图例
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # 自动调整布局
    plt.tight_layout()

    plt.gca().invert_yaxis()  # 使得原点在左下角
    plt.show()


if __name__ == '__main__':
    folder = 'C:/Users/25492/Desktop/集训模型 3（研）/附件2/'
    file_names = ['8x8map.txt', '16x16map.txt', '64x64map.txt']
    grid, st, ok = preprocess.read_data1(folder + file_names[1])
    plot_map(grid, [], st)
