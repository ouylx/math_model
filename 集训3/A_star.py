import heapq

# 定义方向：上、下、左、右
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]  # 可以原地等待
# 计算曼哈顿距离作为启发式函数
def manhattan_distance(point1, point2):
    return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])


# 检查节点是否在网格范围内并且可通行
def is_valid_position(grid, position):
    x, y = position
    return 0 <= x < len(grid) and 0 <= y < len(grid[0]) and grid[x][y] != '@'


# A*算法实现
def a_star(grid, start, goal, constraints):
    open_list = []
    closed_list = set()
    start_node = (
        0 + manhattan_distance(start, goal), 0, manhattan_distance(start, goal), start,
        None)  # (f, g, h, position, parent)
    heapq.heappush(open_list, start_node)

    while open_list:  # 当open_list不为空
        current_node = heapq.heappop(open_list)  # 将f值最小的点选出并从open_list中去除
        f, g, h, current_position, parent = current_node
        # 到达终点
        if current_position == goal:
            path = []
            while current_node:
                path.append(current_node[3])
                current_node = current_node[4]
            return path[::-1]  # 返回从起点到终点的路径

        if current_position in closed_list:
            continue

        for direction in DIRECTIONS:
            neighbor_position = (current_position[0] + direction[0], current_position[1] + direction[1])
            if neighbor_position in closed_list:
                continue
            if not is_valid_position(grid, neighbor_position):
                continue
            if (g + 1, neighbor_position) in constraints:  # 点冲突约束
                continue
            if (g + 1, (current_position, neighbor_position)) in constraints:  # 边冲突约束
                continue
            new_g = g + 1
            new_h = manhattan_distance(neighbor_position, goal)
            new_f = new_h + new_g
            i_list = [i for i, node in enumerate(open_list) if neighbor_position == node[3]]
            if i_list:
                i = i_list[0]  # 正常情况下i_list只有一个元素或者为空
                if new_g < open_list[i][1]:  # 比较g值看是否更新节点
                    open_list.pop(i)  # 确定更新后删除原节点信息
                    neighbor_node = (new_f, new_g, new_h, neighbor_position, current_node)
                    heapq.heappush(open_list, neighbor_node)
                else:
                    continue
            else:
                neighbor_node = (new_f, new_g, new_h, neighbor_position, current_node)
                heapq.heappush(open_list, neighbor_node)
        closed_list.add(current_position)
    return None  # 没有找到路径
