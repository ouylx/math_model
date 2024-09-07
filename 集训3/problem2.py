import heapq
from itertools import count
import matplotlib.pyplot as plt
import time
import preprocess
import route_graph
import A_star


# 检查是否存在顶点或边冲突
def detect_conflicts(paths):
    max_time = max(len(path) for path in paths)
    conflicts = []
    for t in range(max_time):  # 点冲突检测
        occupied_positions = {}
        for i, path in enumerate(paths):
            step = min(t, len(path) - 1)
            pos = path[step]
            if pos in occupied_positions:
                conflicts.append({'type': 'vertex', 'step': [step, occupied_positions[pos][1]],
                                  'robots': [i, occupied_positions[pos][0]]})
            occupied_positions[pos] = [i, step]

        for i, path_i in enumerate(paths):  # 边冲突检测
            if t < len(path_i) - 1:
                for j, path_j in enumerate(paths):
                    if j <= i:
                        continue
                    if t < len(path_j) - 1:
                        if path_i[t] == path_j[t + 1] and path_i[t + 1] == path_j[t]:
                            conflicts.append({'type': 'edge', 'step': [t + 1, t + 1], 'robots': [i, j]})
        if conflicts:
            return conflicts[0], len(conflicts)
        else:
            return None, 0


# CBS主算法
def cbs(grid, starts_goals):
    open_list = []
    counter = count()  # 用于生成唯一的计数器
    paths = [A_star.a_star(grid, start_goal[0], start_goal[1], {}) for start_goal in starts_goals]
    root = {
        'constraints': {},
        'paths': paths,
        'cost': max(len(path) - 1 for path in paths)
    }
    heapq.heappush(open_list, (root['cost'], next(counter), root))

    while open_list:
        _, _, node = heapq.heappop(open_list)

        # 检查冲突
        conflict, conflict_num = detect_conflicts(node['paths'])
        if not conflict:
            return node['paths']

        # 生成两个子问题
        robots = conflict['robots']
        steps = conflict['step']

        for robot, step in zip(robots, steps):
            new_constraints = {**node['constraints']}
            if conflict['type'] == 'vertex':
                new_constraints.setdefault(robot, []).append((step, node['paths'][robot][step]))  # 如果有冲突则
            elif conflict['type'] == 'edge':
                new_constraints.setdefault(robot, []).append(
                    (step, (node['paths'][robot][step - 1], node['paths'][robot][step])))

            new_paths = node['paths'][:]
            new_paths[robot] = A_star.a_star(grid, starts_goals[robot][0], starts_goals[robot][1],
                                      new_constraints.get(robot))  # 添加冲突约束重新计算
            if new_paths[robot]:
                new_node = {
                    'constraints': new_constraints,
                    'paths': new_paths,
                    'cost': max(len(path) - 1 for path in new_paths) + conflict_num
                }
                heapq.heappush(open_list, (new_node['cost'], next(counter), new_node))

    return None  # 没有找到解


# 测试用例
if __name__ == "__main__":
    folder = 'C:/Users/25492/Desktop/集训模型 3（研）/附件2/'
    file_names = ['8x8map.txt', '16x16map.txt', '64x64map.txt']
    grid, st, ok = preprocess.read_data1(folder + file_names[2])
    if ok:
        time1 = time.time()
        paths = cbs(grid, st)
        time2 = time.time()
        solution_time = time2 - time1
        print(f'计算时间：{solution_time}')
        if paths:
            route_graph.plot_map(grid, paths, st)
            for i, path in enumerate(paths):
                print(f"机器人 {i} 的路径: {path}")
            plt.show()
            time_cost = max([len(path) - 1 for path in paths])
            print(f'时间开销：{time_cost}')
        else:
            print("未找到可行路径")
