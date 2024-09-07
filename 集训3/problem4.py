import heapq
from itertools import count
import matplotlib.pyplot as plt
import time
import preprocess
import route_graph
import A_star
import Greedy4
import numpy as np
import DP4


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


def calculate_ori_path(robot, grid, agent_pos, st, roles):
    path_start1 = A_star.a_star(grid, agent_pos[robot], st[roles[robot][0]][0], {})
    path_start2 = A_star.a_star(grid, st[roles[robot][0]][0], st[roles[robot][0]][1], {})[1:]
    path = path_start1 + path_start2
    length = len(path_start1) + len(path_start2)
    len_list = [len(path_start1), length]
    for role in roles[robot][1:]:
        gap_path = A_star.a_star(grid, path[-1], st[role][0], {})[1:]
        role_path = A_star.a_star(grid, st[role][0], st[role][1], {})[1:]
        length += len(gap_path)
        len_list.append(length)
        length += len(role_path)
        len_list.append(length)
        path += gap_path + role_path
    return path, len_list


def calculate_constraint_path(robot, grid, agent_pos, st, roles, constraints, len_list):
    constraints_split = [[] for _ in range(len(len_list))]
    for c in constraints:
        for i in range(1, len(len_list)):
            if len_list[i] > c[0] >= len_list[i - 1]:
                c[0] = c[0] + 1 - len_list[i - 1]
                constraints_split[i].append(c)

    for i, role in zip(range(0, len(constraints_split), 2), roles[robot]):
        if i == 0:
            path_start1 = A_star.a_star(grid, agent_pos[robot], st[roles[robot][0]][0], constraints_split[i])
            path_start2 = A_star.a_star(grid, st[roles[robot][0]][0], st[roles[robot][0]][1],
                                        constraints_split[i + 1])[1:]
            path = path_start1 + path_start2
            new_len_list = [len(path_start1), len(path_start1) + len(path_start2)]
        else:
            gap_path = A_star.a_star(grid, path[-1], st[role][0], constraints_split[i])[1:]
            new_len_list.append(new_len_list[-1] + len(gap_path))
            role_path = A_star.a_star(grid, st[role][0], st[role][1], constraints_split[i + 1])[1:]
            new_len_list.append(new_len_list[-1] + len(role_path))
            path += gap_path + role_path
    return path, new_len_list


# CBS主算法
def cbs(grid, agent_pos, st, roles):
    open_list = []
    counter = count()  # 用于生成唯一的计数器
    paths = [calculate_ori_path(i, grid, agent_pos, st, roles)[0] for i in range(len(agent_pos))]
    root = {
        'constraints': {},
        'len_list': [calculate_ori_path(i, grid, agent_pos, st, roles)[1] for i in range(len(agent_pos))],
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
            new_len_list = node['len_list'][:]
            new_paths[robot] = calculate_constraint_path(robot, grid, agent_pos, st, roles,
                                                         new_constraints.get(robot), new_len_list[robot])[
                0]  # 添加冲突约束重新计算

            new_len_list[robot] = calculate_constraint_path(robot, grid, agent_pos, st, roles,
                                                            new_constraints.get(robot), new_len_list[robot])[1]
            if new_paths[robot]:
                new_node = {
                    'constraints': new_constraints,
                    'len_list': new_len_list,
                    'paths': new_paths,
                    'cost': max(len(path) - 1 for path in new_paths) + conflict_num
                }
                heapq.heappush(open_list, (new_node['cost'], next(counter), new_node))

    return None  # 没有找到解


# 测试用例
if __name__ == "__main__":
    k = 2
    folder = 'D:/数学建模/集训模型 3（研）/附件4/'
    file_names = ['8x8map.txt', '16x16map.txt', '64x64map.txt']
    file_names1 = ['d4_8x8.npy', 'd4_16x16.npy', 'd4_64x64.npy']
    file_names2 = ['d_start4_8x8.npy', 'd_start4_16x16.npy', 'd_start4_64x64.npy']
    grid, agent_pos, st, ok = preprocess.read_data4_1(folder + file_names[k])
    dist = np.load(file_names1[k])
    dist_start = np.load(file_names2[k])
    n = dist_start.shape[0]  # 机器人数量
    m = dist_start.shape[1]  # 任务数
    allowed = [set(), {0, 6, 7, 17, 23, 32, 40, 10}, set(), {1, 8, 5, 20}, set()]  # 64*64
    #allowed = [{0, 1, 5}, set()]  # 16*16
    #allowed = [{0, 1, 2}, set()]  # 8*8
    restricted_tasks = set()
    for i in range(len(allowed)):
        restricted_tasks = allowed[i] | restricted_tasks
    unrestricted_tasks = set(range(m)) - restricted_tasks
    if ok:
        time1 = time.time()
        assignment_roles, _ = Greedy4.greedy_main(dist, dist_start, allowed, unrestricted_tasks)  # 贪心算法求解任务分配
        #_, assignment_roles = DP4.min_cost(dist, dist_start, allowed, unrestricted_tasks)  # 动态规划求解任务分配
        paths = cbs(grid, agent_pos, st, assignment_roles)  # cbs计算
        time2 = time.time()
        solution_time = time2 - time1
        print(f'计算时间：{solution_time}')
        for i, role in enumerate(assignment_roles):
            print(f"机器人 {i} 的任务顺序:", role)
        if paths:
            route_graph.plot_map(grid, paths, st)
            for i, path in enumerate(paths):
                print(f"机器人 {i} 的路径: {path}")
            plt.show()
            time_cost = max([len(path) - 1 for path in paths])
            print(f'最小总时间：{time_cost}')
        else:
            print("未找到可行路径")
