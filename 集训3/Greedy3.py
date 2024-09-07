import time
import numpy as np
import problem2 as p2
import preprocess


def Calculate_T(cost, assign_roles, remain_roles, d, d_s):
    T = np.zeros((len(cost), len(remain_roles)))
    for i in range(len(cost)):
        for j, role_j in enumerate(remain_roles):
            if not assign_roles[i]:
                T[i, j] = d_s[i, role_j]
            else:
                T[i, j] = d[assign_roles[i][-1], role_j] + cost[i]  # path本身长度加上末尾任务终点到该任务起点的步长
    return T


def greedy_main(d, d_s):
    m = d_s.shape[0]  # 机器人数量
    n = d_s.shape[1]  # 任务数
    cost = [0]*m # agent的目前行驶距离
    assign_roles = [[] for _ in range(m)]
    remain_roles = list(range(n))
    while remain_roles:
        T = Calculate_T(cost, assign_roles, remain_roles,d,d_s)
        flat_index = np.argmin(T)
        i, j = np.unravel_index(flat_index, T.shape)
        role = remain_roles[j]
        assign_roles[i].append(role)
        cost[i] = T[i, j] + d[role, role]
        remain_roles.remove(role)
    return assign_roles,cost

if __name__ == '__main__':
    file_names1 = ['d3_8x8.npy', 'd3_16x16.npy', 'd3_64x64.npy']
    file_names2 = ['d_start3_8x8.npy', 'd_start3_16x16.npy', 'd_start3_64x64.npy']
    k = 2
    dist = np.load(file_names1[k])
    dist_start = np.load(file_names2[k])
    n = dist_start.shape[0]  # 机器人数量
    m = dist_start.shape[1]  # 任务数
    time1 = time.time()
    assignment,cost = greedy_main(dist, dist_start)
    time2 = time.time()
    solution_time = time2 - time1
    print('计算时间：', solution_time)
    print("最小总时间:", max(cost))
    for i in range(n):
        print(f"机器人 {i} 的任务顺序:", assignment[i])

