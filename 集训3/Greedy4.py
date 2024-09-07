import time
import numpy as np
import problem2 as p2
import preprocess


def Calculate_T(cost, assign_roles, remain_roles, d, d_s,allowed,unrestricted_tasks):
    T = np.zeros((len(cost), len(remain_roles)))
    for i in range(len(cost)):
        for j, role_j in enumerate(remain_roles):
            if role_j in allowed[i] or role_j in unrestricted_tasks:
                if not assign_roles[i]:
                    T[i, j] = d_s[i, role_j]
                else:
                    T[i, j] = d[assign_roles[i][-1], role_j] + cost[i]  # path本身长度加上末尾任务终点到该任务起点的步长
            else:
                T[i,j] = float('inf')
    return T


def greedy_main(d, d_s,allowed,unrestricted_tasks):
    m = d_s.shape[0]  # 机器人数量
    n = d_s.shape[1]  # 任务数
    cost = [0]*m # agent的目前行驶距离
    assign_roles = [[] for _ in range(m)]
    remain_roles = list(range(n))
    while remain_roles:
        T = Calculate_T(cost, assign_roles, remain_roles,d,d_s,allowed,unrestricted_tasks)
        flat_index = np.argmin(T)
        i, j = np.unravel_index(flat_index, T.shape)
        role = remain_roles[j]
        assign_roles[i].append(role)
        cost[i] = T[i, j] + d[role, role]
        remain_roles.remove(role)
    return assign_roles,cost

if __name__ == '__main__':
    file_names1 = ['d4_8x8.npy', 'd4_16x16.npy', 'd4_64x64.npy']
    file_names2 = ['d_start4_8x8.npy', 'd_start4_16x16.npy', 'd_start4_64x64.npy']
    k = 1
    dist = np.load(file_names1[k])
    dist_start = np.load(file_names2[k])
    m = dist_start.shape[0]  # 机器人数量
    n = dist_start.shape[1]  # 任务数
    allowed = [{0, 1, 5}, set()]
    restricted_tasks = set()
    for i in range(len(allowed)):
        restricted_tasks = allowed[i] | restricted_tasks
    unrestricted_tasks = set(range(n)) - restricted_tasks
    time1 = time.time()
    assignment,cost = greedy_main(dist, dist_start,allowed,unrestricted_tasks)
    time2 = time.time()
    solution_time = time2 - time1
    print('计算时间：', solution_time)
    print("最小总时间:", max(cost))
    for i in range(m):
        print(f"机器人 {i} 的任务顺序:", assignment[i])

