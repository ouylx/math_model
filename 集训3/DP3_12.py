import numpy as np
import time


def tsp_single_robot(dist, dist_start):
    m = dist_start.shape[1]  # 任务数
    # dp[S][j] 表示当前完成任务集合 S，并且最后一个任务是 j 的最小时间
    dp = [[float('inf')] * m for _ in range(1 << m)]
    path = [[-1] * m for _ in range(1 << m)]  # 记录路径

    # 初始状态：从任务 0 开始
    for j in range(m):
        dp[1 << j][j] = dist[j, j] + dist_start[0, j]  # 机器人从初始位置到任务 j

    # 枚举所有可能的状态
    for S in range(1 << m):  # 遍历所有任务集合
        for j in range(m):  # 遍历最后一个任务
            if not (S & (1 << j)):  # 如果任务 j 不在集合 S 中，跳过
                continue
            for i in range(m):  # 尝试添加下一个任务
                if S & (1 << i):  # 如果任务 i 已经在集合 S 中，跳过
                    continue
                # 状态转移，更新 dp[S | (1 << i)][i]
                next_S = S | (1 << i)
                cost = dp[S][j] + dist[j, i] + dist[i, i]
                if cost < dp[next_S][i]:
                    dp[next_S][i] = cost
                    path[next_S][i] = j  # 记录从 j 转移到 i

    # 找到完成所有任务的最小时间
    final_state = (1 << m) - 1
    min_time = min(dp[final_state])
    last_task = dp[final_state].index(min_time)

    # 追溯路径
    optimal_path = []
    current_state = final_state
    while last_task != -1:
        optimal_path.append(last_task)
        prev_task = path[current_state][last_task]
        current_state ^= (1 << last_task)
        last_task = prev_task

    optimal_path.reverse()  # 逆序得到正确的任务顺序
    return min_time, [optimal_path]


if __name__ == '__main__':
    file_names1 = ['d3_8x8.npy', 'd3_16x16.npy', 'd3_64x64.npy']
    file_names2 = ['d_start3_8x8.npy', 'd_start3_16x16.npy', 'd_start3_64x64.npy']
    k = 1
    dist = np.load(file_names1[k])
    dist_start = np.load(file_names2[k])
    n = dist_start.shape[0]
    m = dist_start.shape[1]  # 任务数
    time1 = time.time()
    # 运行函数
    min_time, optimal_paths = tsp_single_robot(dist, dist_start)
    time2 = time.time()
    solution_time = time2 - time1
    print('计算时间：', solution_time)
    print("最小总时间:", min_time)
    print(f"机器人的任务顺序:", optimal_paths)
