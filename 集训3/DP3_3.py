def min_cost_multi_robot(n, m, dist, task_dist):
    # 初始化 dp 表，dp[state] 是一个数组，表示每个机器人在该状态下的任务总时间
    dp = {}
    paths = {}

    def set_dp(state, times, path):
        dp[state] = times
        paths[state] = path

    def get_dp(state):
        return dp.get(state, [float('inf')] * n)

    def get_path(state):
        return paths.get(state, [[] for _ in range(n)])

    # 初始状态：每个机器人可以选择其任意一个任务作为第一个任务
    for k in range(n):
        for j in range(m):
            state = tuple([1 << j if i == k else 0 for i in range(n)])
            times = [0] * n
            times[k] = dist[0][j] + task_dist[j]
            path = [[] for _ in range(n)]
            path[k].append(j)
            set_dp(state, times, path)

    # 枚举所有可能的任务子集组合
    all_states = []
    for state in dp.keys():
        all_states.append(state)

    for current_state in all_states:
        for k in range(n):  # 遍历每个机器人
            current_tasks = current_state[k]
            for j in range(m):  # 遍历所有任务
                if current_tasks & (1 << j):  # 如果任务 j 已经在当前任务集中
                    continue
                # 任务 j 可以由任意机器人执行
                new_state = list(current_state)
                new_state[k] |= (1 << j)  # 更新机器人 k 的任务集
                new_state = tuple(new_state)

                last_task = paths[current_state][k][-1] if paths[current_state][k] else 0
                new_times = get_dp(current_state)[:]
                new_times[k] += dist[last_task][j] + task_dist[j]

                if max(new_times) < max(get_dp(new_state)):  # 更新为更优的状态
                    new_path = [p[:] for p in get_path(current_state)]
                    new_path[k].append(j)
                    set_dp(new_state, new_times, new_path)

    # 寻找覆盖所有任务的最优分配
    final_result = float('inf')
    optimal_paths = [[] for _ in range(n)]

    # 枚举所有可能的机器人任务组合
    for state in dp:
        covered_tasks = set()
        for k in range(n):
            for task in paths[state][k]:
                covered_tasks.add(task)

        if len(covered_tasks) == m:  # 如果所有任务被覆盖
            if max(dp[state]) < final_result:
                final_result = max(dp[state])
                optimal_paths = paths[state]

    return final_result, optimal_paths

# 示例数据
n = 3  # 3个机器人
m = 5  # 5个任务
dist = [
    [0, 2, 9, 10, 6],
    [1, 0, 6, 4, 8],
    [15, 7, 0, 8, 3],
    [6, 3, 12, 0, 7],
    [4, 9, 5, 6, 0]
]
task_dist = [5, 7, 3, 4, 6]  # 每个任务的自身运输距离

# 运行函数
min_time, optimal_paths = min_cost_multi_robot(n, m, dist, task_dist)
print("最小最大完成时间:", min_time)
for i in range(n):
    print(f"机器人 {i} 的任务顺序:", optimal_paths[i])
