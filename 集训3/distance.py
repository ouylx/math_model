import numpy as np
import A_star
import preprocess

folder = 'C:/Users/25492/Desktop/集训模型 3（研）/附件4/'
file_names = ['8x8map.txt', '16x16map.txt', '64x64map.txt']
grid, agent_pos, st, ok = preprocess.read_data4_1(folder + file_names[2])
m = len(agent_pos)
n = len(st)
d = np.zeros((n, n))  # 每个任务终点到其他任务起点的距离矩阵
for i1, site1 in enumerate(st):
    for i2, site2 in enumerate(st):
        route = A_star.a_star(grid, site1[1], site2[0], {})
        d[i1, i2] = len(route)-1
np.save('d4_64x64.npy', d)
d_start = np.zeros((m, n))
for i in range(m):
    for j in range(n):
        route = A_star.a_star(grid, agent_pos[i], st[j][0],{})
        d_start[i, j] = len(route)-1
np.save('d_start4_64x64.npy', d_start)
