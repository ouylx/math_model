import preprocess
import numpy as np


def manhattan_distance(point1, point2):
    return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])


assignment_role = [1, 11, 7, 9, 4, 8, 3, 5, 10, 6, 0, 2]
k = 1
folder = 'C:/Users/25492/Desktop/集训模型 3（研）/附件3/'
file_names = ['8x8map.txt', '16x16map.txt', '64x64map.txt']
grid, agent_pos, st, ok = preprocess.read_data3(folder + file_names[k])
file_names1 = ['d3_8x8.npy', 'd3_16x16.npy', 'd3_64x64.npy']
file_names2 = ['d_start3_8x8.npy', 'd_start3_16x16.npy', 'd_start3_64x64.npy']
dist = np.load(file_names1[k])
dist_start = np.load(file_names2[k])
cost = manhattan_distance(agent_pos[0],st[assignment_role[0]][0])
for i in range(len(assignment_role)-1):
    r = assignment_role[i]
    r1 = assignment_role[i+1]
    cost += manhattan_distance(st[r][0], st[r][1]) + manhattan_distance(st[r][1], st[r1][0])
cost += manhattan_distance(st[assignment_role[-1]][0] ,st[assignment_role[-1]][1])
print(cost)
