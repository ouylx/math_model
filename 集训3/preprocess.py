import numpy as np
import networkx 
import matplotlib.pyplot as plt


# 读取文件

# D:\git仓库\mathmodel_2024\questions\集训模型 3（研）\附件1\8x8map.txt

def read_data1(path):
    '''
    输入：文件路径
    输出：地图的二维数组，机器人任务列表
    输出：数据是否有误
    '''
    is_ok = True
    # 读取文件
    with open(path, 'r') as file:
        lines = file.readlines()

    # 获取地图大小
    map_size = lines[1].strip().split()
    map_height = int(map_size[0])
    map_width = int(map_size[1])

    # 获取地图的二维数组
    map_array = []
    for i in range(2, 2 + map_height):
        row = lines[i].strip().split()
        # row = [0 if i == '.' else 1 for i in row]
        map_array.append(row)


    # 获取机器人的任务
    tasks = []
    num_tasks = int(lines[3 + map_height].strip())
    for i in range(4 + map_height, 4 + map_height + num_tasks):
        task = list(map(int, lines[i].strip().split()))
        tasks.append([(task[0], task[1]), (task[2], task[3])])

    # # 输出地图的二维数组和机器人任务列表
    # print("地图的二维数组:")
    # print(map_array)

    # print("\n机器人起点和终点的数组:")
    # for task in tasks:
    #     print(f"起点: {task[0]}, 终点: {task[1]}")

    # 是否数据有误
    for task in tasks:
        if map_array[task[0][0]][task[0][1]] == 1 or map_array[task[1][0]][task[1][1]] == 1:
            print("机器人的起点或终点有误"+ str(task))
            is_ok = False
            break

    return map_array, tasks,is_ok


def read_data3(path):
    '''
    输入：文件路径
    输出：地图的二维数组，机器人初始位置，任务列表
    输出：数据是否有误
    '''
    is_ok = True
    # 读取文件
    with open(path, 'r') as file:
        lines = file.readlines()

    # 获取地图大小
    map_size = lines[1].strip().split()
    map_height = int(map_size[0])
    map_width = int(map_size[1])

    # 获取地图的二维数组
    map_array = []
    for i in range(2, 2 + map_height):
        row = lines[i].strip().split()
        # row = [0 if i == '.' else 1 for i in row]
        map_array.append(row)

    # 获取机器人的初始位置
    i += 3
    agent_pos = []
    while lines[i].strip() != 'tasks' and lines[i].strip() != 'task':
        pos = list(map(int, lines[i].strip().split()))
        agent_pos.append((pos[0], pos[1]))
        i += 1


    # 获取机器人的任务
    tasks = []
    ind = i + 1
    num_tasks = int(lines[ind].strip())
    for i in range(ind+1, ind + num_tasks+1):
        task = list(map(int, lines[i].strip().split()))
        tasks.append([(task[0], task[1]), (task[2], task[3])])

    # # 输出地图的二维数组和机器人任务列表
    # print("地图的二维数组:")
    # print(map_array)

    # print("\n机器人起点和终点的数组:")
    # for task in tasks:
    #     print(f"起点: {task[0]}, 终点: {task[1]}")

    # 是否数据有误
    for task in tasks:
        if map_array[task[0][0]][task[0][1]] == 1 or map_array[task[1][0]][task[1][1]] == 1:
            print("机器人的起点或终点有误"+ str(task))
            is_ok = False
            break

    return map_array,agent_pos, tasks,is_ok

def is_valid(x, y, map_array):
    return 0 <= x < len(map_array) and 0 <= y < len(map_array[0])

def plot_graph(map_array,st):
    G = networkx.Graph()

    # 定义四个可能的移动方向：上，下，左，右
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]


    # 遍历地图的每个位置
    for x in range(len(map_array)):
        for y in range(len(map_array[0])):
            if map_array[x][y] == '.':
                # 当前点可通行，添加为图中的一个节点
                G.add_node((x, y))
                # 检查该点的四个方向，添加边（弧）
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if is_valid(nx, ny, map_array) and map_array[nx][ny] == '.':
                        G.add_edge((x, y), (nx, ny))
    
    # 将节点映射到字典中，Key 从 1 开始
    node_dict = {i : node for i, node in enumerate(G.nodes())}
    reverse_node_dict = {v: k for k, v in node_dict.items()}  # 反向字典，便于查找

    # 找出每个点可以到达的邻域的点
    reachable_dict = {}
    for key, node in node_dict.items():
        neighbors = list(G.neighbors(node))
        reachable_keys = [reverse_node_dict[neighbor] for neighbor in neighbors]
        reachable_dict[key] = reachable_keys

    # 找出起止点的Key
    st_key = [(reverse_node_dict[i[0]], reverse_node_dict[i[1]]) for i in st]

                        
        
    # # 输出节点和边
    # print("节点:", G.nodes())
    # print("边:", G.edges())

    # # 可视化图
    # pos = {(x, y): (y, -x) for x, y in G.nodes()}  # 定义节点位置，保持网格结构
    # networkx.draw(G, pos, with_labels=True, node_size=500, node_color="lightblue", font_size=8, font_color="black")
    # # 部分起点node设置为红色
    # networkx.draw_networkx_nodes(G, pos, nodelist=[i[0] for i in st], node_color='r', node_size=500,label='start')
    # # 终点node设置为黄色
    # networkx.draw_networkx_nodes(G, pos, nodelist=[i[1] for i in st], node_color='y', node_size=500,label='target')
    # plt.grid(True)  # 显示网格
    # # 设置图例

    # plt.show()
    # print(G.nodes())

    return G,node_dict,st_key,reverse_node_dict,reachable_dict

def read_data4_1(path):
    '''
    输入：文件路径
    输出：地图的二维数组，机器人初始位置，任务列表
    输出：数据是否有误
    '''
    '''
    输入：文件路径
    输出：地图的二维数组，机器人初始位置，任务列表
    输出：数据是否有误
    '''
    is_ok = True
    # 读取文件
    with open(path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # 获取地图大小
    map_size = lines[1].strip().split()
    map_height = int(map_size[0])
    map_width = int(map_size[1])

    # 获取地图的二维数组
    map_array = []
    for i in range(2, 2 + map_height):
        row = lines[i].strip().split()
        # row = [0 if i == '.' else 1 for i in row]
        map_array.append(row)

    # 获取机器人的初始位置
    i += 3
    agent_pos = []
    while lines[i].strip() != 'constraint':
        pos = list(map(int, lines[i].strip().split()))
        agent_pos.append((pos[0], pos[1]))
        i += 1

    while lines[i].strip() != 'tasks' and lines[i].strip() != 'task':

        i += 1

    # 获取机器人的任务
    tasks = []
    ind = i + 1
    num_tasks = int(lines[ind].strip())
    for i in range(ind+1, ind + num_tasks+1):
        task = list(map(int, lines[i].strip().split()))
        tasks.append([(task[0], task[1]), (task[2], task[3])])

    return map_array,agent_pos, tasks,is_ok




if __name__ == '__main__':

    

    folder = 'D:/git仓库/mathmodel_2024/questions/集训模型 3（研）/附件1/'
    folder = 'D:/git仓库/mathmodel_2024/questions/集训模型 3（研）/附件4/'
    file_names = ['8x8map.txt', '16x16map.txt', '64x64map.txt']

    # maps = []
    # tasks = []
    # is_oks = []
    # for file_name in file_names:
    #     path = folder + file_name
    #     print(f"文件路径: {path}")
    #     graph,task,ok = read_data1(path)
    #     if ok:
    #         print(file_name + "数据无误")
    #         print('地图：\n ',graph)
    #         print('任务：\n ')
    #         for i in task:
    #             print(i)
    #     print("\n\n")


    # map_array,st,ok = read_data1(folder + file_names[0]) #读取附件1数据

    # 读取附件3数据
    map_array,agent_pos,st,is_ok = read_data4_1(folder + file_names[1])

    



    for i in st:
        print(i)

    G,node_dict,st_key,reverse_node_dict,reachable_dict = plot_graph(map_array,st)

    agent_pos_key = [reverse_node_dict[i] for i in agent_pos] # 机器人初始位置的key  

    

    # length = []
    # for i in st:
    #     shortest_path_length = networkx.shortest_path_length(G, source=i[0], target=i[1])
    #     length.append(shortest_path_length)


    # print(length)
    # print('理论最短时间:' + str(min(length)))
                  