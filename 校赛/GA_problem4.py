import random
import pandas as pd
import re
import numpy as np
import time


class Problem:
    def __init__(self, box_num, package_num, slot_num, print_info, clean_time, population_size, generations, cross_rate,
                 mutation_rate):
        self.b_num = box_num
        self.m = package_num
        self.n = slot_num
        self.print_info = print_info
        self.T = clean_time
        self.population_size = population_size
        self.maxgen = generations
        self.cross_rate = cross_rate
        self.mutation_rate = mutation_rate

    def initialize_population(self):
        """
        初始化种群
        individual:包含编码和包装顺序的列表
        """
        population = []
        for _ in range(self.population_size):
            indi = [0] * self.n
            individual = []
            pack_list = list(range(self.m))
            random.shuffle(pack_list)
            for i in pack_list:
                index = sorted(random.sample(range(self.n), len(self.print_info[i])))
                for j in range(len(index)):
                    indi[index[j]] = print_info[i][j]
                individual.append(indi.copy())  # 下一次打印插槽若没有插入其他墨盒则继承上一个墨盒的颜色
            population.append([np.array(individual), pack_list])  # 后面的individual变成了包含染色体编码和包装顺序的列表
        return population

    def calculate_fitness(self, individual):
        """计算适应度值，问题中为插槽清洗次数越小越好"""
        clean_time = 0
        individual = individual[0].astype(int)
        for i in range(self.m - 1):
            index = individual[i + 1, :] != individual[i, :]
            clean_time += np.sum(self.T[individual[i, index], individual[i + 1, index]])
        fitness = 1 / clean_time
        return fitness

    def selection(self, population):
        """选择操作，二元锦标赛"""
        selected = []
        for _ in range(self.population_size):
            individual1 = random.choice(population)
            individual2 = random.choice(population)
            if self.calculate_fitness(individual1) > self.calculate_fitness(individual2):
                selected.append(individual1)
            else:
                selected.append(individual2)
        return selected

    def sequence_check(self, individual):
        for i in range(self.m):
            index = []
            for element in print_info[individual[1][i]]:
                index.append(np.where(individual[0][i, :] == element)[0][0])
            for j in range(len(index) - 1):
                if index[j] > index[j + 1]:
                    individual[0][i, j], individual[0][i, j + 1] = individual[0][i, j + 1], individual[0][i, j]
                    index[j], index[j + 1] = index[j + 1], index[j]
        return individual

    def initial_check(self, individual):
        """第一组打印墨盒检测，此时应该不存在重复的颜色"""
        # 检测第一行重复元素
        unique_elements, inverse_indices, counts = np.unique(individual[0][0, :], return_inverse=True,
                                                             return_counts=True)
        duplicate_indices = np.where(counts > 1)[0]
        # 对于每个重复元素，将其索引的其中一个更改为 0
        for idx in duplicate_indices:
            duplicate_values = unique_elements[idx]
            duplicate_idxs = np.where(individual[0][0, :] == duplicate_values)[0]
            # 将重复元素的其中一个索引更改为 0
            individual[0][0, duplicate_idxs[1]] = 0
        return individual

    def cross_check(self, individual, cross_column):
        for i in range(self.m):
            for element in self.print_info[individual[1][i]]:
                if np.isin(element, individual[0][i]):
                    pass
                else:
                    individual[0][i, cross_column] = element
        self.initial_check(individual)
        self.sequence_check(individual)

        return individual

    def crossover(self, parent1, parent2):
        """交叉"""
        child1 = parent1.copy()
        child2 = parent2.copy()
        if random.random() < self.cross_rate:
            crossover_column = random.randint(0, self.n - 1)
            child1[0][:, crossover_column] = parent2[0][:, crossover_column]  # 列交叉
            child2[0][:, crossover_column] = parent1[0][:, crossover_column]
            self.cross_check(child1, crossover_column)
            self.cross_check(child2, crossover_column)
        return child1, child2

    def mutation_check(self, individual):
        """检查变异后是否存在颜色不继承"""
        for j in range(self.n):
            for i in range(self.m - 1):
                if individual[0][i + 1, j] == 0 and individual[0][i, j] != 0:
                    individual[0][i + 1, j] = individual[0][i, j]
        return individual

    def mutate(self, individual):
        """变异"""
        if random.random() < self.mutation_rate:
            idx1, idx2 = random.sample(range(self.m), 2)
            point1 = individual[0][idx1, :].copy()
            point2 = individual[0][idx2, :].copy()
            individual[0][idx1, :], individual[0][idx2, :] = point2, point1  # 行交换
            point1 = individual[1][idx1]
            point2 = individual[1][idx2]
            individual[1][idx1], individual[1][idx2] = point2, point1  # 交换包装打印顺序
            self.initial_check(individual)
            self.mutation_check(individual)
        return individual

    def genetic_algorithm(self):
        """主函数"""
        best_fitness = []
        best_individual = []
        t1 = int(round(time.time() * 1000))
        population = self.initialize_population()
        for generation in range(self.maxgen):
            population = sorted(population, reverse=True, key=lambda x: self.calculate_fitness(x))
            selected = self.selection(population)
            next_population = []
            while len(next_population) < self.population_size:
                parent1, parent2 = random.sample(selected, 2)
                child1, child2 = self.crossover(parent1, parent2)
                next_population.append(self.mutate(child1))
                next_population.append(self.mutate(child2))
            best_individual.append(max(population, key=lambda x: self.calculate_fitness(x)))
            best_fitness.append(self.calculate_fitness(best_individual[-1]))
            population = next_population
        t2 = int(round(time.time() * 1000))
        Solution_time = t2 - t1
        return best_individual, best_fitness, Solution_time


def keep_first_occurrence(arr):
    # 遍历数组，只保留第一次出现的重复元素
    for i in range(arr.shape[0]):
        seen = set()
        for j in range(arr.shape[1]):
            if arr[i,j] in seen:
                arr[i,j] = 0
            else:
                seen.add(arr[i,j])
    return arr


data_1 = pd.read_excel(r'C:\Users\25492\Desktop\2024杭电第14届研究生数学建模竞赛赛题\2024杭电数模校赛——研究生14届赛题B题\附件数据（B题）\附件4'
                       r'\Ins4_20_40_10.xlsx', sheet_name='包装种类及其所需墨盒')
data_2 = pd.read_excel(r'C:\Users\25492\Desktop\2024杭电第14届研究生数学建模竞赛赛题\2024杭电数模校赛——研究生14届赛题B题\附件数据（B题）\附件4'
                       r'\Ins4_20_40_10.xlsx', sheet_name='墨盒切换时间', index_col=0)
CleanTime = np.array(data_2)
CleanTime = np.vstack((np.array([0] * CleanTime.shape[1]), CleanTime))
CleanTime = np.hstack((np.array([0] * CleanTime.shape[0]).reshape(-1, 1), CleanTime))
package_num = 20
box_num = 40
slot_num = 10
print_info = []
for i in range(package_num):
    num = re.findall(r'\d+', data_1.iloc[i, 1])
    print_info.append([int(j) for j in num])
# 遗传算法参数
population_size = 100
generations = 500
mutation_rate = 0.5  # 需要增加包装印刷顺序变化的概率
cross_rate = 0.8
problem = Problem(box_num, package_num, slot_num, print_info, CleanTime, population_size, generations, cross_rate,
                  mutation_rate)
best_individual, best_fitness, Solution_time = problem.genetic_algorithm()
solution = best_individual[np.argmax(best_fitness)]
box_solution = keep_first_occurrence(solution[0])
result = np.around(1 / np.max(best_fitness))
print_sequence = [solution[1][i] + 1 for i in range(package_num)]
print('Ins4_20_40_10')
print('墨盒放置方案：\n', box_solution)
print('最优顺序：', print_sequence)
print('最小清洗时间：', result)
print('计算时间：', Solution_time)
df = pd.DataFrame(box_solution, index=print_sequence)
df.to_csv('问题四Ins4_20_40_10遗传算法解.csv')
