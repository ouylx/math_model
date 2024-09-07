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
        individual:插槽被墨盒染色的编号组成的二维矩阵
        """
        population = []
        for _ in range(self.population_size):
            indi = [0] * self.n
            individual = np.zeros((self.m, self.n))
            for i in range(package_num):
                index = sorted(random.sample(range(self.n), len(self.print_info[i])))
                for j in range(len(index)):
                    indi[index[j]] = print_info[i][j]
                individual[i, :] = indi
            population.append(individual)
        return population

    def calculate_fitness(self, individual):
        """计算适应度值，问题中为插槽清洗次数越小越好"""
        clean_time = 0
        individual = individual.astype(int)
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
            for element in print_info[i]:
                index.append(np.where(individual[i, :] == element)[0][0])
            for j in range(len(index) - 1):
                if index[j] > index[j + 1]:
                    individual[i, j], individual[i, j + 1] = individual[i, j + 1], individual[i, j]
                    index[j], index[j + 1] = index[j + 1], index[j]
        return individual

    def initial_check(self, individual):
        # 找到数组中的重复元素及其索引
        unique_elements, inverse_indices, counts = np.unique(individual[0, :], return_inverse=True, return_counts=True)
        duplicate_indices = np.where(counts > 1)[0]
        # 对于每个重复元素，将其索引的其中一个更改为 0
        for idx in duplicate_indices:
            duplicate_values = unique_elements[idx]
            duplicate_idxs = np.where(individual[0, :] == duplicate_values)[0]
            # 将重复元素的其中一个索引更改为 0
            individual[0, duplicate_idxs[1]] = 0
        return individual

    def cross_check(self, individual, cross_column):
        for i in range(self.m):
            for element in self.print_info[i]:
                if np.isin(element, individual[i]):
                    pass
                else:
                    individual[i, cross_column] = element
        individual = self.initial_check(individual)
        individual = self.sequence_check(individual)

        return individual

    def crossover(self, parent1, parent2):
        """交叉"""
        child1 = parent1.copy()
        child2 = parent2.copy()
        if random.random() < self.cross_rate:
            crossover_column = random.randint(0, self.n - 1)
            child1[:, crossover_column] = parent2[:, crossover_column]  # 行交叉
            child2[:, crossover_column] = parent1[:, crossover_column]
            child1 = self.cross_check(child1, crossover_column)
            child2 = self.cross_check(child2, crossover_column)
        return child1, child2

    def mutate(self, individual):
        """变异"""
        if random.random() < self.mutation_rate:
            i = random.randint(0, self.m - 1)
            idx1, idx2 = random.sample(range(self.n), 2)  # 随机选择一行中的两个点进行交换
            point1 = individual[i, idx1].copy()
            point2 = individual[i, idx2].copy()
            individual[i, idx1], individual[i, idx2] = point2, point1
            individual = self.sequence_check(individual)
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
            if arr[i, j] in seen:
                arr[i, j] = 0
            else:
                seen.add(arr[i, j])
    return arr


data_1 = pd.read_excel(r'C:\Users\25492\Desktop\2024杭电第14届研究生数学建模竞赛赛题\2024杭电数模校赛——研究生14届赛题B题\附件数据（B题）\附件3'
                       r'\Ins4_30_60_10.xlsx', sheet_name='包装种类及其所需墨盒')
data_2 = pd.read_excel(r'C:\Users\25492\Desktop\2024杭电第14届研究生数学建模竞赛赛题\2024杭电数模校赛——研究生14届赛题B题\附件数据（B题）\附件3'
                       r'\Ins4_30_60_10.xlsx', sheet_name='墨盒切换时间', index_col=0)
CleanTime = np.array(data_2)
CleanTime = np.vstack((np.array([0] * CleanTime.shape[1]), CleanTime))
CleanTime = np.hstack((np.array([0] * CleanTime.shape[0]).reshape(-1, 1), CleanTime))
package_num = 30
box_num = 60
slot_num = 10
print_info = []
for i in range(package_num):
    num = re.findall(r'\d+', data_1.iloc[i, 1])
    print_info.append([int(j) for j in num])
# 遗传算法参数
population_size = 100
generations = 1000
mutation_rate = 0.1
cross_rate = 0.8
problem = Problem(box_num, package_num, slot_num, print_info, CleanTime, population_size, generations, cross_rate,
                  mutation_rate)
best_individual, best_fitness, Solution_time = problem.genetic_algorithm()
solution = best_individual[np.argmax(best_fitness)]
box_solution = keep_first_occurrence(solution)
result = np.around(1 / np.max(best_fitness))
print('Ins4_30_60_10')
print('墨盒放置方案：\n', box_solution)
print('最小清洗时间：', result)
print('计算时间：', Solution_time)
df = pd.DataFrame(box_solution)
df.to_csv('问题三Ins4_30_60_10遗传算法解.csv')
