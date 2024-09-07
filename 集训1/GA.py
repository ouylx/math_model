import random
import pandas as pd
import re
import numpy as np
import time
import random

def solve_cubic_real(n):
    coefficients = [-3.624e-8, 2.510e-5, -0.001, 0.0069-n]
    roots = np.roots(coefficients)
    # 仅保留实数根
    real_roots = [root.real for root in roots if np.isclose(root.imag, 0)]
    return real_roots
def select_numbers(lower, upper, min_gap):
    numbers = [lower]
    while numbers[-1] + min_gap < upper:
        num = random.randint(numbers[-1] + min_gap, numbers[-1] + 2 * min_gap)
        numbers.append(num)
    return sorted(numbers)


class Problem:
    def __init__(self, temperature, t_max, a, population_size, generations, cross_rate,
                 mutation_rate):
        self.T = temperature
        self.t_max = t_max
        self.a = a
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
            individual = select_numbers(0, self.t_max, 10)
            individual = self.check(individual)
            population.append(individual)
        return population

    def check(self, individual):
        individual = sorted(individual)
        for i in range(len(individual) - 1):
            if individual[i + 1] - individual[i] < 10:
                individual[i + 1] = individual[i] + 10
        return individual

    def calculate_fitness(self, individual):
        """计算适应度值，问题中为每个小时的人数总和最大"""
        indi1 = [num for num in individual if num <= 60]
        indi2 = [num for num in individual if num > 60]
        if indi1[-1] != 60:
            indi1.append(60)
        if indi2[-1] != 120:
            indi2.append(120)
        b1 = []
        b2 = []
        for i in range(len(indi1) - 1):
            t = indi1[i + 1] - indi1[i]
            b1.append(-np.log(0.5) / t - self.a * 10 ** ((self.T[0] - 25) / 5))
        for i in range(len(indi2) - 1):
            t = indi2[i + 1] - indi2[i]
            b2.append(-np.log(0.5) / t - self.a * 10 ** ((self.T[1] - 25) / 5))
        b = np.log(0.5) + self.a * 10 ** ((25 - self.T[0]) / 5) + self.a * 10 ** ((25 - self.T[1]) / 5)
        n1 = int(solve_cubic_real(min(b1))[0])
        n2 = int(solve_cubic_real(min(b2))[0])
        if n1>= 520:
            n1 = 520
        if n2 >= 520:
            n2 = 520
        fitness = n1+n2
        return [fitness,[n1,n2]]

    def selection(self, population):
        """选择操作，二元锦标赛"""
        selected = []
        for _ in range(self.population_size):
            individual1 = random.choice(population)
            individual2 = random.choice(population)
            if self.calculate_fitness(individual1)[0] > self.calculate_fitness(individual2)[0]:
                selected.append(individual1)
            else:
                selected.append(individual2)
        return selected

    def crossover(self, parent1, parent2):
        """交叉"""
        index1 = random.randint(1, len(parent1) - 1)
        cross_part1 = parent1[index1:]
        index2 = None
        for i in range(len(parent2)):
            if parent2[i] >= parent1[index1 - 1] + 10:
                index2 = i
                break
        if index2 is None:
            return parent1, parent2
        cross_part2 = parent2[index2:]
        child1 = parent1[:index1] + cross_part2
        child2 = parent2[:index2] + cross_part1
        child1 = self.check(child1)
        child2 = self.check(child2)
        return child1, child2

    def mutate(self, individual):
        """变异"""
        if random.random() < self.mutation_rate:
            for i in range(len(individual) - 1):
                if individual[i + 1] - individual[i] >= 20:
                    num = individual[i] + 10
                    individual = individual[:i + 1] + [num] + individual[i + 1:]
        individual = self.check(individual)
        return individual

    def genetic_algorithm(self):
        """主函数"""
        best_fitness = []
        best_individual = []
        t1 = int(round(time.time() * 1000))
        population = self.initialize_population()
        for generation in range(self.maxgen):
            population = sorted(population, reverse=True, key=lambda x: self.calculate_fitness(x)[0])
            selected = self.selection(population)
            next_population = []
            while len(next_population) < self.population_size:
                parent1, parent2 = random.sample(selected, 2)
                child1, child2 = self.crossover(parent1, parent2)
                next_population.append(self.mutate(child1))
                next_population.append(self.mutate(child2))
            best_individual.append(max(population, key=lambda x: self.calculate_fitness(x)[0]))
            best_fitness.append(self.calculate_fitness(best_individual[-1]))
            population = next_population
        t2 = int(round(time.time() * 1000))
        Solution_time = t2 - t1
        return best_individual, best_fitness, Solution_time


data = pd.read_csv(r'C:\Users\25492\Desktop\9月9温度.csv')
T = np.array(data.iloc[:,1])
T = np.reshape(T,(4*3))
a = 0.4621
T = np.array([T[i:i + 3] for i in range(0, len(T), 3)])
t_max = 120
# 遗传算法参数
population_size = 200
generations = 500
mutation_rate = 0.2
cross_rate = 0.8
problem = Problem(T[0, :], t_max, a, population_size, generations, cross_rate, mutation_rate)
best_individual, best_fitness, Solution_time = problem.genetic_algorithm()
print('最优解：', best_individual[np.argmax(best_fitness)])
print('b：', np.max(best_fitness))
print('计算时间：', Solution_time, 'ms')
