# 开发者：欧阳利鑫
# 开发时间：2023/7/18
import numpy as np
import pulp


class GRA:
    def __init__(self, Q):
        self.m = Q.shape[0]  # agent数
        self.n = Q.shape[1]  # role数
        self.Q = Q

    def GRA_solve(self):
        magent = range(self.m)
        nrole = range(self.n)
        # 定义问题
        PB = pulp.LpProblem("GRA_Problem", sense=pulp.LpMinimize)
        # 定义变量
        x_ij = pulp.LpVariable.dict("x_ij", (magent, nrole), cat=pulp.LpBinary)
        y = pulp.LpVariable("y", cat='Integer', lowBound=0)
        # 目标函数
        PB += y
        # 约束条件
        for i in magent:
            PB += pulp.lpSum(x_ij[i, j] * self.Q[i, j] for j in nrole) - y <= 0
        for j in nrole:
            PB += pulp.lpSum(x_ij[i, j] for i in magent) == 1
        PB.solve(pulp.PULP_CBC_CMD(msg=False))
        T = [v.varValue for v in PB.variables()]
        z = PB.variables()
        return T, pulp.value(PB.objective)


if __name__ == '__main__':
    Q = np.array([[1, 6, 4, 9], [3, 1, 6, 8], ])  # i墨盒切换到j墨盒的时间
    m = Q.shape[0]
    n = Q.shape[1]
    GRA_Problem = GRA(Q)
    ori_T, objective_value = GRA_Problem.GRA_solve()
    T = np.array([[ori_T[i * n + j] for j in range(n)] for i in range(m)])
    print("T=", T)
    print("Total (W)GRA =", objective_value)
