import problem1
paths = [[(1, 7), (2, 7), (2, 6), (2, 5), (2, 4), (2, 4), (2, 3), (3, 3)],
[(5, 0), (6, 0), (6, 1), (6, 2), (6, 3), (7, 3), (7, 4), (7, 5)],
[(7, 4), (7, 5), (6, 5), (5, 5), (5, 4), (5, 3), (5, 2)],
[(4, 5), (4, 6), (4, 6), (5, 6), (5, 6), (6, 6), (6, 7)],
[(6, 5), (5, 5), (4, 5), (3, 5), (2, 5), (1, 5), (0, 5), (0, 6)],
[(5, 2), (5, 3), (5, 3), (4, 3), (4, 4), (4, 4), (3, 4), (2, 4)],
[(0, 4), (0, 3), (1, 3), (2, 3), (3, 3), (4, 3), (5, 3), (6, 3), (7, 3), (7, 2)],
[(4, 1), (4, 2), (4, 2), (3, 2), (3, 2), (2, 2), (1, 2)]]
conflict = problem1.detect_conflicts(paths)
print(conflict)

