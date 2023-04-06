from EvaluateQTable import evaluateQTable
from QTable import QTable
import numpy as np

q_table = QTable.loadFromFile("/home/mirjam/Nextcloud/Uni/S7/Projekt/code/BreakoutWithQLearning/qTables/04|06|2023|17:19:25.qtable")

results = []

for _ in range(100):
    score = evaluateQTable(q_table)
    results.append(score)

mean_score = np.mean(results)

print("Mean Score: " + str(mean_score))
