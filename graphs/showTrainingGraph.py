import matplotlib.pyplot as plt
from get_project_root import root_path

project_root = root_path(ignore_cwd=True)
file = "/qNets/breakoutWorking/" + "log"

steps = []
reward = []
epsilon = []

with open(project_root + file, 'r') as file:
    line = file.readline()

    while line:
        values = line.split(" ")
        steps.append(int(values[0]))
        reward.append(float(values[1]))
        epsilon.append(float(values[2].replace('\n', '')))
        line = file.readline()

# Reward
fig, ax = plt.subplots()
ax.plot(steps, reward)
ax.set(xlabel="Steps", ylabel="Running Reward over 100 Episodes", title="Reward")
xlabels = ['{:,.0f}'.format(x) + 'K' for x in ax.get_xticks()/1000]
ax.set_xticklabels(xlabels)
plt.savefig("reward.png")

# Epsilon
fig, ax = plt.subplots()
ax.plot(steps, epsilon)
ax.set(xlabel="Steps", ylabel="Epsilon", title="Epsilon")
ax.set_xticklabels(xlabels)
plt.savefig("epsilon.png")
