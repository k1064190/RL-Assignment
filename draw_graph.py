import matplotlib
import matplotlib.pyplot as plt

x = [1000, 4000, 7000, 10000]
y = [3267, 4459, 4763, 4794]

plt.xlabel('num_agent_train_steps_per_iter')
plt.ylabel('eval average reward')
plt.title('Ant-v4')
plt.xticks(x)
plt.plot(x, y, label='Ant', color='b', marker='o', linestyle='-')
plt.show()
