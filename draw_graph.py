import matplotlib
import matplotlib.pyplot as plt

# x = [2, 3, 4, 5]
# y = [987 ,279, 694, 785]

x = [1000, 4000, 7000, 10000]
y = [987, 3225, 4298, 4587]

plt.xlabel('num_agent_train_steps_per_iter')
plt.ylabel('eval average reward')
plt.title('Walker2d-v4')
plt.xticks(x)
plt.plot(x, y, label='Walker2d', color='b', marker='o', linestyle='-')
plt.show()