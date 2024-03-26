import matplotlib
import matplotlib.pyplot as plt
#
# x = [1000, 4000, 7000, 10000]
# y = [3267, 4459, 4763, 4794]
#
# plt.xlabel('num_agent_train_steps_per_iter')
# plt.ylabel('eval average reward')
# plt.title('Ant-v4')
# plt.xticks(x)
# plt.plot(x, y, label='Ant', color='b', marker='o', linestyle='-')
# plt.show()

x = ['Ant', 'Walker2d']
y = [4788, 5163]

plt.xlabel('env_name')
plt.ylabel('eval average reward')

plt.bar(x, y, color='b')
plt.show()
