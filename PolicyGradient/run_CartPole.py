import gym
from RL_brain import PolicyGradient
import matplotlib.pyplot as plt

RENDER = False  # 在屏幕上显示模拟窗口会拖慢运行速度, 我们等计算机学得差不多了再显示模拟
DISPLAY_REWARD_THRESHOLD = 400  # 当回合总 reward 大于 400 时显示模拟窗口

env = gym.make('CartPole-v0')   # CartPole 这个模拟
env = env.unwrapped     # 取消限制
env.seed(1)     # 普通的 Policy gradient 方法, 使得回合的 variance 比较大, 所以我们选了一个好点的随机种子

print(env.action_space)     # 显示可用 action
print(env.observation_space)    # 显示可用 state 的 observation
print(env.observation_space.high)   # 显示 observation 最高值
print(env.observation_space.low)    # 显示 observation 最低值

# 定义
RL = PolicyGradient(
    n_actions=env.action_space.n,
    n_features=env.observation_space.shape[0],
    learning_rate=0.02,
    reward_decay=0.99,   # gamma
    # output_graph=True,    # 输出 tensorboard 文件
)

for i_episode in range(3000):

    observation = env.reset()
    print('observation', observation)

    while True:
        if RENDER: env.render()

        action = RL.choose_action(observation)
        print('action:', action)

        observation_, reward, done, info = env.step(action)
        print('observation_:', observation_)
        print('observation:', observation)
        print('reward:', reward)
        print('info:', info)

        RL.store_transition(observation, action, reward)    # 存储这一回合的transition

        # 将下一个 state_ 变为 下次循环的 state
        observation = observation_

        # 回合结束
        if done:
            # 一个回合的reward求和
            ep_rs_sum = sum(RL.ep_rs)
            # globals() 函数会以字典类型返回当前位置的全部全局变量
            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                # 这步为什么，没搞懂
                running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
            if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True     # 判断是否显示模拟
            print("episode:", i_episode, "  reward:", int(running_reward))

            # 一个回合的标准化后的收获
            vt = RL.learn() # 学习, 输出 vt, 我们下节课讲这个 vt 的作用

            if i_episode == 0:
                plt.plot(vt)    # plot 这个回合的 vt
                plt.xlabel('episode steps')
                plt.ylabel('normalized state-action value')
                plt.show()
            break

        