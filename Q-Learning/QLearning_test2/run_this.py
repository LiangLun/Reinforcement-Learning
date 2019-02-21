'''
程序参考莫凡PYTHON
Q-Learning的一个例子--走迷宫
 是一个二维的空间，让探索者学会走迷宫. 黄色的是天堂 (reward 1), 黑色的地狱 (reward -1)
'''

from maze_env import Maze 
from RL_brain import QLearningTable

def update():
    # 学习100回合
    for episode in range(100):
        # 初始化state的观测值
        observation = env.reset()

        while True:
            # 更新可视化环境
            env.render()

            # RL 大脑根据 state 的观测值挑选 action
            action = RL.choose_action(str(observation))

            # 探索者在环境中实施这个 action, 并得到环境返回的下一个 state 观测值, reward 和 done (是否是掉下地狱或者升上天堂)
            observation_, reward, done = env.step(action)

            # RL 从这个序列 (state, action, reward, state_) 中学习
            RL.learn(str(observation), action, reward, str(observation_))

            # 将下一个state的值传到下一次循环
            observation = observation_

            # 如果掉下地狱或者升上天堂, 这回合就结束了 
            if done:
                break
    # 结束游戏并关闭窗口
    print("game over")
    env.destroy()

if __name__ == '__main__':
    # 定义环境
    env = Maze()
    RL = QLearningTable(actions = list(range(env.n_actions)))

    # 开始可视化环境env
    env.after(100, update())
    env.mainloop()