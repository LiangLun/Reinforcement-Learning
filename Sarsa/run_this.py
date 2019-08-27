from maze_env import Maze
from RL_brain import SarsaTable
from RL_brain import SarsaLambdaTable


# Sarsa
'''
def update():
    # 学习100回合
    for episode in range(100):
        # 初始化state的观测值
        observation = env.reset()
        # RL 大脑根据 state 的观测值挑选 action。与Q-Learning区别1：因为后面的action
        action = RL.choose_action(str(observation))
        while True:
            # 更新可视化环境
            env.render()

            # 探索者在环境中实施这个 action, 并得到环境返回的下一个 state 观测值, reward 和 done (是否是掉下地狱或者升上天堂)
            observation_, reward, done = env.step(action)

            # 区别2：下一状态的action_由在当前状态S确定(S先确定下一状态S_，再由S_确定下一状态的action_)
            action_ = RL.choose_action(str(observation_))

            # RL 从这个序列 (state, action, reward, state_, action_)中学习。比Q-Learning的多了action_
            RL.learn(str(observation), action, reward, str(observation_), action_)

            # 将下一个state和action的值传到下一次循环
            observation = observation_
            action = action_

            # 如果掉下地狱或者升上天堂, 这回合就结束了 
            if done:
                break
    # 结束游戏并关闭窗口
    print("game over")
    env.destroy()
'''

# Sarsa(Lambda)
def update():
    # 学习100回合
    for episode in range(100):
        # 初始化state的观测值
        observation = env.reset()
        # RL 大脑根据 state 的观测值挑选 action。与Q-Learning区别1：因为后面的action
        action = RL.choose_action(str(observation))

        # 新回合, 清零。与上面的Sarsa不同之处
        RL.eligibility_trace *= 0

        while True:
            # 更新可视化环境
            env.render()

            # 探索者在环境中实施这个 action, 并得到环境返回的下一个 state 观测值, reward 和 done (是否是掉下地狱或者升上天堂)
            observation_, reward, done = env.step(action)

            # 区别2：下一状态的action_由在当前状态S确定(S先确定下一状态S_，再由S_确定下一状态的action_)
            action_ = RL.choose_action(str(observation_))

            # RL 从这个序列 (state, action, reward, state_, action_)中学习。比Q-Learning的多了action_
            RL.learn(str(observation), action, reward, str(observation_), action_)

            # 将下一个state和action的值传到下一次循环
            observation = observation_
            action = action_

            # 如果掉下地狱或者升上天堂, 这回合就结束了 
            if done:
                break
    # 结束游戏并关闭窗口
    print("game over")
    env.destroy()

if __name__ == "__main__":
    env = Maze()
    # RL = SarsaTable(actions=list(range(env.n_actions)))
    RL = SarsaLambdaTable(actions=list(range(env.n_actions)))


    env.after(100, update)
    env.mainloop()