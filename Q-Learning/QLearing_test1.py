'''
程序参考莫凡PYTHON
Q-Learning的一个简单例子
在一个线段(一维空间)上，最右端是宝藏T，探索者o在线段中，他只能向左或者向右来寻找宝藏。
找到宝藏，则获得奖励，游戏终止。
'''

import numpy as np 
import pandas as pd 
import time

N_STATES = 6 # 线段的长度
ACTIONS = ["left","right"] # 探索者可以选择的动作
EPSILON = 0.9 # 贪婪度 greedy
ALPHA = 0.1 # 学习率
GAMMA = 0.9 # 奖励递减值
MAX_EPISODES = 13 # 最大的回合数
FRESH_TIME = 0.3 # 移动间隔时间

# 建立Q表
# pd.DataFrame建立Q表
def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states,len(actions))), # 6*2的0矩阵
        columns = actions, # column指列名为actions
        )
    return table
# 建立的Q表q_table:
"""
   left  right
0   0.0    0.0
1   0.0    0.0
2   0.0    0.0
3   0.0    0.0
4   0.0    0.0
5   0.0    0.0
"""

# 定义action,用greedy策略选择动作
def choose_action(state, q_table):
    state_actions = q_table.iloc[state, :] # 选出这个state的所有action值
    if(np.random.uniform()>EPSILON) or (state_actions.all() == 0): # 非贪婪 or 或者这个 state 还没有探索过
        action_name = np.random.choice(ACTIONS)
    else:
        action_name = state_actions.argmax() # 在贪婪情况下
    return action_name

# 环境反馈，即下一个状态S_和奖励情况R
def get_env_feedback(S,A):
    if A == 'right':
        if S == N_STATES-2: #终止条件，倒数第二段
            S_ = 'terminal'
            R = 1
        else:
            S_ = S + 1
            R = 0
    else:
        R = 0
        if S == 0: #到达最左端
            S_ = S
        else:
            S_ = S - 1
    return S_, R

# 动作选择后，环境更新。显示界面的变化
def update_env(S, episode, step_counter):
    env_list = ['-']*(N_STATES-1)+['T']
    if S == 'terminal':
        interaction = 'Episode %s: tatal_steps = %s'%(episode+1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                                ', end='')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)

# 强化学习的主循环
def RL():
    q_table = build_q_table(N_STATES, ACTIONS) # 初始化Q表，全0
    for episode in range(MAX_EPISODES): # 回合
        step_counter = 0
        S = 0  # 初始状态
        is_terminated = False # 是否回合结束
        update_env(S, episode, step_counter) # 环境结束

        while not is_terminated:
            A = choose_action(S, q_table) # 选动作
            S_, R = get_env_feedback(S, A) # 实施行为并得到环境的反馈
            q_predict = q_table.loc[S, A] # 估算的Q值
            if S_ != 'terminal':
                q_target = R + GAMMA*q_table.iloc[S_,:].max() # 实际的Q值（回合没结束）
            else:
                q_target = R # 实际的Q值（回合结束）
                is_terminated = True
            # 更新Q表
            q_table.loc[S, A] += ALPHA*(q_target - q_predict) 
            S = S_ # 探索者移动到下一个 state
            # 更新环境
            update_env(S, episode, step_counter+1)
            step_counter += 1
    return q_table

if __name__ == '__main__':
    q_table = RL()
    print("Q表：")
    print(q_table)