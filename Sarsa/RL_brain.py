
import numpy as np
import pandas as pd

class RL(object):
    def __init__(self, action_space, learning_rate=0.01,reward_decay=0.9,e_greedy=0.1):
        self.actions = action_space # 动作空间
        self.lr = learning_rate # 学习率
        self.gamma = reward_decay # 奖励衰减
        self.epsilon = e_greedy # 贪婪率
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64) # 放行为价值函数的Q表
        # 建立的Q表
        '''
           action1  action2 ...
        0   0.0      0.0  ...

        '''

    # 选行为
    def choose_action(self, observation):
        self.check_state_exist(observation) # 检测state在q_table中是否存在
        # 选择action
        if np.random.uniform() > self.epsilon: # 选择Q值最高的action
            state_action = self.q_table.loc[observation,:]
            # 同一个 state, 可能会有多个相同的 Q action value, 所以我们乱序一下
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else: # 随机选择action
            action = np.random.choice(self.actions)
        return action

    # 检测state是否存在，若不存在，则加入Q表中
    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # 将状态加入q_table中
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index = self.q_table.columns,
                    name = state,
                    )
                )

    def learn(self, *args):
        pass  # 每种的都有点不同, 所以用pass

# Q-Learning, off-policy
class QLearningTable(RL):   # 继承了父类 RL
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.1):
        super(QLearningTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)    # 表示继承关系

    # 学习更新参数, learn 的方法在每种类型中有不一样, 需重新定义
    def learn(self, s, a, r, s_):
        self.check_state_exist(s_) # 检测q_table中是否有s_,没有则加入
        q_predict = self.q_table.loc[s,a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_,:].max()
        else:
            q_target = r # 下个state是终止符
        # 更新相应的q值
        self.q_table.loc[s,a] += self.lr * (q_target - q_predict)

# Sarsa, On-policy
class SarsaTable(RL):   # 继承 RL class
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.1):
        super(SarsaTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)    # 表示继承关系
    def learn(self, s, a, r, s_, a_): # 比Q-Learning多了一个a_
        self.check_state_exist(s_) # 检测q_table中是否有s_,没有则加入
        q_predict = self.q_table.loc[s,a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, a_]  # q_target 基于选好的 a_ 而不是 Q(s_)的最大值
        else:
            q_target = r
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # 更新 q_table    


# Sarsa(Lambda), On-policy
class SarsaLambdaTable(RL):  # 继承 RL class
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.1, trace_decay=0.9):
        super(SarsaLambdaTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)    # 表示继承关系
         # 后向观测算法, eligibility trace.
        self.lambda_ = trace_decay
        self.eligibility_trace = self.q_table.copy()    # 空的 eligibility trace表

    # 检测 state 是否存在 
    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            to_be_append = pd.Series(
                    [0] * len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            self.q_table = self.q_table.append(to_be_append)

            # 同样更新 eligibility trace
            self.eligibility_trace = self.eligibility_trace.append(to_be_append)

    def learn(self, s, a, r, s_, a_):
        # 这部分和 Sarsa 一样
        self.check_state_exist(s_)
        q_predict = self.q_table.ix[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.ix[s_, a_]
        else:
            q_target = r
        error = q_target - q_predict

        # 这里开始不同:
        # 对于经历过的 state-action, 我们让他+1, 证明他是得到 reward 路途中不可或缺的一环
        self.eligibility_trace.ix[s, a] += 1

        # 更有效的方式:
        # self.eligibility_trace.ix[s, :] *= 0
        # self.eligibility_trace.ix[s, a] = 1

        # Q table 更新
        self.q_table += self.lr * error * self.eligibility_trace

        # 随着时间衰减 eligibility trace 的值, 离获取 reward 越远的步, 他的"不可或缺性"越小
        self.eligibility_trace *= self.gamma * self.lambda_