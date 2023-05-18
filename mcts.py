#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author : Administrator
# date   : 2018/6/26
import numpy as np
import pandas as pd
from TSGenv import env
from copy import deepcopy

class Node:
    def __init__(self, state, parent=None):
        self.state = deepcopy(state)
        self.untried_actions = env.get_available_actions(self,state=self.state.curstate)
        self.parent = parent
        self.children = {}
        self.Q = 0  # 节点最终收益价值
        self.N = 0  # 节点被访问的次数

    def weight_func(self, c_param=1.4):
        # zyh: 如果这个结点已经访问过，那么就计算他的UCB值大小
        if self.N != 0:
            # tip： 这里使用了-self.Q 因为子节点的收益代表的是对手的收益
            # zyh:这里的 W 其实算的就是UCB值，由于CITO中不存在对手，所以不用将其的值设置为负的
            w = self.Q / self.N + c_param * np.sqrt(2 * np.log(self.parent.N) / self.N)
        else:
            # zyh: 如果没有访问过，那么其UCB的值就是0，这里在文章中显示的应该是无穷大的才对
            #     w = 0.0
            w = 1000.0
        return w

    @staticmethod
    def get_random_action(available_actions):
        '''
        随机从可以使用的动作中抽取动作
        :param available_actions: 有效动作空间
        :return:                返回动作对应的index值
        '''
        action_number = len(available_actions)
        action_index = np.random.choice(range(action_number))
        return available_actions[action_index]

    def select(self, c_param=1.4):
        """
        根据当前的子节点情况选择最优的动作并返回子节点
        :param c_param: 探索参数用于探索的比例
        :return: 最优动作，最优动作下的子节点
        """
        weights = [child_node.weight_func(c_param) for child_node in self.children.values()]
        action_temp = pd.Series(data=weights, index=self.children.keys())
        # zyh: idxmax()是取value值最大对应的索引
        action = action_temp.idxmax()
        # action = pd.Series(data=weights, index=self.children.keys()).idxmax()
        next_node = self.children[action]
        return action, next_node

    def expand(self):
        """
        扩展子节点并返回刚扩展的子节点
        :return: 刚扩展出来的子节点
        """
        # 从没有尝试的节点中选择           pop操作直接将此动作弹出
        action = self.untried_actions.pop()

        # 获得下一步的局面
        next_state = deepcopy(self.state)
        next_state.step(action)
        child_node = Node(next_state, self)
        self.children[action] = child_node  # self.children{acation:child_node} 是个字典
        return child_node

    def update(self, winner):
        """
        经过模拟之后更新节点的价值和访问次数
        :param winner: 返回模拟的胜者
        :return:
        """
        self.N += 1
        self.Q += winner

        if self.is_root_node():
            # 向上父节点也要更新
            self.parent.update(winner)

    def rollout(self):
        """
        从当前节点进行蒙特卡洛模拟返回模拟结果
        :return: 模拟结果
        """
        current_state = deepcopy(self.state)
        while True:
            is_over, winner = current_state.get_state_result()
            if is_over:
                break
            available_actions = current_state.get_available_actions(current_state.curstate)  # 获取可以使用的动作空间
            action = Node.get_random_action(available_actions)
            current_state.step(action)
        return winner

    def is_full_expand(self):
        """
        检测节点是否是已经完全扩展了
        :return: 返回节点是否完全扩展
        """
        return len(self.untried_actions) == 0

    def is_root_node(self):
        """
        检测节点是否是根节点
        如果是根节点返回None
        否则返回父节点
        :return: 返回节点是否是根节点
        """
        return self.parent

class MCTS:
    def __init__(self):
        self.root = None
        self.current_node = None

    def __str__(self):
        return "monte carlo tree search ai"

    def simulation(self, count=1000):
        """
        用于模拟蒙特卡罗搜索
        :param count: 模拟的次数
        :return:
        """
        for _ in range(count):
            leaf_node = self.simulation_policy()  # 找到叶子结点
            winner = leaf_node.rollout()
            leaf_node.update(winner)

    def simulation_policy(self):
        """
        模拟过程中找到当前的叶子节点
        :return: 叶子节点
        """
        current_node = self.current_node
        while True:
            is_over, _ = current_node.state.get_state_result()
            if is_over:
                break
            if current_node.is_full_expand():  # 当前结点是否已经完全被探索过，如果已经完全探索了之后
                # 就是当前结点下可选择的动作是否为 0 ，如果为0，就在当前状态下选择一个动作，并得到下一个动作
                _, current_node = current_node.select()
            else:
                return current_node.expand()
        leaf_node = current_node
        return leaf_node

    def take_action(self, current_state):
        """
        蒙特卡罗模拟选择最优动作
        :param current_state: 当前的状态
        :return: 最优动作
        """
        if not self.root:  # 第一次初始化
            self.root = Node(current_state, None)
            self.current_node = self.root
            # 遍历当前结点的子结点，如果这个结点已经被访问过
            # 之所以会有这么一些内容，是因为源代码中，是AI下完一步棋，
            # 人类也要下一步，人类下完一步之后得到状态，
            # 然后再从AI的状态中遍历是否AI的当前状态的子状态可以遇到过这种
            # 但是在CITO中，不需要这些，因为只有AI自己在不断的选择动作

            # for child_node in self.current_node.children.values():  # 跳转到合适的状态
            #     if child_node.state == current_state:
            #         self.current_node = child_node
            #         break
            # else:  # 游戏重新开始的情况下
            #     self.current_node = self.root
        self.simulation(500)  # 每一次选择都是模拟200次之后的选择

        # 下面这一步是经过200步模拟之后，做出的对于当前状态下较好的动作
        # action : 是选择出来的最佳的动作，next_node : 是选最佳动作之后，得到的状态对应的结点。
        action, next_node = self.current_node.select(0.0)
        self.current_node = next_node  # 跳转到对手状态上
        return action
