import torch
import Visualizer
import numpy as np

from Environment import environment
from Agent import agent
from ReplayMemory import ReplayMemory
from Q_network import Score
from Q_network import Qnet
from Metrics import Metrics

seed = 1
#넘파이 랜덤 시드 고정
np.random.seed(seed)
#파이토치 랜덤 시드 고정
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQNLearner:
    def __init__(self,
                 lr=1e-4, tau=0.005,
                 discount_factor=0.9, trading_unit=10,
                 batch_size=30, memory_size=100,
                 chart_data=None, K=None, cost=0.0025,
                 min_trading_price=None, max_trading_price=None):

        assert min_trading_price >= 0
        assert max_trading_price > 0
        assert max_trading_price >= min_trading_price

        self.environment = environment(chart_data)
        self.memory = ReplayMemory(max_size=memory_size)
        self.chart_data = chart_data
        self.batch_size = batch_size

        self.EPS_END = 0.05
        self.EPS_START = 0.9
        self.EPS_DECAY = 1e+3

        self.score_net = Score().to(device)
        self.qnet = Qnet(self.score_net, K).to(device)
        self.qnet_target = Qnet(self.score_net, K).to(device)

        self.lr = lr
        self.tau = tau
        self.cost = cost
        self.K = K
        self.discount_factor = discount_factor
        self.min_trading_price = min_trading_price
        self.max_trading_price = max_trading_price
        self.trading_unit = trading_unit

        self.agent = agent(environment=self.environment,
                           qnet=self.qnet, K=self.K,
                           qnet_target=self.qnet_target,
                           lr=self.lr, tau=self.tau, cost=self.cost,
                           discount_factor=self.discount_factor,
                           trading_unit=self.trading_unit,
                           min_trading_price=min_trading_price,
                           max_trading_price=max_trading_price)

    def reset(self):
        self.environment.reset()
        self.agent.reset()

    def prepare_training_inputs(self, sampled_exps):
        states1 = []
        states2 = []
        indice = []
        actions = []
        rewards = []
        next_states1 = []
        next_states2 = []
        dones = []

        for sampled_exp in sampled_exps:
            states1.append(sampled_exp[0])
            states2.append(sampled_exp[1])
            indice.append(sampled_exp[2])
            actions.append(sampled_exp[3])
            rewards.append(sampled_exp[4])
            next_states1.append(sampled_exp[5])
            next_states2.append(sampled_exp[6])
            dones.append(sampled_exp[7])

        states1 = torch.cat(states1, dim=0).float()
        states2 = torch.cat(states2, dim=0).float()
        indice = torch.cat(indice, dim=0)
        actions = torch.cat(actions, dim=0)
        rewards = torch.cat(rewards, dim=0).float()
        next_states1 = torch.cat(next_states1, dim=0).float()
        next_states2 = torch.cat(next_states2, dim=0).float()
        dones = torch.cat(dones, dim=0).float()
        return states1, states2, indice, actions, rewards, next_states1, next_states2, dones


    def run(self, num_episode=None, balance=None):
        self.agent.set_balance(balance)
        metrics = Metrics()
        steps_done = 0

        for episode in range(num_episode):
            self.reset()
            cum_r = 0
            state1 = self.environment.observe()
            state2 = self.agent.portfolio
            while True:
                self.agent.epsilon = self.EPS_END + (self.EPS_START - self.EPS_END) * np.exp(-1.*steps_done/self.EPS_DECAY)
                self.agent.epsilon = 0

                index, action, confidences = \
                    self.agent.get_action(torch.tensor(state1, device=device).float().view(1, self.K, -1),
                                          torch.tensor(state2, device=device).float().view(1, self.K+1))

                m_action, next_state1, next_state2, reward, done = self.agent.step(action, confidences)
                steps_done += 1

                experience = (torch.tensor(state1, device=device).float().view(1, self.K, -1),
                              torch.tensor(state2, device=device).float().view(1, self.K+1),
                              torch.tensor(index, device=device).view(1, -1),
                              torch.tensor(m_action, device=device).view(1, -1),
                              torch.tensor(reward, device=device).float().view(1, -1),
                              torch.tensor(next_state1, device=device).float().view(1, self.K, -1),
                              torch.tensor(next_state2, device=device).float().view(1, self.K+1),
                              torch.tensor(done, device=device).float().view(1, -1))

                self.memory.push(experience)
                cum_r += reward
                state1 = next_state1
                state2 = next_state2

                if steps_done % 300 == 0:
                    p = self.agent.portfolio
                    pv = self.agent.portfolio_value
                    sv = self.agent.portfolio_value_static
                    balance = self.agent.balance
                    stocks = self.agent.num_stocks
                    epsilon = self.agent.epsilon
                    profitloss = self.agent.profitloss
                    loss = self.agent.loss
                    np.set_printoptions(precision=4, suppress=True)
                    print(f"episode:{episode}")
                    print(f"price:{self.environment.get_price()}")
                    print(f"action:{action.reshape(1,-1)}")
                    print(f"maction:{m_action.reshape(1,-1)}")
                    print(f"stocks:{stocks}")
                    print(f"portfolio:{p}")
                    print(f"portfolio value:{pv}")
                    print(f"static value:{sv}")
                    print(f"balance:{balance}")
                    print(f"cum reward:{cum_r}")
                    print(f"epsilon:{epsilon}")
                    print(f"profitloss:{profitloss}")
                    print(f"loss:{loss}")
                    print("===========================================================================================")


                #학습
                if len(self.memory) >= self.batch_size:
                    sampled_exps = self.memory.sample(self.batch_size)
                    sampled_exps = self.prepare_training_inputs(sampled_exps)
                    self.agent.update(*sampled_exps)
                    self.agent.soft_target_update(self.agent.qnet.parameters(), self.agent.qnet_target.parameters())

                #metrics 마지막 episode 대해서만
                if episode == range(num_episode)[-1]:
                    metrics.portfolio_values.append(self.agent.portfolio_value)
                    metrics.profitlosses.append(self.agent.profitloss)

                if done:
                    break

            if episode == range(num_episode)[-1]:
                #metric 계산과 저장
                metrics.get_profitlosses()
                metrics.get_portfolio_values()

                #계산한 metric 시각화와 저장
                Visualizer.get_portfolio_value_curve(metrics.portfolio_values)
                Visualizer.get_profitloss_curve(metrics.profitlosses)

    def save_model(self, path):
        torch.save(self.agent.qnet.state_dict(), path)
