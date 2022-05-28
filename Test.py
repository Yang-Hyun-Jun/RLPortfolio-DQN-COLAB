import Visualizer
import utils
import torch
import numpy as np

from Metrics import Metrics
from Environment import environment
from Agent import agent
from Q_network import Score
from Q_network import Qnet

seed = 1

#넘파이 랜덤 시드 고정
np.random.seed(seed)
#파이토치 랜덤 시드 고정
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQNTester:
    def __init__(self,
                 test_data, balance, cost, trading_unit,
                 min_trading_price, max_trading_price, K):
        self.test_data = test_data

        self.state1_dim = 5
        self.state2_dim = 2
        self.K = K
        self.cost = cost

        self.score_net = Score().to(device)
        self.qnet = Qnet(self.score_net, self.K).to(device)
        self.qnet_target = Qnet(self.score_net, self.K)

        self.balance = balance
        self.trading_unit = trading_unit
        self.min_trading_price = min_trading_price
        self.max_trading_price = max_trading_price

        self.env = environment(chart_data=test_data)
        self.agent = agent(environment=self.env, cost=self.cost, trading_unit=self.trading_unit,
                           qnet=self.qnet, qnet_target=self.qnet_target, K=self.K,
                           lr=0.0, tau=0.0, discount_factor=0.0,
                           min_trading_price=self.min_trading_price,
                           max_trading_price=self.max_trading_price)

        model_path = utils.SAVE_DIR + "/Models" + "/DQNPortfolio.pth"
        self.agent.epsilon = 0
        self.agent.qnet.load_state_dict(torch.load(model_path))
        self.agent.qnet_target.load_state_dict(self.agent.qnet.state_dict())

    def run(self):
        metrics = Metrics()
        self.agent.set_balance(self.balance)
        self.agent.reset()
        self.agent.environment.reset()
        state1 = self.agent.environment.observe()
        state2 = self.agent.portfolio
        steps_done = 0

        while True:
            index, actions, confidences = \
                self.agent.get_action(torch.tensor(state1, device=device).float().view(1, self.K, -1),
                                      torch.tensor(state2, device=device).float().view(1, self.K + 1))

            _, next_state1, next_state2, reward, done = self.agent.step(actions, confidences)

            state1 = next_state1
            state2 = next_state2

            metrics.portfolio_values.append(self.agent.portfolio_value)
            metrics.profitlosses.append(self.agent.profitloss)
            metrics.balances.append(self.agent.balance)
            if steps_done % 1 == 0:
                print(f"balance:{self.agent.balance}")
                print(f"stocks:{self.agent.num_stocks}")
                print(f"actions:{actions}")
            if done:
                print(f"model{self.agent.profitloss}")
                break

        # Benchmark: B&H
        self.agent.set_balance(self.balance)
        self.agent.reset()
        self.agent.environment.reset()
        self.agent.environment.observe()
        while True:
            action = np.zeros(self.K)
            confidence = np.ones(self.K)

            _, next_state1, next_state2, reward, done = self.agent.step(action, confidence)
            metrics.profitlosses_BH.append(self.agent.profitloss)

            if done:
                print(f"B&H{self.agent.profitloss}")
                break

        Vsave_path2 = utils.SAVE_DIR + "/Metrics" + "/Portfolio Value Curve_test"
        Vsave_path4 = utils.SAVE_DIR + "/Metrics" + "/Profitloss Curve_test"
        Msave_path1 = utils.SAVE_DIR + "/Metrics" + "/Portfolio Value_test"
        Msave_path2 = utils.SAVE_DIR + "/Metrics" + "/Profitloss_test"
        Msave_path3 = utils.SAVE_DIR + "/Metrics" + "/Profitloss B&H"
        Msave_path4 = utils.SAVE_DIR + "/Metrics" + "/Balances"

        metrics.get_portfolio_values(save_path=Msave_path1)
        metrics.get_profitlosses(save_path=Msave_path2)
        metrics.get_profitlosses_BH(save_path=Msave_path3)
        metrics.get_balances(save_path=Msave_path4)

        Visualizer.get_portfolio_value_curve(metrics.portfolio_values, save_path=Vsave_path2)
        Visualizer.get_profitloss_curve(metrics.profitlosses, metrics.profitlosses_BH, save_path=Vsave_path4)

