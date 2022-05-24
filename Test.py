import DataManager
import Visualizer
import utils
import torch
import numpy as np

from Metrics import Metrics
from Environment import environment
from Agent import agent
from Q_network import Score
from Q_network import Qnet

if __name__ == "__main__":
    stock_code = ["010140", "005930", "034220"]

    path_list = []
    for code in stock_code:
        path = utils.Base_DIR + "/" + code
        path_list.append(path)

    #test data load
    train_data, test_data = DataManager.get_data_tensor(path_list,
                                                        train_date_start="20090101",
                                                        train_date_end="20150101",
                                                        test_date_start="20170102",
                                                        test_date_end=None)

    #dimension
    K = len(stock_code)

    #Test Model load
    score_net = Score()
    qnet = Qnet(score_net, K)
    qnet_target = Qnet(score_net, K)
    # qnet = Qnet(state2_dim=K+1, K=K)
    # qnet_target = Qnet(state2_dim=K+1, K=K)

    balance = 15000000
    min_trading_price = 0
    max_trading_price = 500000

    #Agent
    environment = environment(chart_data=test_data)
    agent = agent(environment=environment,
                  qnet=qnet, K=K,
                  qnet_target=qnet_target,
                  lr = 1e-4, tau = 0.005, discount_factor=0.9,
                  min_trading_price=min_trading_price,
                  max_trading_price=max_trading_price)

    agent.epsilon = 0.0
    #Model parameter load
    model_path = utils.SAVE_DIR + "/DQNPortfolio/Models" + "/DQNPortfolio.pth"
    agent.qnet.load_state_dict(torch.load(model_path))
    agent.qnet_target.load_state_dict(agent.qnet.state_dict())

    #Model Test
    metrics = Metrics()
    agent.set_balance(balance)
    agent.reset()
    agent.environment.reset()
    agent.epsilon = 0
    state1 = agent.environment.observe()
    state2 = agent.portfolio
    steps_done = 0

    while True:
        index, actions, confidences = agent.get_action(torch.tensor(state1).float().view(1, K, -1),
                                                       torch.tensor(state2).float().view(1, K+1))

        _, next_state1, next_state2, reward, done = agent.step(actions, confidences)

        state1 = next_state1
        state2 = next_state2

        metrics.portfolio_values.append(agent.portfolio_value)
        metrics.profitlosses.append(agent.profitloss)
        metrics.balances.append(agent.balance)
        if steps_done % 50 == 0:
            print(f"balance:{agent.balance}")
            print(f"stocks:{agent.num_stocks}")
            # print(f"actions:{actions}")
        if done:
            print(f"model{agent.profitloss}")
            break


    #Benchmark: B&H
    agent.set_balance(balance)
    agent.reset()
    agent.environment.reset()
    agent.epsilon = 0
    state1 = agent.environment.observe()
    state2 = agent.portfolio
    while True:
        action = np.zeros(K)
        confidence = np.ones(K)
        _, next_state1, next_state2, reward, done = agent.step(action, confidence)

        state1 = next_state1
        state2 = next_state2
        metrics.profitlosses_BH.append(agent.profitloss)
        if done:
            print(f"B&H{agent.profitloss}")
            break

    Vsave_path2 = utils.SAVE_DIR + "/DQNPortfolio" + "/Metrics" + "/Portfolio Value Curve_test"
    Vsave_path4 = utils.SAVE_DIR + "/DQNPortfolio" + "/Metrics" + "/Profitloss Curve_test"
    Msave_path1 = utils.SAVE_DIR + "/DQNPortfolio" + "/Metrics" + "/Portfolio Value_test"
    Msave_path2 = utils.SAVE_DIR + "/DQNPortfolio" + "/Metrics" + "/Profitloss_test"
    Msave_path3 = utils.SAVE_DIR + "/DQNPortfolio" + "/Metrics" + "/Profitloss B&H"
    Msave_path4 = utils.SAVE_DIR + "/DQNPortfolio" + "/Metrics" + "/Balances"

    metrics.get_portfolio_values(save_path=Msave_path1)
    metrics.get_profitlosses(save_path=Msave_path2)
    metrics.get_profitlosses_BH(save_path=Msave_path3)
    metrics.get_balances(save_path=Msave_path4)

    Visualizer.get_portfolio_value_curve(metrics.portfolio_values, save_path=Vsave_path2)
    Visualizer.get_profitloss_curve(metrics.profitlosses, metrics.profitlosses_BH, save_path=Vsave_path4)

