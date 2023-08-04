from scipy.spatial.distance import pdist, squareform
from swarm_dqn import DQN
from swarm_dqn import Net
from swarm_dqn import GraphConvolution
import networkx as nx
import numpy as np
import env_swarm
import torch
import matplotlib.pyplot as plt

device = torch.device("cuda:1")

R = 10
TIMES = 5
NODE_NUM = 32


def nx_graph(adj):
    G = nx.Graph()
    tmp = adj

    for i in range(0, len(tmp)):
        G.add_node(i)

    for i in range(0, len(tmp)):
        idx = np.where(tmp[i, :] == 1)
        sq_idx = idx[0]
        for j in range(0, len(sq_idx)):
            G.add_edge(i, sq_idx[j])

    tag = nx.is_connected(G)
    if tag:
        d = nx.diameter(G)
    else:
        d = 0
    # ps = nx.shell_layout(G)
    # nx.draw(G, ps, with_labels=False, node_size=6.png)
    return tag, d


def evaluate_network(adj):
    eval_vec = np.zeros(2)
    # 连通性指标，传输效率指标
    # conn, d = nx_graph(adj)

    # 网络连边
    count = np.sum(adj)

    # eval_vec[0] = conn
    eval_vec[1] = count

    return torch.from_numpy(eval_vec)


def draw_evaluation(e_v):
    data1 = e_v[0, :, 0].detach().numpy()
    data2 = e_v[0, :, 1].detach().numpy()

    plt.plot(data1)
    plt.show()
    plt.close()

    plt.plot(data2)
    plt.show()
    plt.close()


if __name__ == '__main__':

    model_rl = DQN(NODE_NUM)

    evaluate_v = torch.zeros(TIMES, 200, 2)
    all_adj = torch.zeros(TIMES, 200, NODE_NUM, NODE_NUM)

    file_rl = './trained_model/dqn_32_300_0'

    model_rl = torch.load(file_rl)
    model_rl.eval_net.to(device)
    model_rl.target_net.to(device)

    file_path = './trace_data/uav_trace_32_adap_neb_'

    for i in range(TIMES):
        str_num = str(i)
        file_name = file_path + str_num
        # file_name = 'uav_trace_train_32'
        pos_uav = torch.load(file_name)

        A = np.ones([NODE_NUM, NODE_NUM]) - np.eye(NODE_NUM)

        for j in range(200):
            t_A = torch.from_numpy(A)
            all_adj[i, j, :, :] = t_A

            dis_uav = squareform(pdist(pos_uav[j, :, :].numpy()))
            dis_i, v_adj_i = env_swarm.get_uav_input(dis_uav, R)

            nor_dis_i = dis_i / R

            a = model_rl.choose_action(t_A.to(device), nor_dis_i.to(device))

            A_next = env_swarm.step(a + 1, R, dis_uav)
            evaluate_v[i, j, :] = evaluate_network(A_next)

            A = A_next.copy()

        # print(torch.mean(evaluate_v[i, :, 1]))

    #  draw_evaluation(evaluate_v)
    # torch.save(all_adj, "./network_data/uav_network_32_adap_neb")





