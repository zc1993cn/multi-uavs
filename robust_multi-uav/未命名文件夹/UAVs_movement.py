import torch
import numpy as np
import random
from scipy.spatial.distance import pdist, squareform
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation


def draw_trace(trace_matrix):
    # color_vec = ['r', 'b', 'y', 'b', 'm', 'k']
    # entity_vec = ['entity1', 'entity2', 'entity3']
    font = {
        'color': 'k',
        'style': 'oblique',
        'size': 20,
        'weight': 'bold'
    }

    fig = plt.figure()
    ax = Axes3D(fig)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=20, azim=15)
    ax.set_title("Entity Traces", fontdict=font)
    # ax.set_xlim(0, 1000)
    # ax.set_ylim(0, 800)
    # ax.set_zlim(0, 600)

    for i in range(0, 6):
        x_trace = trace_matrix[:, i, 0].detach().numpy()
        y_trace = trace_matrix[:, i, 1].detach().numpy()
        z_trace = trace_matrix[:, i, 2].detach().numpy()

        # ax.scatter3D(x_trace, y_trace, z_trace)  # 绘制散点图 cmap='Blues'
        ax.plot3D(x_trace, y_trace, z_trace)
        # ax.legend()

    plt.show()
    plt.close()


def get_obstacle_coordinate(center, radius):
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
    y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]

    return x, y, z


def draw_dynamic_graph(graph_data, times, center, radius):

    font = {
        'color': 'k',
        'style': 'oblique',
        'size': 20,
        'weight': 'bold'
    }

    fig = plt.figure()
    ax = Axes3D(fig)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=20, azim=15)
    ax.set_title("Entity Traces", fontdict=font)

    for r_i in range(center.shape[0]):
        obs_x, obs_y, obs_z = get_obstacle_coordinate(center[r_i, :], radius)
        # surface plot rstride 值越大，图像越粗糙
        ax.plot_surface(obs_x, obs_y, obs_z, rstride=4, cstride=4, color='b')

    def update(t):
        for i in range(6):
            x_trace = graph_data[0:(t + 1), i, 0].detach().numpy()
            y_trace = graph_data[0:(t + 1), i, 1].detach().numpy()
            z_trace = graph_data[0:(t + 1), i, 2].detach().numpy()

            ax.plot3D(x_trace, y_trace, z_trace)
        # ax.legend()

    ani = FuncAnimation(fig, update, frames=times, interval=500, blit=False, repeat=False)  # 创建动画效果
    plt.show()
    # ani.save('line.gif', writer='pillow')
    plt.close()


def init_position(uav_n):
    uav_pos = np.zeros([uav_n, 3])
    g_r = np.arange(0, 13, 0.1)
    uav_pos[:, 0] = np.array(random.sample(set(g_r), uav_n))
    uav_pos[:, 1] = np.array(random.sample(set(g_r), uav_n))
    uav_pos[:, 2] = np.array(random.sample(set(g_r), uav_n))

    return uav_pos


def create_obstacle(obs_n, ep):
    obs_p = np.zeros([obs_n, 3])
    g_x = np.arange(0, ep[0], 1)
    g_y = np.arange(0, ep[1], 1)
    g_z = np.arange(0, ep[2], 1)

    obs_p[:, 0] = np.array(random.sample(set(g_x), obs_n))
    obs_p[:, 1] = np.array(random.sample(set(g_y), obs_n))
    obs_p[:, 2] = np.array(random.sample(set(g_z), obs_n))

    return obs_p


def calculate_dis_obs(uav_pos, obs_pos):
    expand_mat = np.zeros([obs_pos.shape[0], 3])
    dis_obs = np.zeros([uav_pos.shape[0], obs_pos.shape[0]])
    for i in range(uav_pos.shape[0]):
        expand_mat[:] = uav_pos[i, :]
        diff_mat = expand_mat - obs_pos
        dis_obs[i] = np.sqrt(np.sum(diff_mat * diff_mat, axis=1))

    return dis_obs


def compute_unit_vector(vs, ve):
    v_direct = ve - vs
    modulus = np.sqrt(np.dot(v_direct, v_direct))

    if modulus == 0:
        normal_v = np.zeros(3)

    else:
        normal_v = v_direct / modulus

    return normal_v


def v_repulsion(pos, obj_pos, obj_dis, s_r):
    v_force = np.zeros([obj_dis.shape[0], 3])
    for i in range(obj_dis.shape[0]):
        force_mag = 0
        v_direct = compute_unit_vector(obj_pos[i, :], pos)
        if obj_dis[i] > 0:
            force_mag = (1 / obj_dis[i] - 1 / s_r) * ((1 / obj_dis[i]) ** 3)

        v_force[i] = v_direct * force_mag

    v_rep = np.sum(v_force, axis=0)

    return v_rep


def v_attraction(pos, ep):
    v_direct = compute_unit_vector(pos, ep)
    # force_mag = np.sqrt(np.dot(ep - pos, ep - pos)) * 0.01
    # v_att = v_direct * force_mag
    v_att = v_direct

    return v_att


def v_neighbor(pos, uav_pos, c_r, uav_dis, s_r):
    nei_direct = np.zeros(3)
    for i in range(uav_pos.shape[0]):
        if uav_dis[i] > s_r:
            v_direct = compute_unit_vector(pos, uav_pos[i, :])

            force_mag = 2 * uav_dis[i] - c_r * 0.001
            nei_direct = nei_direct + force_mag * v_direct

    nei_direct = nei_direct

    return nei_direct


class UAVSystem:
    def __init__(self, num, pos_mat, safety_r, comm_r, v_m):
        self.uav_num = num
        self.safety_range = safety_r
        self.position = pos_mat
        self.communication_range = comm_r
        self.velocity = np.zeros([num, 3])
        self.v_max = v_m
        self.start = pos_mat.copy()
        self.alpha = 10
        self.beta = 1000  # 10
        self.gamma = 0.003  # 0.00165

    def get_acceleration(self, pos, dis_uav, dis_obs, ep, obs_r, obs_pos, uav_start):
        dv = np.zeros([4, 3])

        uav_idx = np.where(dis_uav < self.safety_range)[0]
        obs_idx = np.where(dis_obs < (self.safety_range + obs_r))[0]
        comm_idx = np.where(dis_uav < self.communication_range)[0]

        if uav_idx.size > 0:
            uav_pos = self.position[uav_idx, :]
            uav_dis = dis_uav[uav_idx]
            dv[0, :] = self.alpha * v_repulsion(pos, uav_pos, uav_dis, self.safety_range)

        if obs_idx.size > 0:
            obs_pos_tmp = obs_pos[obs_idx, :]
            dis_obs_tmp = dis_obs[obs_idx]
            dv[1, :] = self.alpha * v_repulsion(pos, obs_pos_tmp, dis_obs_tmp, self.safety_range)

        curr_to_final = np.sqrt(np.dot(ep - pos, ep - pos))
        start_to_final = np.sqrt(np.dot(ep - uav_start, ep - uav_start))
        cff = curr_to_final / start_to_final
        # cff = 1

        if curr_to_final > eps:
            dv[2, :] = self.beta * v_attraction(pos, ep)

        if comm_idx.size > 0:
            uav_cmm_pos = self.position[comm_idx, :]
            uav_dis_comm = dis_uav[comm_idx]
            dv[3, :] = cff * self.gamma * v_neighbor(pos, uav_cmm_pos, self.communication_range,
                                               uav_dis_comm, self.safety_range)

        resultant_dv = np.sum(dv, axis=0)

        return resultant_dv

    def get_next_position(self, obs_pos, ep, obs_r):
        acceleration = np.zeros([self.uav_num, 3])

        dis_uav_uav_tmp = squareform(pdist(self.position))
        dis_uav_uav = dis_uav_uav_tmp + np.eye(self.uav_num) * 100.0
        dis_uav_obs = calculate_dis_obs(self.position, obs_pos)

        for i in range(self.uav_num):
            acceleration[i, :] = self.get_acceleration(self.position[i, :], dis_uav_uav[i, :], dis_uav_obs[i, :],
                                                       ep, obs_r, obs_pos, self.start[i, :])

            velocity_tmp = self.velocity[i, :] + acceleration[i, :]
            v_length = np.sqrt(np.dot(velocity_tmp, velocity_tmp))
            if self.v_max < v_length:
                self.velocity[i, :] = self.v_max * velocity_tmp / v_length

            else:
                self.velocity[i, :] = velocity_tmp

        self.position = self.position + self.velocity

        return self.position


if __name__ == '__main__':
    uav_num = 32     # 32
    obstacle_num = 50  # 100
    step = 200     # 200

    eps = 5
    v_max = 2.5  # 2
    safety_range = 0.1
    obstacle_r = 5
    communication_range = 10

    start_point = np.array([0, 0, 0])
    end_point = np.array([135, 120, 110])

    all_position = torch.zeros([step, uav_num, 3])

    uav_position = init_position(uav_num)
    obs_position = create_obstacle(obstacle_num, end_point)

    us_obj = UAVSystem(uav_num, uav_position, safety_range, communication_range, v_max)

    all_position[0, :, :] = torch.from_numpy(uav_position)

    for it in range(step - 1):
        next_position = us_obj.get_next_position(obs_position, end_point, obstacle_r)
        all_position[it+1, :, :] = torch.from_numpy(next_position)

    draw_dynamic_graph(all_position, step, obs_position, obstacle_r)
    # torch.save(all_position, './trace_data/uav_trace_32_adap_neb_3')
    # torch.save(all_position, 'uav_trace_train_32')

    # print(all_position[:, 30, :])


