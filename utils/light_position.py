import torch
import numpy as np


#            z     y
#            '\`  /|\
#              \   |
#               \  |
#                \ |
#                 \|
#  x <--------------------------------
#                  |\
#                  | \
#                  |  \
#                  |   \
#                  |    \
#                  |     \


def light_direction(random=0.5, device='cpu'):
    # theta: [0, pi] angle between the light direction and the positive direction of y axis
    # gamma: [0, pi] angle between the projection of the light on x-z plan and the positive direction of x axis
    angle_range = [np.pi / 4, np.pi * 3 / 4]
    rand_num = np.random.rand()
    if rand_num < random:
        theta, gamma = torch.rand(2) * (angle_range[1] - angle_range[0]) + angle_range[0]
        direction = torch.tensor([[torch.sin(theta) * torch.cos(gamma),
                                   torch.cos(theta),
                                   -torch.sin(theta) * torch.sin(gamma)]]).to(device)
    else:
        direction = torch.tensor([[0.0, 0.0, -3.0]]).to(device)
    return direction


def light_point(random=0.5, device='cpu'):
    rand_num = np.random.rand()
    if rand_num < random:
        direction = light_direction(random=True, device=device)
        dist = np.random.uniform(2, 5)
        position = dist * direction
    else:
        position = torch.tensor([[0.0, 0.0, -10.0]]).to(
            device)  # if not random, set a far distance to simulate the parallel direction light

    return position
