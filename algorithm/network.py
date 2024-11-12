# Huawei Hackaton 2024
# Daniel Purroy, Francesc Purroy

import numpy as np
import torch as th
import torch.nn as nn

from dataclasses import dataclass

from feature_extraction import full_read


@dataclass
class Args:
    pass


class Network(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, env.single_action_space.n),
        )

    def forward(self, x):
        return self.network(x / 255.0)


def main():
    pass


if __name__=='__main__':
    main()


