import numpy as np
import gym
from gym import spaces


class TrafficEnv(gym.Env):
    def __init__(self):
        super(TrafficEnv, self).__init__()

        self.observation_space = spaces.Box(
            low=0, high=100, shape=(4,), dtype=np.float32
        )

        self.action_space = spaces.Discrete(4)

        self.state = None

    def reset(self):
        self.state = np.array([
            np.random.randint(5, 50),   # vehicle_count
            np.random.randint(0, 20),   # pedestrian_count
            np.random.randint(0, 3),    # congestion_level
            np.random.randint(0, 2)     # anomaly_flag
        ])
        return self.state

    def step(self, action):
        vehicle, ped, congestion, anomaly = self.state

        # Action effect
        if action == 1:
            congestion = max(congestion - 1, 0)
        elif action == 2:
            congestion = min(congestion + 1, 2)
        elif action == 3 and anomaly == 1:
            congestion = 0

        # Reward function
        reward = -congestion * 5
        if anomaly == 1 and action != 3:
            reward -= 10

        done = False

        self.state = np.array([vehicle, ped, congestion, anomaly])
        return self.state, reward, done, {}






