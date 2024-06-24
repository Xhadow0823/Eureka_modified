class FrankaCubeStackRRR(VecTask):    
    def compute_observations(self):
        self._refresh()
        obs = ["cubeA_quat", "cubeA_pos", "cubeA_to_cubeB_pos", "eef_pos", "eef_quat"]
        obs += ["q_gripper"]
        obs += ["FSM_p"]
        self.obs_buf = torch.cat([self.states[ob].view(self.num_envs, -1) for ob in obs], dim=-1)

        # here are some important property that you may used to build the reward function
        self.cubeA_size = 0.05
        self.cubeB_size = 0.07
        self.cubeA_pos = self.states["cubeA_pos"]
        self.cubeA_to_cubeB_pos = self.states["cubeA_to_cubeB_pos"]
        self.eef_pos = self.states["eef_pos"]
        self.FSM = self.states["FSM"]  # this is a Tensor (dtype=long) that every agent's current state
        self.a_gripper = self.actions[:, -1]  # The value range of a_gripper is from -1 to 1. When a_gripper is smaller, the gripper is more closed; when a_gripper is larger, the gripper is more open.
        self.cubeA_height = self.states["cubeA_height"]  # this is the height of cubeA from table surface
        self.cubeA_pos_relative = self.states["cubeA_pos_relative"]  # this is a tensor, equal to cubeA's pos - eef'pos

        return self.obs_buf
        