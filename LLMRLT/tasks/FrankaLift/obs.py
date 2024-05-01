class FrankaLift(VecTask):
    """Rest of the environment definition omitted."""
    def compute_observations(self):
        self._refresh()
        obs = self.obs_name
        self.obs_buf = torch.cat([self.states[ob] for ob in obs], dim=-1)

        # some important observation
        self.cubeA_pos = self.states["cubeA_pos"]
        self.cubeA_quat = self.states["cubeA_quat"]

        self.eef_pos = self.states["eef_pos"]
        self.eef_quat = self.states["eef_quat"]
        self.cubeA_pos_relative = self.states["cubeA_pos_relative"]  # this is the pos of cubeA minus the pos of eef
        self.a_gripper = self.states["a_gripper"]  # this is the action value of gripper, <= 0 means close the gripper

        self.FSM = self.states["FSM"]  # this is a torch.tensor dtype=long for store the current state of each agent
        
        return self.obs_buf