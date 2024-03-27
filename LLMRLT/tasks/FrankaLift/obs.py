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
        self.q_gripper = self.states["q_gripper"]

        self.cubeB_pos  = self.states["cubeB_pos"]
        
        return self.obs_buf