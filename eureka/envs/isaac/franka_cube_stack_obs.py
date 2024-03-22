class FrankaCubeStack(VecTask):
    """Rest of the environment definition omitted."""
    def compute_observations(self):
        self._refresh()
        obs = ["cubeA_quat", "cubeA_pos", "cubeA_to_cubeB_pos", "eef_pos", "eef_quat"]
        obs += ["q_gripper"] if self.control_type == "osc" else ["q"]
        self.obs_buf = torch.cat([self.states[ob] for ob in obs], dim=-1)

        maxs = {ob: torch.max(self.states[ob]).item() for ob in obs}

        # some important observation
        self.cubeA_quat = self.states["cubeA_quat"]
        self.cubeA_pos = self.states["cubeA_pos"]
        self.cubeA_to_cubeB_pos = self.states["cubeA_to_cubeB_pos"]
        self.eef_pos = self.states["eef_pos"]
        self.eef_quat = self.states["eef_quat"]
        if self.control_type == "osc":
            self.q_gripper = self.states["q_gripper"]

        return self.obs_buf