class FrankaReach(VecTask):
    """Rest of the environment definition omitted."""
    def compute_observations(self):
        # -> new obs spec:
        #   cubeA_quat: 4
        #   cubeA_pos: 3
        #   eef_quat: 4
        #   eef_pos: 3
        # total size: 14

        self._refresh()
        obs = ["cubeA_quat", "cubeA_pos", "eef_quat", "eef_pos"]
        self.obs_buf = torch.cat([self.states[ob] for ob in obs], dim=-1)

        # some properties for computing reward
        self.cubeA_quat = self.states["cubeA_quat"]
        self.cubeA_pos  = self.states["cubeA_pos"]
        self.eef_quat   = self.states["eef_quat"]
        self.eef_pos    = self.states["eef_pos"]

        return self.obs_buf