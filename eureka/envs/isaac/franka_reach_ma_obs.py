class FrankaReachMA(VecTask):
    """Rest of the environment definition omitted."""
    def compute_observations(self):
        # cubeA_quat: 4
        # cubeA_pos: 3
        # eef_quat: 4
        # eef_pos: 3
        # base_pos: 3
        # base_quat: 4
        # cubeA_pos_min_relative: 3

        self._refresh()
        obs_all_targets = self.states["cubeA_pos"].reshape(self.num_envs, -1).repeat_interleave(self.num_agents, dim=0)  # shape: ((self.num_envs*self.num_agents) x (self.num_targets*3))
        
        obs = ["eef_quat", "eef_pos", "cubeA_pos_min_relative"]
        obs_self = torch.cat([self.states[ob].contiguous().view(self.num_envs * self.num_agents, -1) for ob in obs], dim=-1)
        
        _unshifted = self.states["eef_pos"].reshape(self.num_envs, -1)
        shifted = []
        for i in range(self.num_agents):
            shifted.append( torch.cat((_unshifted[..., i*3:], _unshifted[..., :i*3]), dim=1) )
        others_eef_pos = torch.stack(shifted, dim=1)[..., 3:].view(self.num_envs * self.num_agents, -1)

        self.obs_buf = torch.cat([obs_all_targets, obs_self, others_eef_pos], dim=-1)

        # here provide some properties for computing reward
        self.eef_pos                = self.states["eef_pos"]
        self.cubeA_pos_min_relative = self.states["cubeA_pos_min_relative"]
        # self._hands_contact_forces

        return self.obs_buf