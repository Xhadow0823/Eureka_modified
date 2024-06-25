class FrankaLift2(VecTask):
    """Rest of the environment definition omitted."""
    def compute_observations(self):
        self._refresh()
        obs_names = []
        obs_names += ["eef_quat", "eef_pos", "cubeA_pos_relative", "cubeA_height"]
        obs_names += ["q_gripper"]
        obs_names += ["FSM_p"]
        self.obs_buf = torch.cat([self.states[ob].view(self.num_envs, -1) for ob in obs_names], dim=-1)

        # some important observation
        self.cubeA_pos = self.states["cubeA_pos"]
        self.cubeA_quat = self.states["cubeA_quat"]

        self.eef_pos = self.states["eef_pos"]
        self.eef_quat = self.states["eef_quat"]
        self.cubeA_pos_relative = self.states["cubeA_pos_relative"]  # this is the pos of cubeA minus the pos of eef
        self.cubeA_height = self.states["cubeA_height"]  # IMPORTANT: this the height of cubeA (height from table surface)
        self.a_gripper = self.actions[:, -1]     # IMPORTANT: this is the action value of gripper, a_gripper <= 0 means the gripper is closing

        self.arm_base_pos = torch.tensor([-0.45, 0.0, 1.125], dtype=torch.float, device=self.device)  # this is the base position of Franka arm

        self.FSM = self.states["FSM"]  # this is a torch.tensor dtype=long for store the current state of each agent
        
        return self.obs_buf