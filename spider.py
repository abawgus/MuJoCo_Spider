import numpy as np

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from scipy.spatial.transform import Rotation as R
import numpy as np

DEFAULT_CAMERA_CONFIG = {
    "distance": 12.0,
    "azimuth" : -45
    
}


class SpiderEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 20,
    }

    def __init__(
        self,
        xml_file=r"spider.xml",
        ctrl_cost_weight=0.5,
        use_contact_forces=True,
        contact_cost_weight=5e-4,
        healthy_reward=1.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(0.0, 2.3),
        contact_force_range=(-1.0, 1.0),
        reset_noise_scale=0.1,
        exclude_current_positions_from_observation=True,
        **kwargs
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            ctrl_cost_weight,
            use_contact_forces,
            contact_cost_weight,
            healthy_reward,
            terminate_when_unhealthy,
            healthy_z_range,
            contact_force_range,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            **kwargs
        )

        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight

        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range

        self._contact_force_range = contact_force_range

        self._reset_noise_scale = reset_noise_scale

        self._use_contact_forces = use_contact_forces

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        obs_shape = 271
        # if not exclude_current_positions_from_observation:
        #     obs_shape += 2
        # if use_contact_forces:
        #     obs_shape += 84

        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float64
        )

        MujocoEnv.__init__(
            self, xml_file, 5, observation_space=observation_space, 
            # observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs
        )

    @property
    def healthy_reward(self):
        return (
            float(self.is_healthy or self._terminate_when_unhealthy)
            * self._healthy_reward
        )

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    @property
    def contact_forces(self):
        raw_contact_forces = self.data.cfrc_ext
        min_value, max_value = self._contact_force_range
        contact_forces = np.clip(raw_contact_forces, min_value, max_value)
        return contact_forces

    @property
    def contact_cost(self):
        contact_cost = self._contact_cost_weight * np.sum(
            np.square(self.contact_forces)
        )
        return contact_cost

    @property
    def is_healthy(self):
        # state = self.state_vector()
        min_z, max_z = self._healthy_z_range
        x,y,z = self.get_body_com("torso")[:]
        # is_healthy = np.isfinite(state).all() and min_z <= z <= max_z
        is_healthy = min_z <= z <= max_z
        # is_healthy = True
        #flip check
        a,b,c,d = self.data.body("torso").xquat
        r = R.from_quat([a,b,c,d])
        z,y,x = r.as_euler('zyx', degrees=True)
        if z > 130 or z < -130:
            is_healthy = False
        
        return is_healthy

    @property
    def terminated(self):
        terminated = not self.is_healthy if self._terminate_when_unhealthy else False
        return terminated

    def step(self, action):
        xy_position_before = self.get_body_com("torso")[:2].copy()
        velo_before = self.get_body_velo("torso")[:2].copy()
                
        self.do_simulation(action, self.frame_skip)
        
        xy_position_after = self.get_body_com("torso")[:2].copy()
        velo_after = self.get_body_velo("torso")[:2].copy()
        
        
        z_pos = self.get_body_com("torso")[2].copy()
        accel_after, gyro_after = self.get_sensor_data()
        
        gyro_x, gyro_y, gyro_z = gyro_after
        gyro_vec = np.sqrt(gyro_x**2 + gyro_x**2 + gyro_x**2)
        # print(gyro_vec)

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity
        
        accel = (velo_after - velo_before) / self.dt
        x_accel, y_accel = accel

        a,b,c,d = self.data.body("torso").xquat
        r = R.from_quat([a,b,c,d])
        z,y,x = r.as_euler('zyx', degrees=True)
        
        forward_reward = x_velocity
        healthy_reward = self.healthy_reward
        
        rewards = forward_reward + healthy_reward

        costs = ctrl_cost = self.control_cost(action)

        terminated = self.terminated
        observation = self._get_obs()
        info = {
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_survive": healthy_reward,
            "x_position": xy_position_after[0],
            "y_position": xy_position_after[1],
            "z_position" : z_pos,
            "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            "gyro_x" : gyro_x,
            "gyro_y" : gyro_y,
            "gyro_z" : gyro_z,
            "x_acceleration": x_accel,
            "y_acceleration": y_accel,
            "x" : x,
            "y" : y,
            "z" : z,
            "forward_reward": forward_reward,
        }
        if self._use_contact_forces:
            contact_cost = self.contact_cost
            costs += contact_cost
            costs -= z/180
            info["reward_ctrl"] = -contact_cost
        
        reward = rewards - costs

        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, False, info

    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()
        accel, gyro = self.get_sensor_data()
        quat = self.data.body("torso").xquat.flat.copy()

        com = self.data.body("torso").xpos
        
        a,b,c,d = self.data.body("torso").xquat
        r = R.from_quat([a,b,c,d])
        angle = r.as_euler('zyx', degrees=True)

        # if self._exclude_current_positions_from_observation:
        #     position = position[2:]

        if self._use_contact_forces:
            contact_force = self.contact_forces.flat.copy()
            return np.concatenate((position, velocity, contact_force))#, accel, gyro))
        else:
            return np.concatenate((position, velocity))#, accel, gyro))

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = (
            self.init_qvel
            + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv)
        )
        self.set_state(qpos, qvel)

        observation = self._get_obs()

        return observation

    def viewer_setup(self):
        assert self.viewer is not None
        
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

                
