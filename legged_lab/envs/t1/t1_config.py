# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# Original code is licensed under BSD-3-Clause.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
# Modifications are licensed under BSD-3-Clause.
#
# This file contains code derived from Isaac Lab Project (BSD-3-Clause license)
# with modifications by Legged Lab Project (BSD-3-Clause license).

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.utils import configclass

import legged_lab.mdp as mdp
from legged_lab.assets.booster import BOOSTER_T1_CFG
from legged_lab.envs.base.base_env_config import (  # noqa:F401
    BaseAgentCfg,
    BaseEnvCfg,
    BaseSceneCfg,
    DomainRandCfg,
    HeightScannerCfg,
    PhysxCfg,
    RewardCfg,
    RobotCfg,
    SimCfg,
)
from legged_lab.terrains import GRAVEL_TERRAINS_CFG, ROUGH_TERRAINS_CFG


@configclass
class T1RewardCfg(RewardCfg):
    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=1.0, params={"std": 0.5})
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=1.0, params={"std": 0.5})
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    energy = RewTerm(func=mdp.energy, weight=-1e-3)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1.25e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names="(?!.*_foot_link*).*"), "threshold": 1.0},
    )
    fly = RewTerm(
        func=mdp.fly,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_foot_link*"), "threshold": 1.0},
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.5,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_foot_link*"), "threshold": 0.4},
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_foot_link*"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot_link*"),
        },
    )
    feet_force = RewTerm(
        func=mdp.body_force,
        weight=-3e-3,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_foot_link*"),
            "threshold": 500,
            "max_reward": 400,
        },
    )
    feet_too_near = RewTerm(
        func=mdp.feet_too_near_humanoid,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*_foot_link*"]), "threshold": 0.3},
    )
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*_foot_link*"])},
    )
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.0)
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Hip_Yaw", ".*_Hip_Roll"])},
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_Shoulder_.*", ".*_Elbow_.*"]
            )
        },
    )
    joint_deviation_should_roll = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_Shoulder_Roll"]
            )
        },
    )
    # joint_deviation_legs = RewTerm(
    #     func=mdp.joint_deviation_l1,
    #     weight=-0.05,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_pitch.*", ".*_knee.*", ".*_foot_link*"])},
    # )
    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=["Waist"]
            )
        },
    )


@configclass
class T1FlatEnvCfg(BaseEnvCfg):

    reward = T1RewardCfg()

    def __post_init__(self):
        super().__post_init__()
        #For fast validation, we can use a faster simulation time step
        self.sim.dt = 0.005
        self.sim.decimation = 4 # 50 Hz
        #NOTE: The following parameters are set to match the original T1 configuration.
        # For sim2real, recommand to use the following settings:
        #######################################################
        # self.sim.dt = 0.002
        # self.sim.decimation = 10 # 50 Hz
        #######################################################
        self.scene.height_scanner.prim_body_name = "Trunk"
        self.scene.robot = BOOSTER_T1_CFG
        self.scene.terrain_type = "generator"
        self.scene.terrain_generator = GRAVEL_TERRAINS_CFG
        self.robot.terminate_contacts_body_names = ["Trunk"]
        self.robot.feet_body_names = [".*_foot_link"]
        self.domain_rand.events.add_base_mass.params["asset_cfg"].body_names = ["Trunk"]


@configclass
class T1FlatAgentCfg(BaseAgentCfg):
    experiment_name: str = "t1_flat"
    logger = "tensorboard"
    save_interval = 100
    max_iterations = 6001


@configclass
class T1RoughEnvCfg(T1FlatEnvCfg):

    def __post_init__(self):
        super().__post_init__()
        self.scene.height_scanner.enable_height_scan = True
        self.scene.terrain_generator = ROUGH_TERRAINS_CFG
        self.robot.actor_obs_history_length = 1
        self.robot.critic_obs_history_length = 1
        self.reward.track_lin_vel_xy_exp.weight = 1.5
        self.reward.track_ang_vel_z_exp.weight = 1.5
        self.reward.lin_vel_z_l2.weight = -0.25


@configclass
class T1RoughAgentCfg(BaseAgentCfg):
    experiment_name: str = "t1_rough"
    logger = "tensorboard"
    save_interval = 100
    max_iterations = 6001
    def __post_init__(self):
        super().__post_init__()
        self.policy.class_name = "ActorCriticRecurrent"
        self.policy.actor_hidden_dims = [256, 256, 128]
        self.policy.critic_hidden_dims = [256, 256, 128]
        self.policy.rnn_hidden_size = 256
        self.policy.rnn_num_layers = 1
        self.policy.rnn_type = "lstm"
