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
from legged_lab.assets.agility import DIGIT_CFG
from legged_lab.envs.base.legged_env_config import(  # noqa:F401
    LeggedAgentCfg,
    LeggedEnvCfg,
    BaseSceneCfg,
    DomainRandCfg,
    HeightScannerCfg,
    PhysxCfg,
    RewardCfg,
    RobotCfg,
    SimCfg,
    ActionsCfg,
    ObservationsCfg,
)
from legged_lab.envs.digit.env_config import DigitRewardCfg, DigitPhaseRewardCfg

from legged_lab.terrains import GRAVEL_TERRAINS_CFG, ROUGH_TERRAINS_CFG


@configclass
class DigitFlatEnvCfg(LeggedEnvCfg):

    reward = DigitRewardCfg()

    def __post_init__(self):
        super().__post_init__()
        # self.sim.dt = 0.001
        # self.sim.decimation = 20
        self.sim.dt = 0.005
        self.sim.decimation = 4
        self.scene.height_scanner.prim_body_name = "torso_base"
        self.scene.robot = DIGIT_CFG
        self.scene.terrain_type = "generator"
        self.scene.terrain_generator = GRAVEL_TERRAINS_CFG
        self.robot.terminate_contacts_body_names = ["torso_base", ".*hip.*", ".*knee", ".*elbow"]
        self.robot.feet_body_names = [".*_leg_toe_roll"]
        self.robot.min_base_height = 0.4
        self.robot.max_base_height = 1.05
        self.robot.phase_dt = 0.64
        self.domain_rand.events.add_base_mass.params["asset_cfg"].body_names = ["torso_base"]
        self.observations.joint_names = [
            "left_leg_hip_roll",
            "left_leg_hip_yaw",
            "left_leg_hip_pitch",
            "left_leg_knee",
            "left_leg_shin", 
            "left_leg_tarsus",
            "left_leg_heel_spring",
            "left_leg_toe_a",
            "left_leg_toe_b",
            "left_leg_toe_pitch", 
            "left_leg_toe_roll",
            "left_arm_shoulder_roll",
            "left_arm_shoulder_pitch",
            "left_arm_shoulder_yaw",
            "left_arm_elbow",
            "right_leg_hip_roll",
            "right_leg_hip_yaw",
            "right_leg_hip_pitch",
            "right_leg_knee",
            "right_leg_shin", 
            "right_leg_tarsus",
            "right_leg_heel_spring",
            "right_leg_toe_a",
            "right_leg_toe_b",
            "right_leg_toe_pitch",
            "right_leg_toe_roll",
            "right_arm_shoulder_roll",
            "right_arm_shoulder_pitch",
            "right_arm_shoulder_yaw",
            "right_arm_elbow"
        ]
        self.actions.joint_names = [
            "left_leg_hip_roll", 
            "left_leg_hip_yaw", 
            "left_leg_hip_pitch", 
            "left_leg_knee",
            "left_leg_toe_a", 
            "left_leg_toe_b",
            "left_arm_shoulder_roll",
            "left_arm_shoulder_pitch",  
            "left_arm_shoulder_yaw", 
            "left_arm_elbow",
            "right_leg_hip_roll", 
            "right_leg_hip_yaw", 
            "right_leg_hip_pitch", 
            "right_leg_knee",
            "right_leg_toe_a", 
            "right_leg_toe_b",
            "right_arm_shoulder_roll",
            "right_arm_shoulder_pitch",  
            "right_arm_shoulder_yaw", 
            "right_arm_elbow",
        ]

@configclass
class DigitFlatEnvCfg_Play(DigitFlatEnvCfg):

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        
        self.domain_rand.events.push_robot = None

        self.commands.ranges.lin_vel_x = (0.5, 0.5)
        self.commands.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.ranges.ang_vel_z = (-0.0, 0.0)

@configclass
class DigitFlatAgentCfg(LeggedAgentCfg):
    experiment_name: str = "Digit_flat"
    wandb_project: str = "Digit_flat"
    max_iterations = 8001
    save_interval = 400


@configclass
class DigitRoughEnvCfg(DigitFlatEnvCfg):

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
class DigitRoughAgentCfg(LeggedAgentCfg):
    experiment_name: str = "Digit_rough"
    wandb_project: str = "Digit_rough"

    def __post_init__(self):
        super().__post_init__()
        self.policy.class_name = "ActorCriticRecurrent"
        self.policy.actor_hidden_dims = [256, 256, 128]
        self.policy.critic_hidden_dims = [256, 256, 128]
        self.policy.rnn_hidden_size = 256
        self.policy.rnn_num_layers = 1
        self.policy.rnn_type = "lstm"


###################Env Phase Configs###################
@configclass
class DigitFlatEnvPhaseCfg(DigitFlatEnvCfg):
    
    reward = DigitPhaseRewardCfg()
    
    def __post_init__(self):
        super().__post_init__()
        # self.sim.dt = 0.001
        # self.sim.decimation = 20
        self.sim.dt = 0.005
        self.sim.decimation = 4