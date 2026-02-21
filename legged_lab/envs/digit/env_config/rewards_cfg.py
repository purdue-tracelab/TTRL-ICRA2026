from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.utils import configclass
import legged_lab.mdp as mdp

from legged_lab.envs.base.legged_env_config import RewardCfg


@configclass
class DigitRewardCfg(RewardCfg):
    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=2.0, params={"std": 0.5})
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=2.0, params={"std": 0.5})
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    energy = RewTerm(
        func=mdp.energy, 
        weight=-1e-3,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", 
                joint_names=[
                ".*_hip_.*", 
                ".*_leg_knee", 
                ".*_leg_toe_a",
                ".*_leg_toe_b",
                ".*_arm_shoulder_.*",
                ".*_arm_elbow",
                ],
            )
        }
    )
    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2, 
        weight=-1.25e-7,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", 
                joint_names=[
                ".*_hip_.*", 
                ".*_leg_knee", 
                ".*_leg_toe_a",
                ".*_leg_toe_b",
                ".*_arm_shoulder_.*",
                ".*_arm_elbow",
                ],
            )
        }
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    # undesired_contacts = RewTerm(
    #     func=mdp.undesired_contacts,
    #     weight=-1.0,
    #     params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names="(?!.*_leg_toe_roll).*"), "threshold": 1.0},
    # )
    fly = RewTerm(
        func=mdp.fly,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_leg_toe_roll"), "threshold": 1.0},
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.5,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_leg_toe_roll"), "threshold": 0.4},
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_leg_toe_roll"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_leg_toe_roll"),
        },
    )
    feet_force = RewTerm(
        func=mdp.body_force,
        weight=-3e-3,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_leg_toe_roll"),
            "threshold": 500,
            "max_reward": 400,
        },
    )
    feet_too_near = RewTerm(
        func=mdp.feet_too_near_humanoid,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*_leg_toe_roll"]), "threshold": 0.3},
    )
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*_leg_toe_roll"])},
    )
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.0)
    joint_deviation_hip_ry = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", 
                joint_names=[
                    ".*_leg_hip_roll", 
                    ".*_leg_hip_yaw"
                ],
            )
        },
    )
    joint_deviation_arm_ry = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", 
                joint_names=[
                    ".*_arm_shoulder_roll",
                    ".*_arm_shoulder_yaw",
                ],
            )
        },
    )

    joint_deviation_arm_hip_p = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", 
                joint_names=[
                    ".*_arm_shoulder_pitch",
                    ".*_hip_pitch",
                ],
            )
        },
    )

    joint_deviation_arm_p = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.3,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", 
                joint_names=[
                    ".*_arm_shoulder_pitch",
                ],
            )
        },
    )

    joint_deviation_hip_p = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", 
                joint_names=[
                    ".*_hip_pitch",
                ],
            )
        },
    )

    joint_deviation_elbow = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", 
                joint_names=[
                    ".*_arm_elbow",   
                ],
            )
        },
    )

    joint_deviation_toes = RewTerm(
        func=mdp.joint_deviation_l1,  # type: ignore
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_leg_toe_a",
                    ".*_leg_toe_b",
                    ".*_leg_toe_pitch",
                    ".*_leg_toe_roll",
                ],
            )
        },
    )


@configclass
class DigitPhaseRewardCfg(RewardCfg):
    """
    Configuration for the Digit environment.
    This class defines the rewards and penalties for the Digit learning environment with gait phase rewards.
    """
    # tracking rewards
    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=2.0, params={"std": 0.5})
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=2.0, params={"std": 0.5})
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)

    # termination
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    # dof rewards
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2, 
        weight=-1.0e-5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", 
                joint_names=[
                ".*_hip_.*", 
                ".*_leg_knee", 
                ".*_leg_toe_a",
                ".*_leg_toe_b",
                ".*_arm_shoulder_.*",
                ".*_arm_elbow",
                ],
            )
        }
    )
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.0)
    dof_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-5e-4)
    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2, 
        weight=-1.25e-7,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", 
                joint_names=[
                ".*_hip_.*", 
                ".*_leg_knee", 
                ".*_leg_toe_a",
                ".*_leg_toe_b",
                ".*_arm_shoulder_.*",
                ".*_arm_elbow",
                ],
            )
        }
    )

    #base
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    # joint deviation
    joint_deviation_hip_ry = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", 
                joint_names=[
                    ".*_leg_hip_roll", 
                    ".*_leg_hip_yaw"
                ],
            )
        },
    )
    joint_deviation_arm_ry = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", 
                joint_names=[
                    ".*_arm_shoulder_roll",
                    ".*_arm_shoulder_yaw",
                ],
            )
        },
    )

    joint_deviation_arm_hip_p = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", 
                joint_names=[
                    ".*_arm_shoulder_pitch",
                    ".*_hip_pitch",
                ],
            )
        },
    )

    joint_deviation_arm_p = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.3,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", 
                joint_names=[
                    ".*_arm_shoulder_pitch",
                ],
            )
        },
    )

    joint_deviation_hip_p = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", 
                joint_names=[
                    ".*_hip_pitch",
                ],
            )
        },
    )

    joint_deviation_elbow = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", 
                joint_names=[
                    ".*_arm_elbow",   
                ],
            )
        },
    )

    joint_deviation_toes = RewTerm(
        func=mdp.joint_deviation_l1,  # type: ignore
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_leg_toe_a",
                    ".*_leg_toe_b",
                    ".*_leg_toe_pitch",
                    ".*_leg_toe_roll",
                ],
            )
        },
    )
    # feet rewards 
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_leg_toe_roll"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_leg_toe_roll"),
        },
    )
    
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.25,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor", 
                body_names= ["left_leg_toe_roll", "right_leg_toe_roll"],
            ),
            "threshold": 0.34,
        },
    )
    
    feet_force = RewTerm(
        func=mdp.body_force,
        weight=-3e-3,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_leg_toe_roll"),
            "threshold": 500,
            "max_reward": 400,
        },
    )

    foot_contact = RewTerm(
        func=mdp.reward_feet_contact_number,
        weight=2.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor",
                body_names=["left_leg_toe_roll", "right_leg_toe_roll"],
                preserve_order=True,
            ),
            "pos_rw": 1.0,
            "neg_rw": -0.3,
        },
    )

    track_foot_height = RewTerm(
        func=mdp.track_foot_height,
        weight=0.5,
        params={
            "std": 0.5,
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["left_leg_toe_roll", "right_leg_toe_roll"],
                preserve_order=True,
            ),
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor",
                body_names=["left_leg_toe_roll", "right_leg_toe_roll"],
                preserve_order=True,
            ),
        },
    )

    foot_clearance = RewTerm(
        func=mdp.foot_clearance_reward,
        weight=0.5,
        params={
            "target_height": 0.25,
            "std": 0.5,
            "tanh_mult": 2.0,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_leg_toe_roll"),
        },
    )
