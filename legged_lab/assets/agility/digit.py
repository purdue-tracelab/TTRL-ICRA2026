"""Configuration for Unitree robots.

The following configurations are available:

* :obj:`DIGIT_CFG`: Agility robotics Digit_v3

Reference: https://github.com/unitreerobotics/unitree_ros
"""

from legged_lab.assets import ISAAC_ASSET_DIR

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.actuators import ImplicitActuatorCfg, IdealPDActuatorCfg

from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
##
# Configuration
##

DIGIT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/agility/digitv3/digitv3_ver2.usd",
        # activate_contact_sensors=False,
        activate_contact_sensors=True,
        
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
            
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, 
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            fix_root_link=False,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.03077151),
        joint_pos={
            "left_leg_hip_roll": 3.65171317e-01,
            "left_leg_hip_yaw": -6.58221569e-03,
            "left_leg_hip_pitch": 3.16910843e-01,
            "left_leg_knee": 3.57944829e-01,
            "left_leg_tarsus": -0.3311601,
            "left_leg_heel_spring": -0.01160161,
            "left_leg_toe_a": -1.32137105e-01,
            "left_leg_toe_b":  1.24904386e-01,
            "left_leg_toe_pitch":  0.13114588,
            "left_leg_toe_roll":  -0.01159121,
           
            "right_leg_hip_roll": -3.65734576e-01,
            "right_leg_hip_yaw": 6.42881761e-03,
            "right_leg_hip_pitch": -3.16910843e-01,
            "right_leg_knee": -3.58016735e-01,
            "right_leg_tarsus": 0.33119604,
            "right_leg_heel_spring": 0.01160569,
            "right_leg_toe_a": 1.32006717e-01,
            "right_leg_toe_b": -1.25034774e-01,     
            "right_leg_toe_pitch":  -0.13114439,
            "right_leg_toe_roll":  0.01117851,
            
            "left_arm_shoulder_roll": -0.0773893, # -1.50466737e-01,
            "left_arm_shoulder_pitch": 1.14451, #1.09051174e00,
            "left_arm_shoulder_yaw": 0.00121951, #3.43707570e-04,
            "left_arm_elbow": -0.0428987, #-1.39091311e-01,
            
            "right_arm_shoulder_roll": 0.077303, #1.50437975e-01,
            "right_arm_shoulder_pitch": -1.14575, #-1.09045901e00,
            "right_arm_shoulder_yaw": -0.00116583, #-3.51377474e-04,
            "right_arm_elbow": 0.042822, #1.39086517e-01,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.90,
    # NOTE: Set a effort_limit_scale to scale the effort limit of the actuators.
    # effort_limit_sim = effort_limit_sim * effort_limit_scale
    # The DigitActuatorCfg CFG and class file is located in the digitlab/actuators directory.
    # The actuator type is still ImplicitActuatorCfg
    # TODO: Add  effort_limit_scale"
    actuators={
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_leg_hip_roll", 
                ".*_leg_hip_yaw", 
                ".*_leg_hip_pitch", 
                ".*_leg_knee",
                ],
            effort_limit_sim={
                ".*_leg_hip_roll": 126,
                ".*_leg_hip_yaw": 79,
                ".*_leg_hip_pitch": 216,
                ".*_leg_knee": 231,
            },
            velocity_limit_sim={
                ".*_leg_hip_roll": 4.58149,
                ".*_leg_hip_yaw": 7.33038,
                ".*_leg_hip_pitch": 8.50848,
                ".*_leg_knee": 8.50848,
            },
            stiffness={
                ".*_leg_hip_roll": 100.0,
                ".*_leg_hip_yaw": 100.0,
                ".*_leg_hip_pitch": 200.0,
                ".*_leg_knee": 200.0,
                },
            damping={
                ".*_leg_hip_roll": 5.0 + 66.849,
                ".*_leg_hip_yaw": 5.0 + 26.1129,
                ".*_leg_hip_pitch": 5.0 + 38.05,
                ".*_leg_knee": 5.0 + 38.05,
                },
        ),

        "toes": ImplicitActuatorCfg(
            joint_names_expr=[ ".*_leg_toe_a", ".*_leg_toe_b"],
            effort_limit_sim={
                ".*_leg_toe_a": 41,
                ".*_leg_toe_b": 41,
            },
            velocity_limit_sim={
                ".*_leg_toe_a": 11.5192,
                ".*_leg_toe_b": 11.5192,
            },
            stiffness={
                ".*_leg_toe_a": 20.0,
                ".*_leg_toe_b": 20.0
                },
            damping={
                ".*_leg_toe_a": 1.0 + 15.5532,
                ".*_leg_toe_b": 1.0 + 15.5532,
                },
        ),

        "arms": ImplicitActuatorCfg(
            joint_names_expr=[".*_arm_shoulder_roll",  
                              ".*_arm_shoulder_pitch", 
                              ".*_arm_shoulder_yaw", 
                              ".*_arm_elbow"],
            effort_limit_sim={
                ".*_arm_shoulder_roll": 126,
                ".*_arm_shoulder_pitch": 126,
                ".*_arm_shoulder_yaw": 79,
                ".*_arm_elbow": 126,
            },
            velocity_limit_sim={
                ".*_arm_shoulder_roll": 4.58149,
                ".*_arm_shoulder_pitch": 4.58149,
                ".*_arm_shoulder_yaw": 7.33038,
                ".*_arm_elbow": 4.58149, 
            },
            stiffness={
                ".*_arm_shoulder_roll": 150.0,
                ".*_arm_shoulder_pitch": 150.0,
                ".*_arm_shoulder_yaw": 100.0,
                ".*_arm_elbow": 100.0,
            },
            damping={
                ".*_arm_shoulder_roll": 5.0 + 66.849,
                ".*_arm_shoulder_pitch": 5.0 + 66.849,
                ".*_arm_shoulder_yaw": 5.0 + 26.1129,
                ".*_arm_elbow": 5.0 + 66.849,
            },
        ),

        "passive": ImplicitActuatorCfg(
            joint_names_expr=[".*_leg_shin",
                              ".*_leg_tarsus", 
                              ".*_leg_heel_spring", 
                              ".*_leg_toe_pitch", 
                              ".*_leg_toe_roll"],
            stiffness={
                ".*_leg_shin": 6000.0,
                ".*_leg_tarsus": 0.0,
                ".*_leg_heel_spring": 4375.0,
                ".*_leg_toe_pitch": 0.0,
                ".*_leg_toe_roll": 0.0,
            },

            damping={
                ".*_leg_shin": 0.0,
                ".*_leg_tarsus": 0.0,
                ".*_leg_heel_spring": 0.0,
                ".*_leg_toe_pitch": 0.0,
                ".*_leg_toe_roll": 0.0,
            },
        )
    },
)