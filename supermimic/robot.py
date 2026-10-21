"""Unitree G1 (29 DoF) as an mjlab entity, following BeyondMimic's actuator model.

mjlab's G1 MJCF is g1_29dof_rev_1_0, the model the LAFAN1 clips were retargeted to. Actuators are
mjlab's BeyondMimic set (10 Hz natural frequency, damping ratio 2 on the reflected motor armature,
action scale 0.25 * effort_limit / stiffness). Self-collisions are disabled, as in the Isaac Lab
reproduction; every robot collision geom gets condim 3 and the highest priority, so its friction
(the per-env sim context) is the one used against the ground.
"""

from __future__ import annotations

import re

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import (
    G1_ACTION_SCALE,
    G1_ARTICULATION,
    G1_XML,
    KNEES_BENT_KEYFRAME,
    get_spec,
)
from mjlab.entity import EntityCfg
from mjlab.utils.spec_config import CollisionCfg

MJCF_PATH = str(G1_XML)

# Joint order of the LAFAN1 retarget CSVs (Unitree SDK order).
G1_JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]  # fmt: skip

# BeyondMimic's tracked bodies; the anchor/root is the pelvis.
TRACKED_BODIES = [
    "pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
    "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link", "torso_link",
    "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
    "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link",
]  # fmt: skip

DEFAULT_JOINT_POS = dict(KNEES_BENT_KEYFRAME.joint_pos)

COLLISION = CollisionCfg(
    geom_names_expr=(".*_collision",),
    contype=0,
    conaffinity=1,
    condim=3,
    priority=1,
    friction=(0.6,),
)


def _lookup(table: dict[str, float], name: str, default: float) -> float:
    val = default
    for pat, v in table.items():
        if re.fullmatch(pat, name):
            val = v
    return val


def action_scale(joint_name: str) -> float:
    return _lookup(G1_ACTION_SCALE, joint_name, float("nan"))


def default_joint_pos_list() -> list[float]:
    return [_lookup(DEFAULT_JOINT_POS, n, 0.0) for n in G1_JOINT_NAMES]


def g1_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=KNEES_BENT_KEYFRAME,
        collisions=(COLLISION,),
        spec_fn=get_spec,
        articulation=G1_ARTICULATION,
    )
