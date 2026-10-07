"""Publish only the static TFs required by Vive-to-Tianji control.

OpenVR input and the Tianji controller are intentionally not launched here so
this file can be added to an already-running, manually supervised session.
"""

from launch import LaunchDescription

from wuji_teleop_bringup.tf_utils import (
    create_chest_tf_nodes,
    create_tianji_tf_nodes,
)


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        *create_chest_tf_nodes(),
        *create_tianji_tf_nodes(),
    ])
