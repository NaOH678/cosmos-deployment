from glob import glob
import os

from setuptools import find_packages, setup


package_name = "wuji_data_pipeline"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        (
            "share/" + package_name,
            ["package.xml", "README.md", "CLOUD_MODEL_INTEGRATION_GUIDE.md"],
        ),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
    ],
    install_requires=[
        "setuptools",
        "numpy>=1.24.0",
        "scipy>=1.8.0",
        "pyyaml>=6.0",
    ],
    zip_safe=True,
    maintainer="Wuji Robotics",
    maintainer_email="dev@wuji.com",
    description="ROS 2 LMDB recording, replay, and deployment pipeline",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "recorder_node = wuji_data_pipeline.recorder_node:main",
            "record_session = wuji_data_pipeline.record_session:main",
            "arm_teleop_session = wuji_data_pipeline.arm_teleop_session:main",
            "hand_teleop_session = wuji_data_pipeline.hand_teleop_session:main",
            "replay_session = wuji_data_pipeline.replay_session:main",
            "deployment_session = wuji_data_pipeline.deployment_session:main",
            "deployment_node = wuji_data_pipeline.deployment_node:main",
            "deployment_state_trace = wuji_data_pipeline.deployment_state_trace:main",
            "cloud_policy_server = wuji_data_pipeline.cloud_policy_server:main",
            "replay_server = wuji_data_pipeline.replay_server:main",
            "inspect_episode = wuji_data_pipeline.inspect_episode:main",
        ],
    },
)
