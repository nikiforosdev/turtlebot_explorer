from setuptools import setup
import os
from glob import glob

package_name = 'turtlebot_explorer'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email@example.com',
    description='TurtleBot exploration package',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'explorer_controller = turtlebot_explorer.explorer_controller:main',
            'frontier_detector = turtlebot_explorer.frontier_detector:main',
            'path_planner = turtlebot_explorer.path_planner:main',
            'reactive_controller = turtlebot_explorer.reactive_controller:main',
        ],
    },
)