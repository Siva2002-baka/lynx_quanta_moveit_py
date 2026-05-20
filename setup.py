from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'lynx_quanta_moveit_py'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Sai Kumar Alamkaram',
    maintainer_email='saikumar14970@gmail.com',
    description='Python MoveIt 2 interface for Lynx M20 with Piper arm',
    license='BSD',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'moveit_pose_controller = lynx_quanta_moveit_py.moveit_pose_controller:main',
        ],
    },
)