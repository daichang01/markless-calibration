from setuptools import find_packages, setup

package_name = 'point_cloud_dev'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='daichang',
    maintainer_email='18181985920@163.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'oripointcloud_sub = point_cloud_dev.oripointcloud_sub:main',
            'segpointcloud_sub = point_cloud_dev.segpointcloud_sub:main',
            'depthRgbsub_pcpub = point_cloud_dev.depthRgbsub_pcpub:main',
            'scanpcd_pub = point_cloud_dev.scanpcd_pub:main',
            'coarse_fine_registration = point_cloud_dev.coarse_fine_registration:main'
        ],
    },
)
