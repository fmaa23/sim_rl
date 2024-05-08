from setuptools import setup, find_packages

setup(
    name='sim_rl',
    version='0.1.0',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'sim_rl = sim_rl.main:main'
        ]
    }
)