from setuptools import find_packages, setup

setup(
    name='motor_model',
    packages=find_packages(),
    version='0.1.0',
    description='',
    author='Joanna Chang',
    #license='License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
    install_requires = [
        'numpy',
        'config_manager',
        'scikit-learn'
    ]
)
