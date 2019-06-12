# Lib
from setuptools import setup, find_packages

setup(
    name='methQC',
    version='0.1.0',
    description='Quality Control (QC) for Illumina methylation array data.',
    long_description='Quality Control (QC) for Illumina methylation array preprocessing software. For use with methpype library.',
    url='https://github.com/LifeEGX/methQC',
    license='MIT',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy',
        'pandas',
        'seaborn',
        'matplotlib'
    ],
    setup_requires=['pytest-runner'],
    tests_require=['pytest']
)
