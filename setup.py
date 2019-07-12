# Lib
from setuptools import setup, find_packages

setup(
    name='methQC',
    version='0.2.0',
    description='Quality Control (QC), Visualization/plotting, and postprocessing software for Illumina methylation array data.',
    long_description='This python package offers tools for assess sample quality (The QC part), plotting and visualization, and postprocessing/transformation of Illumina methylation array data.  For use with methpype library.',
    url='https://github.com/LifeEGX/methQC',
    license='MIT',
    author='Life Epigenetics',
    author_email='info@lifeegx.com',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy',
        'pandas',
        'seaborn',
        'matplotlib',
        'tqdm',
        'scikit-learn',      
    ],
    extras_require={
        'dev': [
            'pytest',
            'pytest-pep8',
            'pytest-cov',
            'flake8',
            'pytest',
            'coverage',
            'coveralls-python',
            'sphinxcontrib-apidoc',
            'm2r',
            'nbsphinx',
            'sphinx',
            'ipykernel'
        ]
    },
    setup_requires=['pytest-runner'],
    tests_require=['pytest']
    entry_points={
        'console_scripts': [
            'methQC-cli = methQC.cli:cli_parser'
            ]
    }
)
