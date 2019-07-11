from setuptools import setup, find_packages

setup(
    name='methQC',
    version='0.2.0',
    description='Quality Control, Visualization/plotting, and postprocessing software for Illumina methylation array data.',
    long_description='This python package offers tools for assess sample quality (The QC part), plotting and visualization, and postprocessing/transformation of Illumina methylation array data.',
    url='https://github.com/LifeEGX/methQC',
    license='MIT',
    author='Life Epigenetics',
    author_email='info@lifeegx.com',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy',
        'pandas',
        'tqdm',
        'scikit-learn',
        'seaborn',
        'matplotlib',
    ],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'methQC-cli = methQC.cli:cli_parser'
            ]
    },
)
