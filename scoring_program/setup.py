from setuptools import setup,find_packages

setup(
    name='mipi_hybirdevs_demosaic_scoring',
    version='1.0.0',
    description='A scoring program for Demosaic for Hybridevs Camera in MIPI challenge',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'skimage',
    ],
)