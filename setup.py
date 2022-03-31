from setuptools import find_packages
from setuptools import setup

with open("README.md", "r") as file:
    long_description = file.read()

setup(
    # name
    name='chunc',

    # current version
    #   MAJOR VERSION:  1
    #   MINOR VERSION:  0
    #   Maintenance:    0
    version='1.0.0',

    # descriptions
    description='Packages for ML/TDA based analysis of MSSM models.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords='mssm, cmssm, pmssm',

    # my info
    author='Nicholas Carrara',
    author_email='ncarrara.physics@gmail.com',

    # where to find the source
    url='https://github.com/infophysics/CHUNCNet',

    # requirements
    install_reqs = [],

    # packages
    packages=find_packages(
        #where='chunc',
        exclude=['tests'],
    ),
    #package_dir={'': 'chunc'},
    include_package_data=True,

    # classifiers
    classifiers = [
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Experimental Physics',
        'License :: GNU',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
    ],
    python_requires='>3.7',

    # possible entry point
    entry_points={
        'console_scripts': [],
    },
)