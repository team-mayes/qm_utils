#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'six',
    'seaborn',
    'numpy', 'pandas', 'pandas', 'xlsxwriter',
]

test_requirements = [
    'seaborn',
    'six',
    'numpy',
]

setup(
    name='qm_utils',
    version='0.1.0',
    description="tools  for QM projects in Team Mayes & Blue",
    long_description=readme + '\n\n' + history,
    author="Team Mayes and Blue",
    author_email='hbmayes@umich.edu',
    url='https://github.com/hmayes/qm_utils',
    packages=[
        'qm_utils',
    ],
    package_dir={'qm_utils':
                 'qm_utils'},
    entry_points={
        'console_scripts': ['cp_params=qm_utils.cp_params:main',
                            'coord_to_com=qm_utils.coord_to_com:main',
                            'xyz_cluster=qm_utils.xyz_cluster:main',
                            'norm_analysis=qm_utils.norm_analysis:main'
                            ]
    },
    include_package_data=True,
    package_data={'qm_utils': ['cfg/*.*', ], },
    install_requires=requirements,
    license="BSD license",
    zip_safe=False,
    keywords='qm_utils',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    test_suite='tests',
    tests_require=test_requirements
)
