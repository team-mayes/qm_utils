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
    'pandas>=0.19.1',
    'numpy>=1.11.2',
]

test_requirements = [
    'six',
    'seaborn',
    'pandas>=0.19.1',
    'numpy>=1.11.2',
]

setup(
    name='qm_utils',
    version='0.1.0',
    description="tools  for QM projects in Team Mayes & Blue",
    long_description=readme + '\n\n' + history,
    author="Heather Beth Mayes",
    author_email='hbmayes@umich.edu',
    url='https://github.com/hmayes/qm_utils',
    packages=[
        'qm_utils',
    ],
    package_dir={'qm_utils':
                 'qm_utils'},
    entry_points={
        'console_scripts': ['cp_params=qm_utils.cp_params:main',
                            'read_pdb=qm_utils.read_pdb:main',
                            'read_sdf=qm_utils.read_sdf:main',
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
