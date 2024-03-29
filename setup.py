#!/usr/bin/env python3
#-*- coding:utf-8 -*-

#############################################
# File Name: setup.py
# Author: DeepDialog
# Mail: thebotbot@sina.com
# Created Time: 2018-10-23 20:00:00
#############################################

import os
from setuptools import setup, find_packages

current_dir = os.path.realpath(os.path.dirname(__file__))

ON_RTD = os.environ.get('READTHEDOCS') == 'True'
if not ON_RTD:
    INSTALL_REQUIRES = open(os.path.join(
        current_dir,
        'requirements.txt'
    )).read().split('\n')
else:
    INSTALL_REQUIRES = []

VERSION = os.path.join(
    current_dir,
    'tf_tagger',
    'version.txt'
)

setup(
    name='tf-tagger',
    version=open(VERSION, 'r').read().strip(),
    keywords=('pip', 'tensorflow', 'NER', 'tagger'),
    description='NLP tool',
    long_description='NLP tool, NER, POS',
    license='Private',
    url='https://github.com/deepdialog/tf-tagger',
    author='deepdialog',
    author_email='thebotbot@sina.com',
    packages=find_packages(),
    include_package_data=True,
    platforms='any',
    install_requires=INSTALL_REQUIRES
)
