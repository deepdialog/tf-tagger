#!/bin/bash

set -e

isort ./tf_tagger/*.py
yapf --recursive --in-place ./tf_tagger/*.py
flake8 ./tf_tagger/*.py

