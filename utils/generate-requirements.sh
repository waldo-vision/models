#!/bin/bash
cd ..
conda env export > environment.yml
conda list -e > requirements.txt
