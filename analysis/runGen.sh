#!/bin/bash

python analysisImage.py \
--model="../../deep_sort/model/networks/mars-small128.pb" \
--mot_dir="../../MOT16/test" \
--output_dir="MOT16.test.detection" \
