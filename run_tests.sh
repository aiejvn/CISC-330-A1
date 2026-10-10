#!/bin/bash

python tests/test_task1_frames.py  
python run_task2_markers.py  
python tests/test_task3_projector.py 
python run_task4_marker_projection.py  
python tests/test_task7_reconstruction.py python run_task5_drr.py
python tests/test_task5_drr.py
python tests/test_task5_vertebra.py
python tests/test_task5_qualitative.py
python run_task6_localize.py
python tests/test_task6_localization.py
python tests/test_task8_correspondence.py
