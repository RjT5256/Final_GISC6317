# Final_GISC6317
This is my submission for my final project for GISC 6317 Python Programming for Social Sciences that plays off of the work of Florent Poux, A specialist in 3D Data manipulation

# Change Detection Using Python's open3d

**[Application]** is a Python script that parses through point cloud data using various different methods to conduct change detection from file A -> B

## Features

**Feature 1:** Loading PC data
**Feature 2:** Point-to-Point Mapping
**Feature 3:** Georeferencing
**Feature 4:** Heatmapping
**Feature 5:** Change Detection 
**Feature 6:** File Exporting

## Prereqs

Before you begin, ensure you have met the following requirements

* **Python:** version 3.12 or LOWER installation
* **Conda:** package installer for Python using Anaconda Navigator

## Installation
# %% Imports
import open3d as o3d
import numpy as np
from scipy.spatial import KDTree
import laspy
import os
import sys

## Config
Make sure you create an environment with conda to install all necessary imports and with the correct Python verison
