"""
    helper package to import simulator
"""

import os
import sys

file_path_ = os.path.realpath(__file__)
dir_path_ = os.path.dirname(file_path_)

# default path
CPP_SIMULATOR_PATH = os.path.join(dir_path_, '..', '..')

if 'CPP_SIMULATOR_PATH' in os.environ:
    CPP_SIMULATOR_PATH = os.environ['CPP_SIMULATOR_PATH']

sys.path.append(CPP_SIMULATOR_PATH)

from SimDota2 import *

