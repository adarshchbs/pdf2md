"""Copyright (C) 2022 Adarsh Gupta"""
import sys
from os.path import dirname, realpath

repo_path = dirname(dirname(dirname(dirname(realpath(__file__)))))
sys.path.append(repo_path)
