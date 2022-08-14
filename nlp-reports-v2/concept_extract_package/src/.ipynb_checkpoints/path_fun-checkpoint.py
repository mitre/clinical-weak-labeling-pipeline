
import sys
import yaml
import os


def insert_nlp_lib_to_path(path):
	sys.path.insert(1, path)#inserts it into the workflow
	print(sys.path)


def create_dir(Path):
	if not os.path.exists(Path):
		os.makedirs(Path)
	print(Path + ' created')

