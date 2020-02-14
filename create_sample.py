import argparse
import os
import shutil
from distutils.dir_util import copy_tree

parser = argparse.ArgumentParser(description='create sample')
parser.add_argument("--dir", help = "", default="..")

args = parser.parse_args()
cwd = os.getcwd()
samples_dir = os.path.join(os.getcwd(),'samples')
out_dir = os.path.join(os.getcwd(),args.dir)
copy_tree(samples_dir, out_dir)