import argparse
import os
import shutil
from distutils.dir_util import copy_tree

parser = argparse.ArgumentParser(description='generate standalone')
parser.add_argument("--inference_graph", help = "")
parser.add_argument("--training_dir", help = "")
parser.add_argument("--standalone_dir", help = "")

args = parser.parse_args()
cwd = os.getcwd()

inference_graph_dir = os.path.join(os.getcwd(),args.inference_graph)
training_dir = os.path.join(os.getcwd(),args.training_dir)
standalone_dir = os.path.join(os.getcwd(),args.standalone_dir)

graph_file = os.path.join(inference_graph_dir,'frozen_inference_graph.pb')
label_file = os.path.join(training_dir,'labelmap.pbtxt')
scripts_dir = os.path.join(os.getcwd(),'samples/standalone/Scripts')

out_graph_file = os.path.join(standalone_dir,'Model','frozen_inference_graph.pb')
out_label_file = os.path.join(standalone_dir,'Model','labelmap.pbtxt')
out_scripts_dir = os.path.join(standalone_dir,'Scripts')

shutil.copyfile(graph_file, out_graph_file)
shutil.copyfile(label_file, out_label_file)
shutil.rmtree(out_scripts_dir) ## clean scripts directory if it already exists
os.makedirs(out_scripts_dir)
copy_tree(scripts_dir, out_scripts_dir)