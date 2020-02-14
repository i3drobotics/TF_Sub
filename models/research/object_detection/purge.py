import os
import shutil
import glob
import argparse

parser = argparse.ArgumentParser(description='Purge')
parser.add_argument("--dir", help = "")

args = parser.parse_args()

def yes_or_no(question):
    answer = input(question + "(y/n): ").lower().strip()
    print("")
    while not(answer == "y" or answer == "yes" or \
    answer == "n" or answer == "no"):
        print("Input yes or no")
        answer = input(question + "(y/n):").lower().strip()
        print("")
    if answer[0] == "y":
        return True
    else:
        return False

cwd = os.getcwd()
files2CleanList = []
files2CleanList = files2CleanList + glob.glob(os.path.join(cwd,args.dir,'training\\events.out.tfevents*'))
files2CleanList = files2CleanList + glob.glob(os.path.join(cwd,args.dir,'training\\model.ckpt*'))
files2CleanList = files2CleanList + glob.glob(os.path.join(cwd,args.dir,'training\\graph.pbtxt'))
files2CleanList = files2CleanList + glob.glob(os.path.join(cwd,args.dir,'training\\pipeline.config'))
files2CleanList = files2CleanList + glob.glob(os.path.join(cwd,args.dir,'training\\checkpoint'))
files2CleanList = files2CleanList + glob.glob(os.path.join(cwd,args.dir,'inference_graph\\*'))
files2CleanList = files2CleanList + glob.glob(os.path.join(cwd,args.dir,'images\\test_labels.csv'))
files2CleanList = files2CleanList + glob.glob(os.path.join(cwd,args.dir,'images\\train_labels.csv'))

if yes_or_no("Are you sure you would like to purge the training data?"):
    # Iterate over the list of filepaths & remove each file.
    for filePath in files2CleanList:
        print("Deleting file... ",filePath)
        try:
            os.remove(filePath)
        except:
            print("Error while deleting file : {}. Trying to delete it as a folder.".format(filePath))
            try:
                print("Deleting folder... ",filePath)
                shutil.rmtree(filePath)
            except:
                print("Error while deleting : ",filePath)
            

