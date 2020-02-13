import os
import shutil
import glob

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
files2CleanList = files2CleanList + glob.glob(os.path.join(cwd,'training\\events.out.tfevents*'))
files2CleanList = files2CleanList + glob.glob(os.path.join(cwd,'training\\model.ckpt*'))
files2CleanList = files2CleanList + glob.glob(os.path.join(cwd,'training\\graph.pbtxt'))
files2CleanList = files2CleanList + glob.glob(os.path.join(cwd,'training\\pipeline.config'))
files2CleanList = files2CleanList + glob.glob(os.path.join(cwd,'training\\checkpoint'))
files2CleanList = files2CleanList + glob.glob(os.path.join(cwd,'inference_graph\\*'))

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
            

