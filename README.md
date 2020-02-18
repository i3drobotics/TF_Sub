# TensorFlow 1.15 (Faster RCNN Inception V2 COCO)
This repository should not be used directly and should instead be used as a sub folder of a directory of your own dataset.
See [here](https://github.com/i3drobotics/TF_Object_Detect_Template) for a template repository.

## Install Dependencies
Install python 3.5+ *(tested up to Python v3.6.8)*

  * **Linux**
    ```
    sudo apt-get install python3.5
    ```
  * **Windows**

    Download and install python from [here](https://www.python.org/downloads/)

Install tensorflow pip package
```
python -m pip install tensorflow==1.15
```
Install opencv pip package
```
python -m pip install opencv-python
```
Install git

  * *Linux*
    ```
    sudo apt-get install git
    ```
  * **Windows**

    Download and install git from [here](https://git-scm.com/download/win)

Install git LFS

  * **Linux**
    ```
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
    sudo apt-get install git-lfs
    git lfs install
    ```
  * **Windows**

    Download Windows Installer from [here](https://github.com/git-lfs/git-lfs/releases) then run:
    ```
    git lfs install
    ```
**[Optional]**

Install TensorFlow GPU

For faster training and classifying you can use the GPU implimentation of TensorFlow using CUDA. This requires a NVidia GPU with driver version >=410.48. See [here](https://docs.nvidia.com/deploy/cuda-compatibility/index.html) for full compatibility details of CUDA. 
```
python -m pip install tensorflow-gpu==1.15
```
See [here](https://www.tensorflow.org/install/gpu) for detailed install instructions installing GPU tensorflow for Linux/Windows (Currently required to use v1.15).

## Create repository (Option 1)
For easy access and running across multiple machines it is advised to create a git repository for the dataset. To set this up create a folder on your computer (e.g. TF_CatPlants_Detect) then clone this example template repository that has everything setup to your folder.
```
cd PATH_TO_YOUR_DIR/..
git clone --recursive https://github.com/i3drobotics/TF_Object_Detect_Template.git YOUR_FOLDER
```
Push this repository to your own git repository url
- Go to [GitHub](https://www.github.com).
- Log in to your account.
- Click the new repository button in the top-right. You’ll have an option there to initialize the repository with a README file, but don’t.
- Enter a name for your repository (e.g. 'ML_TF_CatPlants_Detect')
- Click the “Create repository” button.
- Make a note of the repository url
```
cd PATH_TO_YOUR_DIR
git remote set-url origin https://github.com/USERNAME/REPOSITORY.git
git push -u origin master
```
This template has gitignore, lfs, and submodule setup to work out of the box.

To make sure you are always using the latest version of TF_Sub pull the master branch:
```
cd PATH_TO_YOUR_FOLDER/TF_Sub
git checkout master
git pull origin master
```

## Create folder (Option 2)
If you do not want to use Git to store your dataset you can run the repository from a folder on your computer. 

Create a folder e.g. 'TF_CatPlants_Detect'.

Clone this repository to TF_Sub inside your folder
```
cd PATH_TO_YOUR_DIR
git clone https://github.com/i3drobotics/TF_Sub.git TF_Sub
```

## Initialise
To create the folder structure expected run the following python script: 
```
cd PATH_TO_YOUR_DIR/TF_Sub
python create_sample.py --dir=../
```
This will create the following folder structure:
```
PATH_TO_YOUR_DIR
    images
        test
        train
        unknown
    inference_graph
    training
```

## Setup VSCode
VS Code tasks are provided in this repository to make running quick and easy. 

Install VSCode [here](https://code.visualstudio.com/Download)

Open the repository in VSCode by running vscode and then: file -> open folder -> [PATH TO REPO]

*Note:* There appears to be an issue with VSCode where it doesn't load environment variables in Windows unless run as an admin. This can cause problems with git LFS so start vscode as an admin: start -> vscode -> right-click -> run as admin

To run these tasks open this repository inside VS code and press F1. Enter 'Tasks' and look for 'Tasks: Run task'. This will then show a list of tasks that can be run. 

To make sure the 'python.pythonPath' used in the tasks is set open a python script within VSCode and select the opropriate python interpretor. This should create settings.json in .vscode and set the 'python.pythonPath' variable.

## Workspace
All machine learning scripts should be run in the directory [PATH TO REPO]/TF_Sub/models/research/object_detection. Set the current working directory to this directory in the terminal.
```
cd PATH_TO_YOUR_DIR/TF_Sub/models/research/object_detection
```
*NOTE:* If you are running vscode tasks this is handled automatically.

## Dataset
### Data
For machine learning to work a labelled dataset is needed to train the model. Images should be placed in 'images' folder.
They are sorted into three categories; 'Train', 'Test', and 'Unknown'. 'Train' images are the images used to train the machine learning. 'Test' are used in the training process as a measure for how well the training is performing. 'Unknown' are used as images never seen by the training process to manually test the model after training. 

In order to add a label classifier some images are needed of the class. For example if we are adding dogs, 20+ images of dogs should be added to the images in 'images/train' and 5+ images of dogs should be added in 'images/test'. To test completely unseen images once training is complete some more images should be added to 'images/unknown' that are later used to test model. The test dataset is used to create the loss metric while training so images in this folder will match very well. The unknown dataset is used to test the model on completely unseen data that the model has no idea about to give a better idea as the effectiveness of generality. 

For an example dataset see the GitHub repository for detecting Cats and Plants [here](https://github.com/i3drobotics/TF_CatPlant_Template)

### Labeling
The new images in test and train need to be labelled so they can be used in the training process. Included in this repo is a program called 'labelImg' that can be used to tag images with bounding boxes to label the location and class of objects in an image.
  * **Linux**
  ```
  cd [PATH TO REPO]/TF_Sub/programs/labelimg
  sudo ./labelImg
  ```
  * **Windows**
  ```
  [PATH TO REPO]/TF_Sub/programs/labelImg/labelImg.exe
  ```

All of the images in 'Train' and 'Test' images should be labelled with what and where the objects are in the images using bounding boxes. These labels take the form of xml documents for each images. 

![Screenshot of labelled image inside labelImg program](https://github.com/i3drobotics/TF_Sub/raw/master/screenshots/Labeling_Data.PNG "Labeling data using labelimg")

Details on using this program can be found in the programs repository [here](https://github.com/tzutalin/labelImg).

After labeling the classes in an images hit 'Save' to create an XML file for that image. This will appear in the same folder as the image.

### Update parameters
As we have added a class to the dataset the config files need to be updated to inform the scripts of the new class. 

The 'labelmap.pbtxt' in 'training' needs to be edited to include the classes required. In the example below three classes were added 'plant','cat', and 'dog':
```
item {
  id: 1
  name: 'plant'
}

item {
  id: 2
  name: 'cat'
}

item {
  id: 3
  name: 'dog'
}
```

The files 'faster_rcnn_inception_v2_coco_win.config' and 'faster_rcnn_inception_v2_coco_linux.config' in 'training' need to be edited to update the number of classes. (**line 10**)
```
...

9:  faster_rcnn {
10:     num_classes: 3 #edit this line
11:     image_resizer {

...
```
Also the paths to some files needed to updated. (**lines 107, 122, 124, 136, 138**)
```
...

107: fine_tune_checkpoint: "PATH_TO_YOUR_DIR/TF_Sub/models/research/object_detection/faster_rcnn_inception_v2_coco_2018_01_28/model.ckpt"

...

122: input_path: "PATH_TO_YOUR_DIR/train.record"
123: }
124: label_map_path: "PATH_TO_YOUR_DIR/training/labelmap.pbtxt"

...

136: input_path: "PATH_TO_YOUR_DIR/test.record"
137: }
138: label_map_path: "PATH_TO_YOUR_DIR/training/labelmap.pbtxt"

...
```

For use in Tensorflow these xml files need to be converted to a single CSV file the lists the filenames and bounding boxes for each dataset. This is performed with the following python script: 
```
python xml_to_csv.py --images_dir=../../../../images
```
Or using the vscode task: **TF: XML to CSV**

Tensorflow cannot actually read csv files directly so TFRecord files must be generated for each dataset. This is performed with the 'generate_tfrecord.py' python script:
```
python generate_tfrecord.py --csv_input=../../../../images/train_labels.csv --image_dir=../../../../images/train --labelmap=../../../../training/labelmap.pbtxt --output_path=../../../../train.record
python generate_tfrecord.py --csv_input=../../../../images/test_labels.csv --image_dir=../../../../images/test --labelmap=../../../../training/labelmap.pbtxt --output_path=../../../../test.record
```
Or using the vscode tasks: **TF: generateTFrecords: train**, **TF: generateTFrecords: test**

### Train
Training is performed with the 'train.py' python script: 
```
[Linux]
python train.py --logtostderr --train_dir=../../../../training/ --pipeline_config_path=../../../../training/faster_rcnn_inception_v2_coco_linux.config

[Windows]
python train.py --logtostderr --train_dir=../../../../training/ --pipeline_config_path=../../../../training/faster_rcnn_inception_v2_coco_win.config
```
Where 'faster_rcnn_inception_v2_coco_*OS*.config' is the pipline config file found in 'training/'. This should be 'faster_rcnn_inception_v2_coco_win.config' if running on windows and 'faster_rcnn_inception_v2_coco_linux.config' if running on linux. This will overwrite the contents of pipline.config. 

This will create a checkpoint every 10 mins. A checkpoint allows for monitoring while the training process is running as well as a safety should a problem occur whilst training process is running. 

The training can be monitored using 'Tensorboard'. This creates a local server the shows the training loss graphs. Run the following command in the terminal to star the server:
```
tensorboard --logdir=../../../../training --host localhost --port 8088
```
This can then be viewed by going to the following address in your default browser http://localhost:8088/

There are two vscode tasks for this: 

**TF: tensorboard: start** to start the server. 

**TF: tensorboard: show** to open the address in google-chrome

![Screenshot of loss graphs](https://github.com/i3drobotics/TF_Sub/raw/master/screenshots/Loss_Graphs.PNG "Tensorboard Loss graphs")

Training will continue untill 200000 steps is reached as this is expected to be a point where no extra learning can be acheived. However, this can be extended if needed by editing faster_rcnn_inception_v2_coco_*OS*.config **line 113**. 

Training can be stopped when the graphs show a loss of less than 0.1. This can be done by simply pressing CTRL+C in the terminal that is running the training.

The same 'train.py' python script will resume a training session if there is checkpoint data in the 'training' folder.

## Export graph
A tensorflow checkpoint file is not useful in its current state so an intererence graph must be created from the model. This can be done using the python script 'export_inference_graph.py':
```
python export_inference_graph.py --input_type image_tensor --pipeline_config_path=../../../../training/faster_rcnn_inception_v2_coco_OS.config --trained_checkpoint_prefix=../../../../training/model.ckpt-XXXX --output_directory=../../../../inference_graph
```
Where 'model.ckpt-*XXXX*' is the checkpoint file with the largest step count e.g. model.ckpt-107679.

And 'faster_rcnn_inception_v2_coco_*OS*.config' is the pipline config file found in 'training/'. This should be 'faster_rcnn_inception_v2_coco_win.config' if running on windows and 'faster_rcnn_inception_v2_coco_linux.config' if running on linux. This will overwrite the contents of pipline.config. 

Or running the vscode task: **TF: Export graph** then editing the prompt displaying 'model.ckpt-XXXX' to the checkpoint file with the largest step count e.g. model.ckpt-107679. The pipeline config file is chosen automatically.

### Classify
Now that the model has been trained and exported it can be tested using the 'unknown' dataset. This will run the unknown images through the inference graph and classify the cats/plants along with a bounding box. This can be run with the python script: 
```
python classify.py --test_image=../../../../images/unknown/unknownX.jpg --inference_graph=../../../../inference_graph --training_dir=../../../../training
```
Where 'unknown*X*.jpg' is the name of the unknow image to classify. 

Or running the vscode task: **TF: Classify** then editing the prompt displaying unknown*X* to the unknown image filename to classify e.g. 'unknown2'. 

![Screenshot of classified image](https://github.com/i3drobotics/TF_Sub/raw/master/screenshots/TestImage_classified.PNG "Classified image")

Also avaiable for testing the classification is a script that runs openCV Video Capture to classify based on the image from a usb camera or video file. This can be run with the following command:
```
python classify_vid.py --inference_graph=../../../../inference_graph --training_dir=../../../../training [--vid=PATH_TO_VIDEO / --cam=0]
```
For a video file use the option '--vid'.

For a usb camera use the option '--cam'.

### Purge
Should you make some edits to the dataset you will need to remove the files created of the old dataset. Any residule training models should be deleted. This can be done by running:
```
python purge.py dir=../../../..
```

Or running the vscode task: **TF: Purge**

### Re-Train
Now the workspace is clean you can run all the steps from the previous sections. Run the following scripts one after another (this includes purging the demo model):
```
python purge.py dir=../../../..
python xml_to_csv.py --images_dir=../../../../images
python generate_tfrecord.py --csv_input=../../../../images/train_labels.csv --image_dir=../../../../images/train --labelmap=../../../../training/labelmap.pbtxt --output_path=../../../../train.record
python generate_tfrecord.py --csv_input=../../../../images/test_labels.csv --image_dir=../../../../images/test --labelmap=../../../../training/labelmap.pbtxt --output_path=../../../../test.record
[Linux]
python train.py --logtostderr --train_dir=../../../../training/ --pipeline_config_path=../../../../training/faster_rcnn_inception_v2_coco_linux.config
[Windows]
python train.py --logtostderr --train_dir=../../../../training/ --pipeline_config_path=../../../../training/faster_rcnn_inception_v2_coco_win.config
```
Or run the vscode task: **TF: Train: New** (this includes purging the demo model)

### Standalone
For standlone running of the classification there is a 'standalone' folder which holds all the scripts needed to load a model and run the classification. Once the model in the repository has been trained and exported, copy the 'frozen_inference_graph.pb' from 'inference_graph' folder in the repository to the 'Model' folder in the standalone directory. Also copy the 'labelmap.pbtxt' inside 'training' to the same 'Model' directory in the standalone folder. Alternatively run the following script:
```
cd PATH_TO_YOUR_DIR/TF_Sub
generate_standalone.py --inference_graph=../inference_graph --training_dir=../training --standalone_dir=../standalone
```
Or copy the required files into the standalone directory by running the vscode task: **TF: Generate standalone**

This 'standalone' folder could now be copied anywhere required and it has all the data and scripts needed to classify. 
Classification can be run the same way as previously as this same script is included in the 'standalone' directory:
```
cd PATH_TO_YOUR_DIR/standalone/scripts/object_detection
classify_vid.py --inference_graph=../../Model --training_dir=../../Model --output_dir=../../Result --cam=0 --flip --split
```
Provided is a vscode task within the standalone directory. To run this open the standalone directory with vscode then hit F1-> Tasks: Run Task -> TF: Classify: Camera Feed