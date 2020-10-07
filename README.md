# object segmentation

This repo is based on and uses https://github.com/CSAILVision/semantic-segmentation-pytorch

First, install the prereqs: 
https://github.com/CSAILVision/semantic-segmentation-pytorch#integration-with-other-projects


# Basic usage
This is intended to run in the ARM lab with ROS.
1. start the kinect: `ssh loki` `roslauch mps_launch_files kinect_vicon_real_robot.launch pov:="victor_head"`
2. Download the pretrained model `cd scripts` `./download_models.sh`
3. Run: `python object_segmentation.py`

# YCB
To use a model pretrained on the YCB dataset using NVidia's FAT (Falling simulated YCB objects), download:
https://drive.google.com/file/d/1EwQDgMegTlfzZhscIoZkDofpTm3I2SIJ/view?usp=sharing
