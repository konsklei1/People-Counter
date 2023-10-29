# People-Counter
A repository containing the code and related data for a people counting system. This system uses computer vision and the MobileNet SSD object detection model to accurately track and count people entering and exiting a defined space, in this example the indoor gym of Komotini. 

In order to be able to use the people counter you must have a Raspberry pi 3 and the required peripherals needed for it to work. 
Also make sure that the OS has python 3, otherwise install it and continue with the steps below.

# Step 1
Install Environment and Necessary Libraries
In order to make it right see this tutorial:
https://www.youtube.com/watch?v=QzVYnG-WaM4

Install OpenCV
Install Dlib
Install NumPy (Python)
Install argparse (Python)
Install imutils (Python)

# Step 2 
Download the code from this GitHub repository.

# Step 3
Open a terminal pointing to the location you saved the folder with the code you downloaded.

# Step 4

Run this command:

python people_counter.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt --model mobilenet_ssd/MobileNetSSD_deploy.caffemodel --input videos/Example_Video.mp4 --output output/Example_Output.avi

if you want to use a video that you have in the 'video' subfolder.

Otherwise run this command:

python people_counter.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt --model mobilenet_ssd/MobileNetSSD_deploy.caffemodel --output output/camera_output.avi

if you want to have the live camera image as input.




