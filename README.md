# Image_Annotation

Program Flow:

1) Run the code in the terminal follows: python ImageAnnotation.py "location of your video file here"
   It's currently set for .mp4 files only, but you can change this by modifying line 252 to the desired video format
2) The program will ask you the following:

    Which Model are We Using:
    1) DetectNet_v2
    2) YoloV3
    3) FasterRCNN
    4) RetinaNet
    5) All
  
  Depending on what is pressed a directory will be created using the model name in this format:
  
  |-- Model Name
    |-- dataset
      |-- images
      |-- labels

  Typing 5 will create a directory for each model. These directories are where the images and kitti files will be stored.
  
3) If the file passed on the command line is a proper video file, two windows will appear, one is running your video file and the other one allows you to change the frame location in your video

4) Controlling the Video
   'w': Pauses/unpauses the video
   'd': Removes the bounding box from the video frame, can't unpause video until this key is pressed
   'a': Saves the current frame as a png and creates a kitti formatted .txt file using the class name Not_Near_OverFlow, only works if a bounding box is drawn
        and video is paused
   's': Saves the current frame as a png and creates a kitti formatted .txt file using the class name Near_OverFlow, only works if a bounding box is drawn
        and video is paused
   'q': Terminates the program if the video is unpaused
   
   Frames and kitti formatted files are stored in the following way
  
      |-- Model Name
      
		    |-- dataset
			 
			     |-- images

		            -- 0.png
      
			     |-- labels
                  -- 0.txt
              
    Each image and text file map to each other using the same name. The index used for each image and text file are incremented after saving a new one.

