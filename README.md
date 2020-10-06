# Image_Annotation

Program Flow:

1) Run the code in the terminal follows: python ImageAnnotation.py "location of your video file here"
   It's currently set for .mp4 files only, but you can change this by modifying line 252 to the desired video format

2) The program will ask you what model you are using.
<p style="text-align:left;">Depending on what is pressed a directory will be created using the model name in this format:</p>
  
      	|-- Model Name
		    |-- dataset
			     |-- images
			     |-- labels

<p style="text-align:left;">Typing 5 will create a directory for each model. These directories are where the images and kitti files will be stored.</p>
3) If the file passed on the command line is a proper video file, two windows will appear, one is running your video file and the other one allows you to change the frame location in your video

<p style="text-align:left;">4) Controlling the Video:</p>
<p style="text-align:left;">'w': Pauses unpauses the video</p>
<p style="text-align:left;">'d': Removes the bounding box from the video frame, can't unpause video until this key is pressed</p>
<p style="text-align:left;">'a': Saves the current frame as a png and creates a kitti formatted .txt file using the class name Not_Near_OverFlow, only works if a bounding box is drawn and video is paused</p>
<p style="text-align:left;">'s': Saves the current frame as a png and creates a kitti formatted .txt file using the class name Near_OverFlow, only works if a bounding box is drawn and video is paused</p>
<p style="text-align:left;">'q': Terminates the program if the video is unpaused</p>
<p style="text-align:left;">'left mouse click down': Left click on the mouse creates the bounding box and sest the upper left hand corner</p>
<p style="text-align:left;">'mouse move': Moving the mouse controls how the bounding box grows</p>
<p style="text-align:left;">'left mouse click up': Sets the final position of the bounding box and at this point the frame can be saved</p>
   
   Frames and kitti formatted files are stored in the following way
  
  
      	|-- Model Name
		    |-- dataset
			     |-- images
		                 -- 0.png
			     |-- labels
                         	 -- 0.txt
             
<p style="text-align:left;">Each image and text file map to each other using the same name. The index used for each image and text file are incremented after saving a new one.</p>

