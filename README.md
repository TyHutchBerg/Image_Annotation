# Automated Annotation

Program Flow:

1) Run the code in the terminal follows: python ImageAnnotation.py "location of your video file here"
   It's currently set for .mp4 files only, but you can change this by modifying line 252 to the desired video format

2) <p style="text-align:left;">The program generates a directory for each model name in the following format:</p>
  
      	|-- Model Name
		    |-- dataset
			     |-- images
			     |-- labels

<p style="text-align:left;">3) A menu appears to explain to the user how to run the program and when the following commands are valid: </p>
<p style="text-align:left;">'s': Saves the bounding box and can not longer be changed once pressed</p>
<p style="text-align:left;">'left mouse click down': Left click on the mouse creates the bounding box and sest the upper left hand corner</p>
<p style="text-align:left;">'mouse move': Moving the mouse controls how the bounding box grows</p>
<p style="text-align:left;">'left mouse click up': Sets the final position of the bounding box</p>
<p style="text-align:left;">'1', '2', '3': Saves the current frame as a png and creates a kitti formatted .txt file using the class name Low, Medium, High respectively</p>
<p style="text-align:left;">'q': Terminates the program</p>
   
   Frames and kitti formatted files are stored in the following way
  
  
      	|-- Model Name
		    |-- dataset
			     |-- images
		                 -- 0.png
			     |-- labels
                         	 -- 0.txt
             
<p style="text-align:left;">Each image and text file map to each other using the same name. The index used for each image and text file are incremented after saving a new one.</p>

<p style="text-align:left;">The annotation_index.txt file keeps of the file number you left off at, so you can run the program through other video files and not write over previously saved images.
