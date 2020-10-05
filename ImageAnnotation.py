import cv2
import numpy as np
import os
import copy
import sys

# DetectNet_v2
# Input size: C * W * H (where C = 1 or 3, W > =480, H >=272 and W, H are multiples of16)
# Image format: JPG, JPEG, PNG
# Label format: KITTI detection

# YOLOv3
# Input size: C * W * H (where C = 1 or 3, W >= 128, H >= 128, W, H are multiples of 32)
# Image format: JPG, JPEG, PNG
# Label format: KITTI detection

# FasterRCNN
# Input size: C * W * H (where C = 1 or 3; W > =160; H >=160)
# Image format: JPG, JPEG, PNG
# Label format: KITTI detection

# RetinaNet
# Input size: C * W * H (where C = 1 or 3, W >= 128, H >= 128, W, H are multiples of 32)
# Image format: JPG, JPEG, PNG
# Label format: KITTI detection

# Directory layout for training images
# |--dataset root
#     |-- images
#         |-- 000000.jpg
#         |-- 000001.jpg
#             .
#             .
#         |-- xxxxxx.jpg
#     |--labels
#         |-- 000000.txt
#         |-- 000001.txt
#             .
#             .
#         |-- xxxxxx.txt

# Format needed for each image used in training in the tlt
# Only class name and bounding box coordinates (min_x, min_y, max_x, max_y)
# need to be defined for the tlt, the rest can use the default
kitti_format = {"Class Name": "",
                "Truncation": 0.0,
                "Occlusion": 0,
                "Alpha": 0.0,
                "min_x": 0,
                "min_y": 0,
                "max_x": 0,
                "max_y": 0,
                "height": 0.0,
                "width": 0.0,
                "length": 0.0,
                "location_x": 0.0,
                "location_y": 0.0,
                "location_z": 0.0,
                "rotation_y": 0.0}

# Stores the minimum width and height of the images accepted by each model
# Multiple represents the value that the height and width should multiples of
model_formats = {'DetectNet_v2': {'width': 480, 'height': 272, 'multiple': 16},
                 'YoloV3': {'width': 128, 'height': 128, 'multiple': 32},
                 'FasterRCNN': {'width': 160, 'height': 160, 'multiple': 1},
                 'RetinaNet': {'width': 128, 'height': 128, 'multiple': 32}}

def resizeImg(img, bounding_box, model_name):
    global model_formats

    # Shape of the original image
    x_shape = img.shape[1]
    y_shape = img.shape[0]

    target_size_dict = model_formats.get(model_name)

    # Model name is not in dictionary
    if target_size_dict is None:
        return False, -1, -1

    # Image is too small and can't be resized
    if target_size_dict.get('width') > x_shape or target_size_dict.get('height') > y_shape:
        return False, -1, -1

    # Largest possible image size following the model's specifications
    x_target = int(x_shape / target_size_dict.get('multiple')) * target_size_dict.get('multiple')
    y_target = int(y_shape / target_size_dict.get('multiple')) * target_size_dict.get('multiple')

    # Scale change of the original image
    x_scale = x_target / x_shape
    y_scale = y_target / y_shape

    # Resize the image to the new target size
    resized_img = cv2.resize(img, (x_target, y_target))

    # Scale the bounding box to fit the resized image
    x_min = int(np.round(bounding_box[0] * x_scale))
    y_min = int(np.round(bounding_box[1] * y_scale))
    x_max = int(np.round(bounding_box[2] * x_scale))
    y_max = int(np.round(bounding_box[3] * y_scale))

    # Draw the new bounding box
    # cv2.rectangle(resized_img, (x_min, y_min), (x_max, y_max), (255, 0, 0), thickness=3)
    # cv2.imshow('resizeImg', resized_img)
    # cv2.waitKey(1)

    return True, resized_img, (x_min, y_min, x_max, y_max)

def saveImg(img, class_name, bounding_box, model_name, img_name):
    global model_formats, kitti_format

    model_dict = model_formats.get(model_name)

    # Return if the model name is invalid
    if model_dict is None:
        return False

    # Finds the path to the model's images and labels directories
    parent_dir = os.path.dirname(os.path.realpath(__file__))
    images_dir = os.path.join(parent_dir, model_name,  'dataset\images')
    labels_dir = os.path.join(parent_dir, model_name, 'dataset\labels')

    if os.path.isdir(images_dir) and os.path.isdir(labels_dir):
        image_filename = str(img_name) + '.png'
        label_filename = str(img_name) + '.txt'

        # Paths to the image and label directories
        image_filename = os.path.join(images_dir, image_filename)
        label_filename = os.path.join(labels_dir, label_filename)

        # Saves the image
        cv2.imwrite(image_filename, img)

        # Creates a textfile for the image following the kitti format
        with open(label_filename, 'w') as writer:
            for key, value in kitti_format.items():
                if key == 'Class Name':
                    writer.write(str(class_name + ' ').rstrip('\n'))
                elif key == 'min_x':
                    writer.write(str(str(bounding_box[0]) + ' ').rstrip('\n'))
                elif key == 'min_y':
                    writer.write(str(str(bounding_box[1]) + ' ').rstrip('\n'))
                elif key == 'max_x':
                    writer.write(str(str(bounding_box[2]) + ' ').rstrip('\n'))
                elif key == 'max_y':
                    writer.write(str(str(bounding_box[3]) + ' ').rstrip('\n'))
                elif key == 'rotation_y':
                    writer.write(str(value))
                else:
                    writer.write(str(str(value) + ' ').rstrip('\n'))

        writer.close()

        print('Saved ' + class_name + ' image ' + image_filename)
        print('Saved ' + class_name + ' kitti file ' + label_filename)

    else:
        return False

    return True

# Returns the length of a video
def getVideoLength(video):
    fps = video.get(cv2.CAP_PROP_FPS)
    numFrames = video.get(cv2.CAP_PROP_FRAME_COUNT)

    length = numFrames / fps
    minutes = int(length/60)
    seconds = int(length % 60)

    print("Video Length (M:S)= " + str(minutes) + ":" + str(seconds))

    # Length of video in seconds
    return length

# Returns the delay needed for playing the video in it's correct FPS
def getVideoDelay(video):
    fps = video.get(cv2.CAP_PROP_FPS)
    delay = int(1 / int(fps) * 1000)  # Delay for each frame in milliseconds

    return delay

# Creates directories for storing the images and kitti files for the models
def createDataDir(model_dir):
    dataset_dir = "dataset"
    images_dir = "images"
    labels_dir = "labels"
    parent_dir = os.path.dirname(os.path.realpath(__file__))

    model_path = os.path.join(parent_dir, model_dir)
    dataset_path = os.path.join(model_path, dataset_dir)
    images_path = os.path.join(dataset_path, images_dir)
    labels_path = os.path.join(dataset_path, labels_dir)

    # Try to create directories, if they don't already exist
    if not os.path.isdir(model_path):
        try:
            os.mkdir(model_path)
        except OSError as error:
            print(error)
        if not os.path.isdir(dataset_path):
            try:
                os.mkdir(dataset_path)
            except OSError as error:
                print(error)

            if not os.path.isdir(images_path):
                try:
                    os.mkdir(images_path)
                except OSError as error:
                    print(error)

            if not os.path.isdir(labels_path):
                try:
                    os.mkdir(labels_path)
                except OSError as error:
                    print(error)

        print(model_name + ' Directory and sub Directories created.')

frame, cache = None, None
drawMode, paused, roi = False, False, False
index = 0
top_left_x, top_left_y = -1, -1
bottom_right_x, bottom_right_y = -1, -1

# Draws the bounding box and saves the
def drawBoundingBox(event, x, y, flags, param):
    global top_left_x, top_left_y, bottom_right_x, bottom_right_y, cache, frame, drawMode, index, roi

    if event == cv2.EVENT_LBUTTONDOWN:
        drawMode = True
        roi = False
        top_left_x, top_left_y = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawMode is True:
            frame = copy.deepcopy(cache)
            bottom_right_x, bottom_right_y = x, y
            cv2.rectangle(frame, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 0, 255), thickness=3)
    elif event == cv2.EVENT_LBUTTONUP:
        frame = copy.deepcopy(cache)
        bottom_right_x, bottom_right_y = x, y
        cv2.rectangle(frame, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 0, 255), thickness=3)
        drawMode = False
        roi = True


if __name__ == '__main__':
    model_name = None

    # Stores the video file passed
    video_file = [i for i in sys.argv if '.mp4' in i]

    # Asks the user which directories they want to create
    while True:
        print('Which Model are We Using:')
        print('1) DetectNet_v2')
        print('2) YoloV3')
        print('3) FasterRCNN')
        print('4) RetinaNet')
        print('5) All')
        choice = input('Choice: ')

        try:
            choice = int(choice)
        except ValueError:
            continue

        if choice in range(1, 6):
            break

    # Creates the directory(s) for the specified model(s)
    if choice == 1:
        model_name = 'DetectNet_v2'
        createDataDir(model_name)
    elif choice == 2:
        model_name = 'YoloV3'
        createDataDir('YoloV3')
    elif choice == 3:
        model_name = 'FasterRCNN'
        createDataDir('FasterRCNN')
    elif choice == 4:
        model_name = 'RetinaNet'
        createDataDir('RetinaNet')
    elif choice:
        model_name = 'All'
        for key, values in model_formats.items():
            createDataDir(key)
    else:
        print('Invalid Model Name')

    # Creates the window for the video, sets the specified mousecallback function and opens the video
    cv2.namedWindow('video')
    cv2.setMouseCallback('video', drawBoundingBox)
    video = cv2.VideoCapture(video_file[0])

    # Sets the video to a designated frame based on the location of the trackbar
    def changeVideoLoc(trackbarValue):
        global frame
        video.set(cv2.CAP_PROP_POS_FRAMES, trackbarValue)
        ret, frame = video.read()
        cv2.imshow('video', frame)

    # Creates a window for video options
    cv2.namedWindow('Video Options')
    numFrames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Creates a trackbar from 0 to the number of frames in the video file, used for controlling video's location
    cv2.createTrackbar('Frame Num', 'Video Options', 0, numFrames, changeVideoLoc)

    # Gets the delay needed to play the video file according to its frame rate
    delay = getVideoDelay(video)

    while(video.isOpened()):
        ret, frame = video.read()
        cache = copy.deepcopy(frame)
        cv2.setTrackbarPos('Frame Num', 'Video Options', int(video.get(cv2.CAP_PROP_POS_FRAMES)))

        cv2.imshow('video', frame)
        key = cv2.waitKey(delay) & 0xFF

        # Pauses the video
        if key == ord('w'):
            while cv2.waitKey(1) & 0xFF != ord('w'):
                cv2.imshow('video', frame)

                # Once a bounding box is drawn, we loop to check if the user wants to save it or to restart the video
                while roi:
                    key = cv2.waitKey(20) & 0xFF
                    class_name = None
                    save_frame = False

                    # Sets the class to Near Overflow
                    if key == ord('s') and roi:
                        class_name = 'Near_OverFlow'
                        save_frame = True
                    # Sets the class to Not Near Overflow
                    elif key == ord('a') and roi:
                        class_name = 'Not_Near_OverFlow'
                        save_frame = True

                    # Remove the bounding box from the frame and redraw the frame
                    elif key == ord('d'):
                        roi = False
                        frame = cache
                        cv2.imshow('video', frame)
                        break

                    # Saves the frame
                    if save_frame and roi:
                        bounding_box = (top_left_x, top_left_y, bottom_right_x, bottom_right_y)

                        # Resizes and saves the frame to all of the model's directories
                        if model_name == 'All':
                            for key, value in model_formats.items():
                                # Resizes the image to the model's specification
                                res, resized_img, resized_bounding_box = resizeImg(cache, bounding_box, key)

                                # Save the frame to the designated model's directory
                                saveImg(resized_img, class_name, resized_bounding_box, key, str(index))

                                # Increments a counter for use in naming future images
                                index = index + 1
                        # Resizes and saves the frame to the selected model
                        else:
                            # Resizes the image to the model's specification
                            res, resized_img, resized_bounding_box = resizeImg(cache, bounding_box, model_name)

                            # Save the frame to the designated model's directory
                            res = saveImg(cache, class_name, resized_bounding_box, model_name, str(index))

                            # Increments a counter for use in naming future images
                            index = index + 1

        # Quits the video and closes all windows
        elif key == ord('q'):
            cv2.destroyAllWindows()
            break