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

def saveImg(img, class_name, bounding_box, model_name, img_name, video_filename, frame_num, fps):
    global model_formats, kitti_format, print_statements

    model_dict = model_formats.get(model_name)

    # Return if the model name is invalid
    if model_dict is None:
        return False

    # Finds the path to the model's images, labels and metadata directories
    parent_dir = os.path.dirname(os.path.realpath(__file__))
    images_dir = os.path.join(parent_dir, model_name,  'dataset\images')
    labels_dir = os.path.join(parent_dir, model_name, 'dataset\labels')
    metadata_dir = os.path.join(parent_dir, 'metadata')

    if os.path.isdir(images_dir) and os.path.isdir(labels_dir) and os.path.isdir(metadata_dir):
        image_filename = str(img_name) + '.png'
        label_filename = str(img_name) + '.txt'
        metadata_filename = str(img_name) + '_metadata.txt'

        # Paths to the image and label directories
        image_filename = os.path.join(images_dir, image_filename)
        label_filename = os.path.join(labels_dir, label_filename)
        metadata_filename = os.path.join(metadata_dir, metadata_filename)

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

        with open(metadata_filename, 'w') as writer:
            metadata = str(video_filename) + ' ' + str(frame_num) + ' ' + getMinuteSec(frame_num, fps)
            writer.write(metadata)

        writer.close()

        if print_statements:
            print('Saved ' + class_name + ' image ' + image_filename)
            print('Saved ' + class_name + ' kitti file ' + label_filename)
            print('Saved ' + ' metadata file ' + metadata_filename)

    else:
        return False

    return True

# Returns the length of a video
def getVideoLength(video):
    fps = video.get(cv2.CAP_PROP_FPS)
    numFrames = video.get(cv2.CAP_PROP_FRAME_COUNT)

    length = numFrames / fps
    minutes = int(length / 60)
    seconds = int(length % 60)

    # print("Video Length (M:S)= " + str(minutes) + ":" + str(seconds))

    # Length of video in seconds
    return int(length)

def getMinuteSec(frame, fps):
    position = frame / fps
    minutes = int(position/60)
    seconds = int(position % 60)

    if seconds in range(0, 11):
        timestamp = timestamp = str(minutes) + ":0" + str(seconds)
    else:
        timestamp = str(minutes) + ":" + str(seconds)

    return timestamp

# Returns the delay needed for playing the video in it's correct FPS
def getVideoDelay(video):
    fps = video.get(cv2.CAP_PROP_FPS)
    delay = int(1 / int(fps) * 1000)  # Delay for each frame in milliseconds

    return delay

# Creates directories for storing the images and kitti files for the models
def createDataDir(model_name):
    dataset_dir = "dataset"
    images_dir = "images"
    labels_dir = "labels"
    metadata_dir = "metadata"

    parent_dir = os.path.dirname(os.path.realpath(__file__))

    model_path = os.path.join(parent_dir, model_name)
    dataset_path = os.path.join(model_path, dataset_dir)
    images_path = os.path.join(dataset_path, images_dir)
    labels_path = os.path.join(dataset_path, labels_dir)
    metadata_path = os.path.join(parent_dir, metadata_dir)

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

    # Create a directory that stores the metadata associated with each image
    if not os.path.isdir(metadata_path):
        try:
            os.mkdir(metadata_path)
        except OSError as error:
            print(error)


frame, cache = None, None
drawMode, lock_roi = False, False
top_left_x, top_left_y = -1, -1
bottom_right_x, bottom_right_y = -1, -1
print_statements = True

# Gets the current image number from a text file
with open('annotation_index.txt', 'r') as reader:
    index = int(reader.readline())

reader.close()

# Draws the bounding box and saves the
def drawBoundingBox(event, x, y, flags, param):
    global top_left_x, top_left_y, bottom_right_x, bottom_right_y, cache, frame, drawMode, roi, lock_roi

    if event == cv2.EVENT_LBUTTONDOWN:
        if not lock_roi:
            drawMode = True
            top_left_x, top_left_y = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawMode is True:
            frame = copy.deepcopy(cache)
            bottom_right_x, bottom_right_y = x, y
            cv2.rectangle(frame, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 0, 255), thickness=3)
    elif event == cv2.EVENT_LBUTTONUP:
        if not lock_roi:
            frame = copy.deepcopy(cache)
            bottom_right_x, bottom_right_y = x, y
            cv2.rectangle(frame, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 0, 255), thickness=3)
            drawMode = False


if __name__ == '__main__':
    frame_index = 100
    frame_increment = 100
    automated = False
    end_loop = False
    frame_range = False

    # Stores the video file passed
    video_file = [i for i in sys.argv if '.mp4' in i]

    if not video_file:
        print('Invalid video file')
        exit(-1)

    for key, values in model_formats.items():
        createDataDir(key)

    # Asks user if the tank footage is all one type to speed up annotating process
    while True:
        print('Annotation Options')
        print('1) Annotate All Low')
        print('2) Annotate All Medium')
        print('3) Annotate All High')
        print('4) Annotate Range Low')
        print('5) Annotate Range Medium')
        print('6) Annotate Range High')
        print('7) Annotate Range Manually')
        print('8) Annotate All Manually')
        choice = input('Choice: ')

        try:
            choice = int(choice)
        except ValueError:
            continue

        if choice in range(1, 9):
            break

    # Removes print statements from the automated annotation process
    if choice in range(1, 7):
        automated = True
        print_statements = False

    # Creates the window for the video, sets the specified mouse callback function and opens the video
    cv2.namedWindow('video')
    cv2.setMouseCallback('video', drawBoundingBox)
    video = cv2.VideoCapture(video_file[0])

    # If a range choice was selected, determine the range to annotate
    if choice in range(4, 8):
        while True:
            start = input('Starting point in seconds: ')
            end = input('Ending point in seconds: ')

            # Tries to convert start point to integer
            try:
                start = int(start)
            except ValueError:
                continue

            # Checks if the start point is in the right range
            if start < 0 or start > getVideoLength(video):
                continue

            # Tries to convert end point to integer
            try:
                end = int(end)
            except ValueError:
                continue

            # Checks if the end point is in the right range
            if end < 0 or end > getVideoLength(video) or end <= start:
                continue

            # Sets the frame_range to true, begins the video at the starting point and reduces the rate the video progresses
            frame_range = True
            frame_index = start * int(video.get(cv2.CAP_PROP_FPS))
            frame_increment = 50
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            break

    # Iterates through the frames of the video normally
    while video.isOpened():
        # Ends the video
        if end_loop:
            break

        # Increments the current frame by the frame_increment
        frame_index = frame_index + frame_increment

        # Sets the video's frame to the frame_index
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        # Reads the frame and saves a copy for redrawing
        ret, frame = video.read()
        cache = copy.deepcopy(frame)

        # Frame could not be obtained
        if not ret:
            break

        # Reached the end of the range
        if frame_range and frame_index > end * int(video.get(cv2.CAP_PROP_FPS)):
            print('End of range selection')
            break

        # Gets the current frame position in the video and the fps
        frame_num = video.get(cv2.CAP_PROP_POS_FRAMES)
        fps = video.get(cv2.CAP_PROP_FPS)

        # If the we have already locked in a region of interest, redraw the bounding box on the new frame
        if lock_roi:
            cv2.rectangle(frame, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 0, 255), thickness=3)

        cv2.imshow('video', frame)

        # Sets the unchangeable bounding box
        if not lock_roi:
            print("Set the bounding box for the video. Pressing 'f' after drawing the bounding box locks it in to place")

        while cv2.waitKey(1) & 0xFF != ord('s') and not lock_roi:
            cv2.imshow('video', frame)

        # Bounding box can no longer be changed
        lock_roi = True

        bounding_box = (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
        class_name = None

        if automated:
            if choice == 1:
                class_name = 'Low'
            elif choice == 2:
                class_name = 'Medium'
            else:
                class_name = 'High'
        else:
            while True:
                print('Enter Class Name (1, 2, 3) or \'q\' to quit: ')
                # Wait for the key press for the class name
                key = cv2.waitKey(0) & 0xFF
                # Class name determined by user input
                if key == ord('1'):
                    class_name = 'Low'
                    break
                elif key == ord('2'):
                    class_name = 'Medium'
                    break
                elif key == ord('3'):
                    class_name = 'High'
                    break
                elif key == ord('q'):
                    end_loop = True
                    break
                else:
                    continue
            # q was pressed so exiting video footage
            if end_loop:
                break

        # Resizes and saves the frame to all of the model's directories
        for key, value in model_formats.items():
            # Resizes the image to the model's specification
            res, resized_img, resized_bounding_box = resizeImg(cache, bounding_box, key)

            # Save the frame to the designated model's directory
            if res:
                saveImg(resized_img,
                        class_name,
                        resized_bounding_box,
                        key,
                        str(index),
                        video_file[0],
                        frame_num,
                        fps)

        # Increments a counter for use in naming future images
        index = index + 1

    # Closes all windows
    cv2.destroyAllWindows()

    # Stores the image number that the program ends on to be reused next time the program starts up
    with open('annotation_index.txt', 'w') as writer:
        writer.write(str(index))

    writer.close()