import cv2
import os

model_dirs = {'DetectNet_v2': {'images_dir': 'DetectNet_v2\\dataset\\images', 'labels_dir': 'DetectNet_v2\\dataset\\labels'},
                 'YoloV3': {'images_dir': 'YoloV3\\dataset\\images', 'labels_dir': 'YoloV3\\dataset\\labels'},
                 'FasterRCNN': {'images_dir': 'FasterRCNN\\dataset\\images', 'labels_dir': 'FasterRCNN\\dataset\\labels'},
                 'RetinaNet': {'images_dir': 'RetinaNet\\dataset\\images', 'labels_dir': 'RetinaNet\\dataset\\labels'}
              }

if __name__ == '__main__':
    # Finds the model's directory
    # Walks through the directory's images
    # Find the corresponding label for the image in the labels/directory
    # Draw the bounding box and the label on the image
    # Displays the image

    # Asks the user which model directoy they want to display the images from
    while True:
        print('Visualize which Directory:')
        print('1) DetectNet_v2')
        print('2) YoloV3')
        print('3) FasterRCNN')
        print('4) RetinaNet')
        choice = input('Choice: ')

        try:
            choice = int(choice)

            if choice in range(1,5):
                break

        except ValueError:
            continue

    if choice == 1:
        model_name = 'DetectNet_v2'
        model_name_dirs = model_dirs.get(model_name)

    elif choice == 2:
        model_name = 'YoloV3'
        model_name_dirs = model_dirs.get(model_name)

    elif choice == 3:
        model_name = 'FasterRCNN'
        model_name_dirs = model_dirs.get(model_name)

    elif choice == 4:
        model_name = 'RetinaNet'
        model_name_dirs = model_dirs.get(model_name)

    parent_dir = os.path.dirname(os.path.realpath(__file__))
    images_dir = os.path.join(parent_dir, model_name_dirs.get('images_dir'))
    labels_dir = os.path.join(parent_dir, model_name_dirs.get('labels_dir'))

    # Exists if image directory does not exist
    if not os.path.isdir(images_dir):
        print(images_dir + ' could not be found')
        exit(-1)

    # Exits if label directory does not exist
    if not os.path.isdir(labels_dir):
        print(labels_dir + ' could not be found')
        exit(-1)

    for filename in os.listdir(images_dir):
        # Checks if the file is an image
        if not (filename.endswith('.jpg') or filename.endswith('.png')):
            continue

        # Split the file name from it's extension to create the label filename
        image_filename_split = os.path.splitext(filename)
        label_filename = image_filename_split[0] + '.txt'
        label_path = os.path.join(labels_dir, label_filename)
        image_path = os.path.join(images_dir, filename)

        # Prints the image and label path for troubleshooting
        print(image_path)
        print(label_path)

        # Checks if the filename exists returns
        if not os.path.isfile(label_path):
            print(label_path + ' does not exist.')
            break

        # Finds and open the corresponding label file for the image and reads the kitti format string
        with open(label_path, 'r') as reader:
            kitti_format_str = reader.readline()

        reader.close()

        # Splits the kitti format string into the individual parameters
        kitti_format_list = kitti_format_str.split(' ')

        # Index 0 is the class name, indexes 4-7 are the bounding box coordinates in the pascal_voc format
        class_name = kitti_format_list[0]
        x_min = int(kitti_format_list[4])
        y_min = int(kitti_format_list[5])
        x_max = int(kitti_format_list[6])
        y_max = int(kitti_format_list[7])

        # Open image
        image = cv2.imread(image_path)

        # Draw the bounding box and the class name
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), thickness=3)
        cv2.putText(image, class_name, (x_min, y_min - 10), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 0, 255), 2)

        cv2.imshow('Image', image)
        key = cv2.waitKey(0) & 0xFF

        # Pressing q exists the loop
        if key == ord('q'):
            break