import cv2
import albumentations as A
import os

# Draws the bounding box and class name on the image and displays it
def visualizeAnnotation(image, bbbox, class_name, image_name):
    cv2.rectangle(image, (bbbox[0], bbbox[1]), (bbbox[2], bbbox[3]), (0, 0, 255), thickness=3)
    cv2.putText(image, class_name, (bbbox[0], bbbox[1] - 10), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 0, 255), 2)
    cv2.imshow(image_name, image)


model_dirs = {'DetectNet_v2': {'images_dir': 'DetectNet_v2\\dataset\\images', 'labels_dir': 'DetectNet_v2\\dataset\\labels'},
                 'YoloV3': {'images_dir': 'YoloV3\\dataset\\images', 'labels_dir': 'YoloV3\\dataset\\labels'},
                 'FasterRCNN': {'images_dir': 'FasterRCNN\\dataset\\images', 'labels_dir': 'FasterRCNN\\dataset\\labels'},
                 'RetinaNet': {'images_dir': 'RetinaNet\\dataset\\images', 'labels_dir': 'RetinaNet\\dataset\\labels'}
              }


if __name__ == '__main__':
    # Gets the current index of the number of images saved
    with open('annotation_index.txt', 'r') as reader:
        index = int(reader.readline())

    reader.close()

    # Asks the user which model directory they want to augment images from
    while True:
        print('Augment which Directory:')
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

    # Exits if the selected directory does not exist
    if not (os.path.isdir(images_dir) or os.path.isdir(labels_dir)):
        print('Directory does not exist')
        exit(-1)

    # Asks the user which augmentation they want to apply
    while True:
        print('View Which Augmentation:')
        print('1) Brightness/Contrast')
        print('2) Blur')
        print('3) Weather')
        print('4) All')
        choice = input('Choice: ')

        try:
            choice = int(choice)

            if choice in range(1, 5):
                break

        except ValueError:
            continue

    # Brightness pipeline
    transform_brightness = A.Compose(
        [
            A.OneOf
            (
                [
                    A.RandomBrightnessContrast(brightness_limit=(-.10, 0))
                ],
                p=1
            ),
        ],
        A.BboxParams(format='pascal_voc', label_fields=['class_labels'])
    )

    # Blur pipeline
    transform_blur = A.Compose(
        [
            A.OneOf
            (
                [
                    A.GaussianBlur(),
                    A.MedianBlur(),
                ],
                p=1
            )
        ],
        A.BboxParams(format='pascal_voc', label_fields=['class_labels'])
    )

    # Weather pipeline
    transform_weather = A.Compose(
        [
            A.OneOf
            (
                [
                    A.RandomFog(),
                    A.RandomRain()
                ],
                p=1
            )
        ],
        A.BboxParams(format='pascal_voc', label_fields=['class_labels'])
    )

    # Iterates through all of the images in the directory
    for filename in sorted(os.listdir(images_dir)):
        # Checks if the file is an image
        if not (filename.endswith('.jpg') or filename.endswith('.png')):
            continue

        # Split the file name from it's extension to create the label filename
        image_filename_split = os.path.splitext(filename)
        label_filename = image_filename_split[0] + '.txt'
        label_path = os.path.join(labels_dir, label_filename)
        image_path = os.path.join(images_dir, filename)
        metadata_filename = image_filename_split[0] + '_metadata.txt'
        metadata_path = os.path.join(parent_dir, 'metadata', metadata_filename)

        # Checks if the filename exists returns
        if not os.path.isfile(label_path):
            print(label_path + ' does not exist.')
            break

        # Finds and opens the corresponding label file for the image and reads the kitti format string
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

        if class_name == 'Low':
            continue

        bbox = [[x_min, y_min, x_max, y_max]]
        class_labels = [class_name]

        # Open image and convert it to RGB
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if choice == 1 or choice == 4:
            # Performs a brightness augmentation on the image
            transformed1 = transform_brightness(image=image, bboxes=bbox, class_labels=class_labels)
            transformed1_image = transformed1['image']
            transformed1_class_label = transformed1['class_labels']
            transformed1_bbox = transformed1['bboxes']

            # Displays the brightness augmentation
            visualizeAnnotation(transformed1_image,
                                (transformed1_bbox[0][0], transformed1_bbox[0][1], transformed1_bbox[0][2],
                                 transformed1_bbox[0][3]),
                                transformed1_class_label[0],
                                'Augmentated Brightness')

        if choice == 2 or choice == 4:
            # Performs a blur augmentation on the image
            transformed2 = transform_blur(image=image, bboxes=bbox, class_labels=class_labels)
            transformed2_image = transformed2['image']
            transformed2_class_label = transformed2['class_labels']
            transformed2_bbox = transformed2['bboxes']

            # Displays the blur augmentation
            visualizeAnnotation(transformed2_image,
                                (transformed2_bbox[0][0], transformed2_bbox[0][1], transformed2_bbox[0][2],
                                 transformed2_bbox[0][3]),
                                transformed2_class_label[0],
                                'Augmentated Blur')

        if choice == 3 or choice == 4:
            # Performs a weather augmentation on the image
            transformed3 = transform_weather(image=image, bboxes=bbox, class_labels=class_labels)
            transformed3_image = transformed3['image']
            transformed3_class_label = transformed3['class_labels']
            transformed3_bbox = transformed3['bboxes']

            # Displays the weather augmentation
            visualizeAnnotation(transformed3_image,
                                (transformed3_bbox[0][0], transformed3_bbox[0][1], transformed3_bbox[0][2], transformed3_bbox[0][3]),
                                transformed3_class_label[0],
                                'Augmentated Weather')

        # Displays the original image
        visualizeAnnotation(image, (x_min, y_min, x_max, y_max), class_name, 'Orginal Image')

        key = cv2.waitKey(0) & 0xFF

        # Pressing q exists the loop
        if key == ord('q'):
            break
