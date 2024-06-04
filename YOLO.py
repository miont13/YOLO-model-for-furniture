import cv2
from ultralytics import YOLO
from PIL import Image
from pathlib import Path
import numpy as np
from functions import furniture_items, filter_classes, combine_tabel_set


model = YOLO("yolov8n.pt")  # load the YOLO predicting model


def model_without_filters(im1_path):
    '''
    Run a YOLO model that is predicting items in an image without filters.
    Default implementation.
    :param im1_path: path of the image that we want to use
    :return: saves and opens an image that has the predicted boxes in it and
    returns the results obtained (these are going to be used in the mask creation)
    '''
    im1 = Image.open(im1_path)  # open the image that we want to make prediction on
    results = model.predict(source=im1, save=True)  # generate the result and save plotted images

    saved_image = Image.open(f"runs/detect/predict/{im1_path}")  # open the predicted saved image
    saved_image.show()
    return results


def model_with_filters(im1_path):
    '''
    Run a YOLO model that is predicting items in an image with filters.
    It will only have furniture items that can be found in the furniture_items list
    :param im1_path: path of the image that we want to use
    :return: saves and opens an image that has the predicted boxes in it
    '''
    im1 = Image.open(im1_path)  # open the image that we want to make prediction on
    filtered_results = model.predict(source=im1,
                                     save=True,
                                     classes=filter_classes(
                                         furniture_items))  # generate the result and save plotted images

    saved_image = Image.open(f"runs/detect/predict/{im1_path}")  # open the predicted saved image
    saved_image.show()
    return filtered_results


def masks(results):
    '''
    Based on the results of a model it create the masks for the elements.
    :param results: the YOLO object that is returned after the predicated model runs
    :return: none, it saves maks on the masks directory and display them
    '''
    #  Iterate detection results (helpful for multiple images)
    for result in results:
        img = np.copy(result.orig_img)
        img_name = Path(result.path).stem  # source image base-name

        # Iterate each object contour (multiple detections)
        for furniture_object_index, furniture_object in enumerate(result):
            #  Get detection class name for names when we are going to save the image
            label = furniture_object.names[furniture_object.boxes.cls.tolist().pop()]
            # print(label)

            # Create binary mask
            b_mask = np.zeros(img.shape[:2], np.uint8)

            # Take the coordinate of the box of the detected furniture object
            x1, y1, x2, y2 = furniture_object.boxes.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Draw rectangle contour onto the mask
            white_box_object = cv2.rectangle(b_mask, (x1, y1), (x2, y2), (255), cv2.FILLED)

            # Save white box object to file
            _ = cv2.imwrite(f"masks/{img_name}_{label}-{furniture_object_index}.png", white_box_object)
            saved_img = Image.open(f"masks/{img_name}_{label}-{furniture_object_index}.png")
            saved_img.show()


def model_with_overlapping_masks(filtered_results):
    '''
    Based on the results of a model it combines the boxes that form a set and creates
    masks for the combined furniture objects.
    :param filtered_results: the YOLO object that is returned after the predicated model runs
    :return: none, it saves maks on the masks directory and display them
    '''
    # Create a dictionary to store bounding boxes for each furniture piece
    label_boxes = dict()
    for item in furniture_items:
        label_boxes[item] = []
    print(label_boxes)

    # im1_path = "image 12.png"
    # im1 = Image.open(im1_path)  # open the image that we want to make prediction on
    # filtered_results = model.predict(source=im1,
    #                                  save=True,
    #                                  classes=filter_classes(
    #                                      furniture_items))  # generate the result and save plotted images
    #
    # saved_image = Image.open(f"runs/detect/predict/{im1_path}")  # open the predicted saved image
    # saved_image.show()

    # Iterate detection results (helpful for multiple images) to add them in the dictionary
    for result in filtered_results:
        img = np.copy(result.orig_img)
        img_name = Path(result.path).stem  # source image base-name

        # Iterate each object contour (multiple detections)
        for furniture_object_index, furniture_object in enumerate(result):
            # Get detection class name for names when we are going to save the image
            label = furniture_object.names[furniture_object.boxes.cls.tolist().pop()]
            # Take the coordinate of the box of the detected furniture object
            x1, y1, x2, y2 = furniture_object.boxes.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Add the object to the corresponding label
            label_boxes[label].append((x1, y1, x2, y2))

        # We want to combine the table class with the chairs and delete the chairs that are not used
        combine_tabel_set(label_boxes, delete_chairs=True)

        # Create masks for each combined bounding box
        for label, boxes in label_boxes.items():
            # For every furniture item type we extract the box of its instance
            for box_index, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                # Create binary mask
                b_mask = np.zeros(img.shape[:2], np.uint8)
                # Draw rectangle contour onto the mask
                cv2.rectangle(b_mask, (x1, y1), (x2, y2), (255), cv2.FILLED)
                # Uncomment the next line to check that chairs are merged
                # isolated = np.dstack([img, b_mask])

                # Save white box object to file named masks
                # Replace 'b_mask' with 'isolated' to check the inclusion
                _ = cv2.imwrite(f"masks/{img_name}_{label}-{box_index}.png", b_mask)
                saved_img = Image.open(f"masks/{img_name}_{label}-{box_index}.png")
                saved_img.show()
