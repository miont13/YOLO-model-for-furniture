from YOLO import model_without_filters
from YOLO import model_with_filters
from YOLO import masks
from YOLO import model_with_overlapping_masks

# Images that we can use to check the features. If we want to check another image change im1_path or im2_path
im1_path = "image 14.jpg"
im2_path = "image 12.png"


#######################################################################
##### NOTE FOR USER: After each run if you want to see the result #####
##### image, make sure to relete the predicted folder befor running ###
##### another test. This count also for the masks folder. Delete ######
########## its contents before running somthing else. #################
#######################################################################


while True:
    try:
        testing = int(input('Select one ot the features that you want to use\n'
                            '0 - We use a YOLO predict model without filters (default model)\n'
                            '1 - We use a YOLO predict model with filters\n'
                            '2 - We create masks for a YOLO model with filters (no class combination)\n'
                            '3 - We create masks for a YOLO model with filters\n'
                            'Please select one of the above:  '))
    except ValueError:
        print("Sorry, I didn't understand that.")
        continue

    if testing not in [0, 1, 2, 3]:
        print("Sorry, your response must either 0, 1, 2 or 3")
        continue

    else:
        break

if testing == 0:
    while True:
        try:
            image_use = int(input('Which image do you want to use:\n'
                                  '0 - Living room image\n'
                                  '1 - Dining room image\n'
                                  'Please select one of the above:  '))
        except ValueError:
            print("Sorry, I didn't understand that.")
            continue

        if image_use not in [0, 1]:
            print("Sorry, your response must either 0 or 1")
            continue
        else:
            break

    if image_use == 0:
        model_without_filters(im1_path)
    else:
        model_without_filters(im2_path)

if testing == 1:
    while True:
        try:
            image_use = int(input('Which image do you want to use:\n'
                                  '0 - Living room image\n'
                                  '1 - Dining room image\n'
                                  'Please select one of the above:  '))
        except ValueError:
            print("Sorry, I didn't understand that.")
            continue

        if image_use not in [0, 1]:
            print("Sorry, your response must either 0 or 1")
            continue
        else:
            break

    if image_use == 0:
        model_with_filters(im1_path)
    else:
        model_with_filters(im2_path)

if testing == 2:
    while True:
        try:
            image_use = int(input('Which image do you want to use:\n'
                                  '0 - Living room image\n'
                                  '1 - Dining room image\n'
                                  'Please select one of the above:  '))
        except ValueError:
            print("Sorry, I didn't understand that.")
            continue

        if image_use not in [0, 1]:
            print("Sorry, your response must either 0 or 1")
            continue
        else:
            break

    if image_use == 0:
        results = model_with_filters(im1_path)
        masks(results)
    else:
        results = model_with_filters(im2_path)
        masks(results)

if testing == 3:
    while True:
        try:
            image_use = int(input('Which image do you want to use:\n'
                                  '0 - Living room image\n'
                                  '1 - Dining room image\n'
                                  'Please select one of the above:  '))
        except ValueError:
            print("Sorry, I didn't understand that.")
            continue

        if image_use not in [0, 1]:
            print("Sorry, your response must either 0 or 1")
            continue
        else:
            break

    if image_use == 0:
        results = model_with_filters(im1_path)
        model_with_overlapping_masks(results)
    else:
        results = model_with_filters(im2_path)
        model_with_overlapping_masks(results)
