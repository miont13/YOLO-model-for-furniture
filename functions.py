import requests  # request the labels of the model

furniture_items = ["chair", "couch", "bed", "dining table"]  # list with items that we are interested in


def filter_classes(furniture_items):
    '''
    Function that filters from a dataset labels the necessary needed labels.
    :param furniture_items: the list with furniture items
    :return: a list with labels indexes that we use
    '''

    labels_url = 'https://raw.githubusercontent.com/amikelive/coco-labels/master/coco-labels-2014_2017.txt'
    data = requests.get(labels_url).content.decode('utf-8')  # decode to access the string
    labels = data.split("\n")  # list with all the labels

    classes = []

    for item in range(len(labels)):
        if labels[item] in furniture_items:
            classes.append(item)

    return classes


def overlaps(box1, box2):
    '''
    Checks if two boxes are overlapping, which is considered if the area of the intersection
    of the boxes if bigger or equal than 1/3 of the area of one of the boxes.
    :param box1: a tuple with coordinates of a bounding box
    :param box2: a tuple with coordinates of a bounding box
    :return: True or False if the boxes overlap or not
    '''
    # Coordinates of the boxes
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2

    # We check if the boxes overlap and return false otherwise
    if x2 < x3 or x4 < x1 or y2 < y3 or y4 < y1:
        return False

    # Calculate overlapping area
    overlapping_area = (min(x2, x4) - max(x1, x3)) * (min(y2, y4) - max(y1, y3))

    # Area of boxes
    area_box1 = (x2 - x1) * (y2 - y1)
    area_box2 = (x4 - x3) * (y4 - y3)

    # Check to see if the intersection of the boxes is big enough (a third of one of the boxes)
    if area_box1 / 3 <= overlapping_area or area_box2 / 3 <= overlapping_area:
        return True


def merge_boxes(box1, box2):
    '''
    Merge two boxes into one.
    :param box1: a tuple with coordinates of a bounding box
    :param box2: a tuple with coordinates of a bounding box
    :return: a tuple with coordinate of the merged box
    '''
    # Merge two bounding boxes into a single one
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2

    return (min(x1, x3), min(y1, y3), max(x2, x4), max(y2, y4))


def combine_tabel_set(label_boxes, delete_chairs=False):
    '''
    Make a table set from the chairs that are overlapping the table. We combine
    the tables with the chairs. We also have the option to keep or delete the
    boxes of the combined chairs.
    :param label_boxes: dictionary with all the labels and the boxes coresponding
    to them in the image
    :param delete_chairs: bool value to determine if we delete or not the chairs boxes
    :return: none, it modifies the given dictionary
    '''
    combined_boxes = []  # the new combined boxes for tables
    chairs_to_remove = []  # used for deleting chairs

    # Go through the list of tables and check if one can be combined with a chair
    for table in label_boxes['dining table']:
        tabel_box = table  # the table box that is going to modify
        for chair in label_boxes['chair']:
            # If they overlap we mergem them
            if overlaps(table, chair):
                tabel_box = merge_boxes(tabel_box, chair)

                # Delete the chars that have been already merged if wanted
                if delete_chairs == True:
                    chairs_to_remove.append(chair)
        # We add the new box of the table in the list
        combined_boxes.append(tabel_box)

    # Remove the merged chairs
    if delete_chairs == True:
        remaining_chairs = []
        for chair in label_boxes['chair']:
            if chair not in chairs_to_remove:
                remaining_chairs.append(chair)
        label_boxes['chair'] = remaining_chairs

    # Add the new values of the boxes of the tables
    label_boxes['dining table'] = combined_boxes
