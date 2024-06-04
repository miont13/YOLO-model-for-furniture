# YOLO-model-for-furniture
A repository for my work on the given assignment about creating a model in YOLO that needs to be accessible to an endpoint using FastAPI.

In this repository we can find the following files:
- image 12.png -> image of a dining room used in the combination of the masks of a table set
- image 14.jpg -> image of a living room used in predicting the model and getting used to YOLO + the initial mask creation (without combination)
- test.py -> test script to play with all the features implemented (YOLO predicting model with and with/ filters and creation of masks with and with/ combination)
- YOLO.py -> the main implementation of all these features (4 features - 2 YOLO predicting models and 2 masks creation functions)
- functions.py -> the additional functions used for the features (filtering the necessary classes, checking for box overlap, merging two boxes, combining a table set)
- runs -> folder where we can see the predicted boxes on the images that the YOLO model returns
- masks -> folder where we can see the black and white masks of the images created by the YOLO model

Additionally, I implemented some extra features like:
- the possibility of seeing the image within the bounding box for the mask combination function (to check the combination)
- deleting the combined items (chairs) from the boxes of the model
- the filter class function that requested the needed furniture objects from the COCO dataset
- input error handling for the test scrip 
