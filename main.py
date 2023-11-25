import os
import cv2
import numpy as np

from classifier import ClassifierInference
from segmentation import character_segmentation


clf = ClassifierInference(
    model_state_path="torch_model.pt", labels_map_json_path="labels2devchar.json"
)


def predict(imagepath):
    """
    input: RGB image of shape (X,Y,3)
    pass plot_images=True, to plot the segmented images
    """
    image = cv2.imread(imagepath)
    seg_imgs = character_segmentation(image, plot_images=False)
    answer = []
    for img in seg_imgs:
        answer.append(clf.predict(img))
    return answer

if __name__ == "__main__":
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    parser.add_argument("--image_path", type=str, default='test_pics/t1.png',help="relative path to word image")
    args = parser.parse_args()
    
    print(args.image_path)
    pred = predict(args.image_path)
    print('Predictions :', pred)