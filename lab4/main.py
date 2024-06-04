from pathlib import Path

import cv2
import os


def preprocess(image):
    # image preprocessing - histogram equalization
    R, G, B = cv2.split(image)

    output1_R = cv2.equalizeHist(R)
    output1_G = cv2.equalizeHist(G)
    output1_B = cv2.equalizeHist(B)

    image = cv2.merge((output1_R, output1_G, output1_B))
    return image


def feature_extraction(image, mode="SIFT"):
    # surf = cv2.xfeatures2d.SURF_create(400) # surf is a non-free tool
    if mode == "SIFT":
        sift = cv2.xfeatures2d.SIFT_create()
        kp, desc = sift.detectAndCompute(image, None)
        return kp, desc

    if mode == "BRIEF":
        star = cv2.xfeatures2d.StarDetector_create()
        kp = star.detect(image, None)
    elif mode == "FAST":
        fast = cv2.FastFeatureDetector_create()
        fast.setNonmaxSuppression(0)
        kp = fast.detect(image, None)



    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
    kp, desc = brief.compute(image, kp)
    print(kp)
    print(desc)
    return kp, desc

# possible options for image preprocessing are "SIFT", "BRIEF" and "FAST"
# BRIEF doesn't return any keypoints
# FAST throws an error in knnMatch later in the code
PREPROCESSING_MODE = "SIFT"
DATA_DIR = "Real_subset"

if __name__ == "__main__":

    # read original image
    original_image_path = Path("Altered-custom", "1__M_Left_index_finger_zamazanie.BMP")
    original_image = cv2.imread(original_image_path)

    # detect keypoints in original image
    original_keypoints, original_descriptors = feature_extraction(
        preprocess(original_image), PREPROCESSING_MODE
    )

    # iterate through files in "real subset" searching for similar keypoints
    for file in os.listdir(DATA_DIR):

        # read a new image
        image = cv2.imread(Path(DATA_DIR, file))
        kp, desc = feature_extraction(preprocess(image),PREPROCESSING_MODE)

        # find matching keypoints between two images
        flann_matcher = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 10}, {})
        matches = flann_matcher.knnMatch(original_descriptors, desc, k=2)

        # filter out the points using a threshold test. The value of threshold is 0.1
        match_points = [p for p, q in matches if p.distance < 0.1 * q.distance]

        # get number of kepoints to display
        num_keypoints = min(len(original_keypoints), len(kp))

        if len(match_points) > 0:
            percentage_matching = round(len(match_points) / num_keypoints * 100, 2)

            print(f"number of matches    : {len(match_points)}")
            print(f"number of keypoints  : {num_keypoints}")
            print(f"% of matching points : {percentage_matching} %")
            print(f"matching file name   : {file}",end="\n\n")

            result = cv2.drawMatches(
                original_image, original_keypoints, image, kp, match_points, None
            )

            result = cv2.resize(result, None, fx=5, fy=5)
            cv2.imshow("result", result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
