import cv2
import os

test_original = cv2.imread("Altered-custom/1__M_Left_index_finger_szum.BMP")

dataDir = "Real_subset"


def preprocess(image):
    # todo preprocessing
    img = cv2.equalizeHist(image)
    return image


def features_extraction(image):
    # todo feature extraction - sift, surf, fast, brief ...
    # surf = cv2.xfeatures2d.SURF_create(400) # surf is a non-free tool

    fast = cv2.FastFeatureDetector_create()
    fast.setNonmaxSuppression(0)
    kp = fast.detect(image, None)


    return sift.detectAndCompute(image, None)


test_preprocessed = preprocess(test_original)
cv2.imshow("Original", cv2.resize(test_preprocessed, None, fx=1, fy=1))
cv2.waitKey(0)
cv2.destroyAllWindows()

sift = cv2.xfeatures2d.SIFT_create()


keypoints_1, descriptors_1 = features_extraction(test_preprocessed)

for file in [file for file in os.listdir("Real_subset")]:
    fingerprint_database_image = cv2.imread("./Real_subset/" + file)

    keypoints_2, descriptors_2 = features_extraction(
        preprocess(fingerprint_database_image)
    )

    matches = cv2.FlannBasedMatcher(dict(algorithm=1, trees=10), dict()).knnMatch(
        descriptors_1, descriptors_2, k=2
    )

    match_points = []

    for p, q in matches:
        if p.distance < 0.1 * q.distance:
            match_points.append(p)

    keypoints = 0

    if len(keypoints_1) <= len(keypoints_2):
        keypoints = len(keypoints_1)
    else:
        keypoints = len(keypoints_2)

    if (len(match_points) / keypoints) > 0:
        print("% match: ", len(match_points) / keypoints * 100)
        print("Fingerprint ID: " + str(file))

        result = cv2.drawMatches(
            test_original,
            keypoints_1,
            fingerprint_database_image,
            keypoints_2,
            match_points,
            None,
        )

        result = cv2.resize(result, None, fx=2.5, fy=2.5)
        cv2.imshow("result", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print(str(len(match_points)) + " " + str(keypoints))
