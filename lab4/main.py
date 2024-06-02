import cv2
import os

test_original = cv2.imread("Altered-custom/1__M_Left_index_finger_szum.BMP")

dataDir = "Real_subset"


def preprocess(image):
    # todo preprocessing
    R, G, B = cv2.split(image)

    output1_R = cv2.equalizeHist(R)
    output1_G = cv2.equalizeHist(G)
    output1_B = cv2.equalizeHist(B)

    image = cv2.merge((output1_R, output1_G, output1_B))
    return image


def features_extraction(image, mode = "SIFT"):
    # todo feature extraction - sift, surf, fast, brief ...
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

    return kp, desc





test_preprocessed = preprocess(test_original)
cv2.imshow("Original", cv2.resize(test_preprocessed, None, fx=1, fy=1))
cv2.waitKey(0)
cv2.destroyAllWindows()



for mode in ["BRIEF", "FAST","SIFT", ]:
    keypoints_1, descriptors_1 = features_extraction(test_preprocessed,mode)

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
