import cv2 as cv
import numpy as np
np.set_printoptions(suppress=True, precision=6)


class FeatureExtractor:
    def __init__(self):
        self.orb = cv.ORB_create()
        self.bf = cv.BFMatcher(cv.NORM_HAMMING)

    def findKeypoints(self, img):
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        features = cv.goodFeaturesToTrack(img, maxCorners=5000, qualityLevel=0.01, minDistance=7)
        kps = [cv.KeyPoint(x=f[0][0], y=f[0][1], size=20) for f in features]
        kps, des = self.orb.compute(img, kps)
        return kps, des
    
    def matcher(self, frame1, frame2):

        kps1, des1 = frame1.getFeatures()
        kps2, des2 = frame2.getFeatures()
        matches = self.bf.knnMatch(des1,des2, k=2)

        lowe_matches = []
        ratio_thresh = 0.75
        for m,n in matches:
            if(m.distance < 32):
                p1x, p1y = kps1[m.queryIdx].pt
                p2x, p2y = kps2[m.trainIdx].pt
                dist = np.sqrt((p1x-p2x)**2 + (p1y-p2y)**2)
                if(dist < 100):
                    if m.distance < ratio_thresh * n.distance:
                        lowe_matches.append(m)
        matches = lowe_matches

        kps1 = [kps1[m.queryIdx] for m in matches]
        des1 = [des1[m.queryIdx] for m in matches]
        kps2 = [kps2[m.trainIdx] for m in matches]
        des2 = [des2[m.trainIdx] for m in matches]



        frame1.setFeatures(kps1, des1)
        #frame2.setFeatures(kps2, des2)
        return kps2, des2


