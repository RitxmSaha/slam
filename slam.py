import cv2 as cv
import open3d as o3d
import g2o
import numpy as np
import features
import threading
np.set_printoptions(suppress=True, precision=6)

class Frame:
    def __init__(self, kps, des):
        self.kps = np.array(kps)
        self.des = np.array(des)
        self.R = 0
        self.t = 0

    def getFeatures(self):
        return (self.kps, self.des)
    
    def setFeatures(self, kps, des):
        self.kps = np.array(kps)
        self.des = np.array(des)

class SLAM:
    def __init__(self):
        self.focal = 0
        self.width = 0
        self.height = 0
        self.K = 0
        self.frames = []
        self.index = -1
        self.prev = 0
        self.curr = 0

    def setFocol(self, focal):
        self.focal = focal
        self.K = np.array([[self.focal, 0         , self.width/2 ],
                      [0         , self.focal, self.height/2],
                      [0         , 0         , 1            ]])
        
    def addFrame(self, frame):
        self.frames.append(frame)
        self.index += 1
        self.prev = self.curr
        self.curr = self.frames[self.index]

def vizualize_slam(slam):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    vis.add_geometry(mesh)
    


        
path = "test_countryroad.mp4"
slam = SLAM()
fe = features.FeatureExtractor()


vis_thread = threading.Thread(target=vizualize_slam, args=(slam,))
vis_thread.start()

cap = cv.VideoCapture(path)

if not cap.isOpened():
    print("Error: Could not read video")
    exit()

#initial frame
ret, frame = cap.read()
slam.height, slam.width = frame.shape[:2]
slam.setFocol(500)

kps, des = fe.findKeypoints(frame)
new_frame = Frame(kps, des)
slam.addFrame(new_frame)


while cap.isOpened():
    #print(slam.index)
    ret, frame = cap.read()
    kps, des = fe.findKeypoints(frame)
    new_frame = Frame(kps, des)
    slam.addFrame(new_frame)
    
    kps2, des2 = fe.matcher(slam.prev,slam.curr)
    kps1, des1 = slam.prev.kps, slam.prev.des

    pts1 = np.float32([kp.pt for kp in kps1])
    pts2 = np.float32([kp.pt for kp in kps2])

    E, inliers = cv.findEssentialMat(pts1,pts2,slam.K,cv.RANSAC,prob=0.9999,threshold=2)

    kps1 = [kps1[i] for i in range(len(kps1)) if inliers[i]]
    kps2 = [kps2[i] for i in range(len(kps2)) if inliers[i]]

    pts1 = np.float32([kp.pt for kp in kps1])
    pts2 = np.float32([kp.pt for kp in kps2])

    _, R, t, mask = cv.recoverPose(E, pts1, pts2, slam.K)

    slam.curr.R = R
    slam.curr.t = t

    for i in range(len(kps1)):
        prev = (int(kps1[i].pt[0]),int(kps1[i].pt[1]))
        curr = (int(kps2[i].pt[0]),int(kps2[i].pt[1]))
        cv.circle(frame, prev, 2, color=(0,255,0))
        cv.circle(frame, curr, 2, color=(0,0,255))
        cv.line(frame,curr, prev,(0,0,255), 1)


    


    # kps1, des1 = prev_features
    # kps2, des2 = computeFeatures(gray)
    # prev_features = (kps2, des2)
    
    # matches = bf.knnMatch(des1,des2, k=2)

    # lowe_matches = []
    # ratio_thresh = 0.75
    # for m,n in matches:
    #     if m.distance < ratio_thresh * n.distance:
    #         lowe_matches.append(m)
    # matches = lowe_matches

    # kps1 = ([kps1[m.queryIdx] for m in matches])
    # des1 = ([des1[m.queryIdx] for m in matches])
    # kps2 = ([kps2[m.trainIdx] for m in matches])
    # des2 = ([des2[m.trainIdx] for m in matches])

    # pts1 = np.float32([kp.pt for kp in kps1])
    # pts2 = np.float32([kp.pt for kp in kps2])

    
    # E, inliers = cv.findEssentialMat(pts1, pts2, K, cv.RANSAC, 0.9999, 1)
    # _, R, t, mask = cv.recoverPose(E, pts1, pts2,K)
    # pts1 = [kps1[i] for i in range(len(kps1)) if inliers[i]]
    # pts2 = [kps2[i] for i in range(len(kps2)) if inliers[i]]
    # #print(np.linalg.norm(t))
    # print(np.diag(R))
    # print(str(len(inliers))+" matches")

    # if(t[2] < 0):
    #     count2+= 1
    # #print(count2/(count))
    # print(t.T)

    


    # modified_frame = cv.drawKeypoints(frame, pts2, None, color=(0,255,0), flags=0)
    # for i in range(len(pts1)):
    #     curr = (int(pts2[i].pt[0]),int(pts2[i].pt[1]))
    #     prev = (int(pts1[i].pt[0]),int(pts1[i].pt[1]))
    #     cv.line(modified_frame,curr, prev,(0,0,255),3)



    cv.imshow('Video', frame)

    if cv.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

# def create_optimizer():
#     optimizer = g2o.SparseOptimizer()
#     solver = g2o.BlockSolverX(g2o.LinearSolverEigenX())
#     solver = g2o.OptimizationAlgorithmLevenberg(solver)
#     optimizer.set_algorithm(solver)
#     return optimizer


# optimizer = create_optimizer()






# path = "test_countryroad.mp4"

# focal = 270

# width = 1920
# height = 1080
# center_x = width/2
# center_y = height/2

# # Intrinsic Camera Matrix
# K = np.array([[focal, 0    , center_x],
#               [0    , focal, center_y],
#               [0    , 0    , 1       ]])

# #orb = cv.SIFT_create(contrastThreshold = 0.01, edgeThreshold=100, sigma=1.6)

# cap = cv.VideoCapture(path)
# orb = cv.ORB_create()
# count = 0
# count2 = 0
# bf = cv.BFMatcher(cv.NORM_HAMMING)

# def computeFeatures(frame):

#     feats = cv.goodFeaturesToTrack(frame, 3000, qualityLevel=0.01, minDistance=5)
#     kps = [cv.KeyPoint(x=f[0][0], y=f[0][1], size=20) for f in feats]

#     kps, des = orb.compute(frame, kps)

#     return kps, des


#     grid_size = 1
#     h, w = frame.shape[:2]
#     grid_height, grid_width = h // grid_size, w // grid_size
#     all_keypoints = []
#     all_descriptors = []

#     for i in range(grid_size):
#         for j in range(grid_size):
#             # Crop the image to the current grid
#             grid = frame[i*grid_height:(i+1)*grid_height, j*grid_width:(j+1)*grid_width]
#             feat_coords = cv.goodFeaturesToTrack(grid, 3000, qualityLevel=0.01, minDistance=3)
#             if feat_coords is not None:
#                 keypoints = [cv.KeyPoint(x=f[0][0], y=f[0][1], size=20) for f in feat_coords]

#                 keypoints, descriptors = orb.compute(grid, keypoints)

#                 for kp in keypoints:
#                     kp.pt = (kp.pt[0] + j*grid_width, kp.pt[1] + i*grid_height)

#                 all_keypoints.extend(keypoints)
#                 if descriptors is not None:
#                     all_descriptors.append(descriptors)

#     all_descriptors = np.concatenate(all_descriptors, axis=0)

#     return all_keypoints, all_descriptors



# if not cap.isOpened():
#     print("Error: Could not read video")
#     exit()

# ret, frame = cap.read()
# gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
# kps1, des1 = computeFeatures(gray)
# prev_features = (kps1, des1)
# while cap.isOpened():
#     count += 1

#     ret, frame = cap.read()
#     gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)


#     kps1, des1 = prev_features
#     kps2, des2 = computeFeatures(gray)
#     prev_features = (kps2, des2)
    
#     matches = bf.knnMatch(des1,des2, k=2)

#     lowe_matches = []
#     ratio_thresh = 0.75
#     for m,n in matches:
#         if m.distance < ratio_thresh * n.distance:
#             lowe_matches.append(m)
#     matches = lowe_matches

#     kps1 = ([kps1[m.queryIdx] for m in matches])
#     des1 = ([des1[m.queryIdx] for m in matches])
#     kps2 = ([kps2[m.trainIdx] for m in matches])
#     des2 = ([des2[m.trainIdx] for m in matches])

#     pts1 = np.float32([kp.pt for kp in kps1])
#     pts2 = np.float32([kp.pt for kp in kps2])

    
#     E, inliers = cv.findEssentialMat(pts1, pts2, K, cv.RANSAC, 0.9999, 1)
#     _, R, t, mask = cv.recoverPose(E, pts1, pts2,K)
#     pts1 = [kps1[i] for i in range(len(kps1)) if inliers[i]]
#     pts2 = [kps2[i] for i in range(len(kps2)) if inliers[i]]
#     #print(np.linalg.norm(t))
#     print(np.diag(R))
#     print(str(len(inliers))+" matches")

#     if(t[2] < 0):
#         count2+= 1
#     #print(count2/(count))
#     print(t.T)

    


#     modified_frame = cv.drawKeypoints(frame, pts2, None, color=(0,255,0), flags=0)
#     for i in range(len(pts1)):
#         curr = (int(pts2[i].pt[0]),int(pts2[i].pt[1]))
#         prev = (int(pts1[i].pt[0]),int(pts1[i].pt[1]))
#         cv.line(modified_frame,curr, prev,(0,0,255),3)



#     cv.imshow('Video', modified_frame)

#     if cv.waitKey(25) & 0xFF == ord('q'):
#         break

# cap.release()
# cv.destroyAllWindows()

# def create_optimizer():
#     optimizer = g2o.SparseOptimizer()
#     solver = g2o.BlockSolverX(g2o.LinearSolverEigenX())
#     solver = g2o.OptimizationAlgorithmLevenberg(solver)
#     optimizer.set_algorithm(solver)
#     return optimizer


# optimizer = create_optimizer()
