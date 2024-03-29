import cv2 as cv
import open3d as o3d
import numpy as np
import features
import optimize
np.set_printoptions(suppress=True, precision=6)

class Frame:
    def __init__(self, kps, des):
        self.kps = np.array(kps)
        self.des = np.array(des)
        self.optimizer_arr = 0
        self.pts = 0
        self.aRt = 0
        self.R = 0
        self.t = 0

    def getFeatures(self):
        return (self.kps, self.des)
    
    def setFeatures(self, kps, des):
        self.kps = np.array(kps)
        self.des = np.array(des)

class SLAM:
    def __init__(self):
        self.ba = optimize.BundleAdjustment()
        self.focal = 0
        self.width = 0
        self.height = 0
        self.K = 0
        self.frames = []
        self.index = -1
        self.prev = 0
        self.curr = 0
        self.Rt = np.eye(4)


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

    def adjust(self):
        self.ba.optimize(self.frames)
    
path = "test_countryroad.mp4"
slam = SLAM()
fe = features.FeatureExtractor()
vis = o3d.visualization.Visualizer()
vis.create_window()
rotation_degrees = 180
rotation_radians = np.deg2rad(rotation_degrees)
camera_boxes = []


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
new_frame.aRt = np.eye(4)
slam.addFrame(new_frame)

while cap.isOpened():
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

    new_frame.R = R
    new_frame.t = t.T

    T = np.eye(4)  # Initialize as 4x4 identity matrix
    T[:3, :3] = R  # Set the upper-left 3x3 block as the rotation matrix
    T[:3, 3] = t.T  # Set the upper-right 3x1 block as the translation vector

    P1 = np.dot(slam.K, np.eye(3, 4))  # Projection matrix for the first camera frame
    P2 = np.dot(slam.K, np.concatenate((R, t), axis=1))  # Projection matrix for the second frame

    # Triangulate points
    pts4d = cv.triangulatePoints(P1, P2, pts1.T, pts2.T)
    pts3d = pts4d[:3] / pts4d[3]

    good_pts4d = (np.abs(pts4d[3, :]) > 0.01) & (pts3d[2, :] > 0)

    pts4d = np.array(pts4d[:, good_pts4d])
    
    pts1 = np.array(pts1.T[:, good_pts4d])
    pts2 = np.array(pts2.T[:, good_pts4d])

    pts3d = np.array(pts3d[:, good_pts4d])
    pts3d = -1*(np.dot(slam.Rt[:3, :3], pts3d) + -1*slam.Rt[:3, 3:4])

    optimizer_arr = [pts4d, pts3d, pts1, pts2]

    new_frame.optimizer_arr = optimizer_arr

    slam.Rt = (np.dot(slam.Rt, T)) 
    new_frame.aRt = slam.Rt

    slam.adjust()

    print(optimizer_arr[0].shape)
    print(optimizer_arr[1].shape)
    print(optimizer_arr[2].shape)
    print(optimizer_arr[3].shape)


    ptso3d = o3d.geometry.PointCloud()
    ptso3d.points = o3d.utility.Vector3dVector(pts3d.T)
    
    num_points = len(ptso3d.points)
    black_color = [0, 0, 0]  # RGB for black
    ptso3d.colors = o3d.utility.Vector3dVector([black_color] * num_points)

    new_frame.pts = ptso3d.points

    # Update the current pose by chaining the new transformation
    

    if(slam.index % 3 == 0):
        vis.add_geometry(ptso3d,False)
    
    camera_box = o3d.geometry.TriangleMesh.create_box(width=0.2, height=0.1, depth=0.2)
    camera_box.paint_uniform_color([1, 0, 0])  # Red color for the camera box
    camera_box.transform(slam.Rt)
    camera_boxes.append(camera_box)


    if(slam.index % 20 == 1):
        vis.add_geometry(camera_box,True)
    else:
        vis.add_geometry(camera_box,False)

    vis.poll_events()
    vis.update_renderer()
    print(new_frame.aRt)

    slam.curr.R = R
    slam.curr.t = t

    for i in range(len(kps1)):
        prev = (int(kps1[i].pt[0]),int(kps1[i].pt[1]))
        curr = (int(kps2[i].pt[0]),int(kps2[i].pt[1]))
        cv.circle(frame, prev, 2, color=(0,255,0))
        cv.circle(frame, curr, 2, color=(0,0,255))
        cv.line(frame,curr, prev,(0,0,255), 1)

    cv.imshow('Video', frame)
    if cv.waitKey(25) & 0xFF == ord('q'):
        break
        
vis.run()
cap.release()
cv.destroyAllWindows()