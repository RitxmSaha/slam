import cv2 as cv
import open3d as o3d
import g2o
import numpy as np
import features
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
        self.Rt = np.eye(4)

    def update_Rt(self, new_Rt):
        self.Rt = new_Rt

    def get_Rt(self):
        return np.copy(self.Rt)

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
slam.addFrame(new_frame)


while cap.isOpened():

    
    #print(slam.index)
    ret, frame = cap.read()
    kps, des = fe.findKeypoints(frame)
    new_frame = Frame(kps, des)
    slam.addFrame(new_frame)

    #print(slam.Rt)
    
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

    T = np.eye(4)  # Initialize as 4x4 identity matrix
    T[:3, :3] = R  # Set the upper-left 3x3 block as the rotation matrix
    T[:3, 3] = t.T  # Set the upper-right 3x1 block as the translation vector

    P1 = np.dot(slam.K, np.eye(3, 4))  # Projection matrix for the first camera frame
    P2 = np.dot(slam.K, np.concatenate((R, t), axis=1))  # Projection matrix for the second frame

    # Triangulate points
    points_4d_hom = cv.triangulatePoints(P1, P2, pts1.T, pts2.T)


    points_3d = points_4d_hom[:3] / points_4d_hom[3]

    points_3d = points_3d[:, points_3d[2] > 0]

    Rt = slam.get_Rt()
    points_3d = np.dot(Rt[:3, :3], points_3d) + Rt[:3, 3:4]


    points_3d_o3d = o3d.geometry.PointCloud()
    points_3d_o3d.points = o3d.utility.Vector3dVector(points_3d.T)
    print(points_3d_o3d.points[0])
    
    num_points = len(points_3d_o3d.points)
    black_color = [0, 0, 0]  # RGB for black
    points_3d_o3d.colors = o3d.utility.Vector3dVector([black_color] * num_points)

    vis.add_geometry(points_3d_o3d,False)

    # Update the current pose by chaining the new transformation
    slam.update_Rt(np.dot(slam.Rt, T)) # Matrix multiplication
    Rt = slam.get_Rt()
    camera_box = o3d.geometry.TriangleMesh.create_box(width=0.2, height=0.1, depth=0.2)
    camera_box.paint_uniform_color([1, 0, 0])  # Red color for the camera box
    camera_box.transform(Rt)
    camera_boxes.append(camera_box)


    if(slam.index < 100):
        vis.add_geometry(camera_box,True)
    else:
        vis.add_geometry(camera_box,False)

    vis.poll_events()
    vis.update_renderer()

    slam.curr.R = R
    slam.curr.t = t

    for i in range(len(kps1)):
        prev = (int(kps1[i].pt[0]),int(kps1[i].pt[1]))
        curr = (int(kps2[i].pt[0]),int(kps2[i].pt[1]))
        cv.circle(frame, prev, 2, color=(0,255,0))
        cv.circle(frame, curr, 2, color=(0,0,255))
        cv.line(frame,curr, prev,(0,0,255), 1)




    #o3d.visualization.draw_geometries([points_3d_o3d])

    cv.imshow('Video', frame)
    if cv.waitKey(25) & 0xFF == ord('q'):
        break
        
    if(slam.index == 100):
        vis.run()
cap.release()
cv.destroyAllWindows()

# def create_optimizer():
#     optimizer = g2o.SparseOptimizer()
#     solver = g2o.BlockSolverX(g2o.LinearSolverEigenX())
#     solver = g2o.OptimizationAlgorithmLevenberg(solver)
#     optimizer.set_algorithm(solver)
#     return optimizer


# optimizer = create_optimizer()