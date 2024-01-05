**Sparse Monocular Visual-SLAM Implementation**

This is my vSLAM Implementation! Given a video with a focal length, my code will generate a 3D Vizualization which includes camera poses of each frame as well as 3D point clouds of the surrounding environment.

My SLAM stack uses:
1.) Shi-Tomasi Corner Detectors
2.) Orb Descriptors
3.) knn-match based of features + lowes ratio to filter
4.) RANSAC filtering
5.) Point Triangulation with OpenCV
6.) Filtering 3D Points based off distance to camera(farther points get dropped due to noise inconsistencies)
7.) Bundle Adjustment
