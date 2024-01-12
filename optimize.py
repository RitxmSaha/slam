import g2o
import numpy as np

class BundleAdjustment:
    def __init__(self):
        # Initialize the optimizer
        self.optimizer = g2o.SparseOptimizer()
        solver = g2o.BlockSolverX(g2o.LinearSolverPCGX())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        self.optimizer.set_algorithm(solver)

    def optimize(self, frames):
        # Add vertices (camera poses and 3D points) to the graph

        # Add camera pose vertices
        for i, frame in enumerate(frames):
            # Convert the relative pose (R, t) to a global pose for the optimizer
            if i == 0:
                global_pose = np.eye(4)  # First frame as the world origin
            else:
                rel_pose = np.eye(4)
                rel_pose[:3, :3] = frame.R
                rel_pose[:3, 3] = frame.t.flatten()
                global_pose = np.dot(frames[i-1].aRt, rel_pose)

            pose = g2o.SE3Quat(global_pose[:3, :3], global_pose[:3, 3])
            v_se3 = g2o.VertexSE3Expmap()
            v_se3.set_id(i)
            v_se3.set_estimate(pose)
            v_se3.set_fixed(i == 0)  # Fix the first frame for gauge freedom
            self.optimizer.add_vertex(v_se3)

        # (Continue with adding 3D point vertices and edges as in the previous example)

        # Perform optimization
        self.optimizer.initialize_optimization()
        self.optimizer.optimize(10)  # number of iterations

        # Update the frames with the optimized global poses
        for i, frame in enumerate(frames):
            pose = self.optimizer.vertex(i).estimate()
            frame.aRt = np.eye(4)
            frame.aRt[:3, :3] = pose.rotation().matrix()
            frame.aRt[:3, 3] = pose.translation()