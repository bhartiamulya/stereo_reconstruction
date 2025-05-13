import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class StereoReconstructor:
    def __init__(self):
        # Increase features significantly
        self.sift = cv2.SIFT_create(nfeatures=10000)  # Much more features
        self.matcher = cv2.BFMatcher()

    def load_images(self, left_path, right_path):
        self.left_img = cv2.imread(left_path)
        self.right_img = cv2.imread(right_path)
        if self.left_img is None or self.right_img is None:
            raise ValueError("Could not load images")
        # Use larger image size
        self.left_img = cv2.resize(self.left_img, (self.left_img.shape[1] * 2 // 3, self.left_img.shape[0] * 2 // 3))
        self.right_img = cv2.resize(self.right_img, (self.right_img.shape[1] * 2 // 3, self.right_img.shape[0] * 2 // 3))
        print(f"✓ Images loaded and resized to {self.left_img.shape[1]}x{self.left_img.shape[0]}")
        return self.left_img, self.right_img

    def find_matches(self):
        left_gray = cv2.cvtColor(self.left_img, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(self.right_img, cv2.COLOR_BGR2GRAY)
        
        # Enhance contrast
        left_gray = cv2.equalizeHist(left_gray)
        right_gray = cv2.equalizeHist(right_gray)
        
        kp1, des1 = self.sift.detectAndCompute(left_gray, None)
        kp2, des2 = self.sift.detectAndCompute(right_gray, None)
        
        if des1 is None or des2 is None or len(des1) < 8 or len(des2) < 8:
            raise ValueError("Not enough features detected")
            
        matches = self.matcher.knnMatch(des1, des2, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.8 * n.distance:  # More relaxed ratio test
                good_matches.append(m)
        
        self.matches = sorted(good_matches, key=lambda x: x.distance)[:1000]  # Keep many more matches
        self.kp1, self.kp2 = kp1, kp2
        print(f"✓ Found {len(self.matches)} good matches")
        return self.matches

    def compute_homography_ransac(self):
        src_pts = np.float32([self.kp1[m.queryIdx].pt for m in self.matches])
        dst_pts = np.float32([self.kp2[m.trainIdx].pt for m in self.matches])
        
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)  # More permissive threshold
        
        if H is None:
            raise ValueError("Could not compute homography")
            
        self.inliers = [m for i, m in enumerate(self.matches) if mask[i]]
        self.src_pts = src_pts[mask.ravel() == 1]
        self.dst_pts = dst_pts[mask.ravel() == 1]
        print(f"✓ Computed homography with {len(self.inliers)} inliers")
        return H, mask

    def triangulate_points(self):
        pts1 = np.float32([self.kp1[m.queryIdx].pt for m in self.matches])
        pts2 = np.float32([self.kp2[m.trainIdx].pt for m in self.matches])
        
        # Store original points for color mapping
        self.orig_pts = pts1.copy()
        
        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 2.0)
        if F is None:
            raise ValueError("Could not compute fundamental matrix")
            
        pts1 = pts1[mask.ravel() == 1]
        pts2 = pts2[mask.ravel() == 1]
        self.orig_pts = self.orig_pts[mask.ravel() == 1]  # Update original points

        height, width = self.left_img.shape[:2]
        focal_length = max(height, width)
        principal_point = (width // 2, height // 2)
        K = np.array([
            [focal_length, 0, principal_point[0]],
            [0, focal_length, principal_point[1]],
            [0, 0, 1]
        ])

        E = K.T @ F @ K
        _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)

        P1 = K @ np.hstack((np.eye(3), np.zeros((3,1))))
        P2 = K @ np.hstack((R, t))

        points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        points_3d = points_4d[:3] / points_4d[3]
        points_3d = points_3d.T

        # More permissive filtering
        valid_points = (points_3d[:, 2] > 0) & (np.abs(points_3d) < 1000).all(axis=1)
        points_3d = points_3d[valid_points]
        self.orig_pts = self.orig_pts[valid_points]  # Update original points

        # Remove statistical outliers
        mean = np.mean(points_3d, axis=0)
        std = np.std(points_3d, axis=0)
        inliers = np.all(np.abs(points_3d - mean) < 3 * std, axis=1)
        points_3d = points_3d[inliers]
        self.orig_pts = self.orig_pts[inliers]  # Update original points

        # Normalize scale
        scale = np.percentile(np.abs(points_3d), 95)
        points_3d = points_3d / scale

        # Get colors from original image
        self.point_colors = []
        for pt in self.orig_pts:
            x, y = int(pt[0]), int(pt[1])
            x = min(max(x, 0), self.left_img.shape[1] - 1)
            y = min(max(y, 0), self.left_img.shape[0] - 1)
            color = self.left_img[y, x] / 255.0  # Normalize color values
            self.point_colors.append(color)
        self.point_colors = np.array(self.point_colors)

        print(f"✓ Generated {len(points_3d)} 3D points with colors")
        return points_3d

    def visualize_matches_and_ransac(self):
        plt.figure(figsize=(15, 6))
        
        match_img = cv2.drawMatches(self.left_img, self.kp1, 
                                  self.right_img, self.kp2, 
                                  self.inliers, None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        plt.subplot(121)
        plt.imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
        plt.title(f'RANSAC Inliers ({len(self.inliers)} matches)')
        plt.axis('off')
        
        plt.subplot(122)
        plt.scatter(self.src_pts[:, 0], self.src_pts[:, 1], c='b', s=20, label='Source')
        plt.scatter(self.dst_pts[:, 0], self.dst_pts[:, 1], c='r', s=20, label='Target')
        
        for src, dst in zip(self.src_pts[:100], self.dst_pts[:100]):  # Show more mapping lines
            plt.plot([src[0], dst[0]], [src[1], dst[1]], 'g-', alpha=0.3)
            
        plt.title('RANSAC Point Mapping')
        plt.legend()
        plt.axis('equal')
        plt.tight_layout()
        print("✓ Visualized RANSAC matches and mapping")
        plt.draw()
        plt.pause(0.1)

    def visualize_3d(self, points_3d):
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot textured point cloud using original image colors
        scatter = ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2],
                           c=self.point_colors, 
                           s=50, alpha=0.8)  # Larger points to see colors better
        
        # Add reference cube
        r = 1.0
        cube_points = np.array([
            [-r, -r, -r], [r, -r, -r], [r, r, -r], [-r, r, -r],
            [-r, -r, r], [r, -r, r], [r, r, r], [-r, r, r]
        ])
        
        edges = [
            (0,1), (1,2), (2,3), (3,0),
            (4,5), (5,6), (6,7), (7,4),
            (0,4), (1,5), (2,6), (3,7)
        ]
        
        for start, end in edges:
            ax.plot3D([cube_points[start,0], cube_points[end,0]],
                     [cube_points[start,1], cube_points[end,1]],
                     [cube_points[start,2], cube_points[end,2]], 
                     'r-', alpha=0.3, linewidth=2)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.title('3D Reconstruction with Original Image Colors')
        
        # Multiple views for better visualization
        views = [(30, 45), (0, 0), (0, 90), (90, 0)]
        for i, (elev, azim) in enumerate(views):
            ax.view_init(elev=elev, azim=azim)
            plt.draw()
            plt.pause(1.0)
            
        ax.view_init(elev=30, azim=45)
        ax.set_box_aspect([1,1,1])
        
        print("✓ Generated 3D visualization with image colors")
        plt.draw()
        plt.pause(0.1)

def main():
    try:
        print("\n=== Starting Stereo Reconstruction ===")
        reconstructor = StereoReconstructor()
        
        print("\n1. Loading images...")
        reconstructor.load_images('left.jpg', 'right.jpg')
        
        print("\n2. Finding matches...")
        reconstructor.find_matches()
        
        print("\n3. Computing homography and visualizing RANSAC...")
        H, mask = reconstructor.compute_homography_ransac()
        reconstructor.visualize_matches_and_ransac()
        
        print("\n4. Computing and visualizing 3D points...")
        points_3d = reconstructor.triangulate_points()
        reconstructor.visualize_3d(points_3d)
        
        print("\n✓ Reconstruction completed successfully!")
        print("The 3D visualization will show multiple views automatically.")
        print("Close the plot windows to exit.")
        
        plt.show()
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")

if __name__ == "__main__":
    main()
