# Stereo Image 3D Reconstruction

This project performs 3D reconstruction from stereo images using computer vision techniques. It implements feature matching, homography computation, RANSAC-based outlier removal, and 3D visualization.

## Features
- SIFT feature detection and matching
- Homography computation with RANSAC
- Outlier removal
- 3D point triangulation
- Interactive 3D visualization
- Match visualization

## Setup
1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage
1. Place your stereo image pair in the project directory:
   - Name them `left.jpg` and `right.jpg`
   - Images should be taken from slightly different viewpoints of the same scene

2. Run the reconstruction:
```bash
python stereo_reconstruction.py
```

The program will:
1. Detect and match features between the images
2. Remove outliers using RANSAC
3. Show the matched features visualization
4. Display the 3D reconstruction

## Technical Details
- Uses SIFT (Scale-Invariant Feature Transform) for feature detection
- Implements RANSAC for robust homography estimation
- Performs triangulation for 3D point reconstruction
- Uses matplotlib for 3D visualization

## Requirements
- Python 3.7+
- OpenCV
- NumPy
- Matplotlib
- SciPy
