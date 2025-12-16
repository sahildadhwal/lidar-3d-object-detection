# lidar-3d-object-detection

# Multi-Task 3D Perception Pipeline

End-to-end 3D perception system integrating object detection, tracking, and semantic segmentation on KITTI dataset.

## Tasks
- **3D Object Detection**: Detect cars, pedestrians, cyclists
- **Multi-Object Tracking**: Track objects across frames
- **Semantic Segmentation**: Label point cloud (road, vehicle, etc.)


## Tech Stack
- PyTorch
- OpenPCDet (3D detection)
- Python 3.9

<!-- 
# Create environment with Python 3.9
conda create -n waymo3d python=3.9 -y

# Activate it
conda activate waymo3d

# Install core packages
pip install torch torchvision torchaudio
pip install numpy pandas matplotlib jupyter
pip install opencv-python

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')" 


-->
