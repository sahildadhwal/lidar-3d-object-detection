"""
Waymo Open Dataset v2.0 loader for 3D object detection.
Loads LiDAR point clouds and 3D bounding boxes from Parquet files.
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from torch.utils.data import Dataset
import torch


class WaymoDataset(Dataset):
    """Waymo Open Dataset v2.0 for 3D object detection."""
    
    # Class names
    CLASS_NAMES = ['Vehicle', 'Pedestrian', 'Cyclist']
    
    # Waymo type ID to class index mapping
    # Type 1=Vehicle, 2=Pedestrian, 4=Cyclist
    TYPE_TO_CLASS = {1: 0, 2: 1, 4: 2}
    
    # Point cloud range [x_min, y_min, z_min, x_max, y_max, z_max]
    POINT_CLOUD_RANGE = [-75.2, -75.2, -2, 75.2, 75.2, 4]
    
    def __init__(self, data_dir, augment=False):
        """
        Args:
            data_dir: Path to directory containing lidar/ and lidar_box/ folders
            augment: Whether to apply data augmentation (for training)
        """
        self.data_dir = Path(data_dir)
        self.augment = augment
        
        # Find all lidar files
        self.lidar_dir = self.data_dir / 'lidar'
        self.label_dir = self.data_dir / 'lidar_box'
        
        self.lidar_files = sorted(list(self.lidar_dir.glob('*.parquet')))
        
        print(f"Found {len(self.lidar_files)} segments")
        
        # Load all segments into memory (small dataset, ~5GB total)
        self.frames = []
        self._load_all_data()
    
    def _load_all_data(self):
        """Load all parquet files and split into individual frames."""
        print("Loading data into memory...")
        
        for lidar_file in self.lidar_files:
            segment_name = lidar_file.stem
            label_file = self.label_dir / f"{segment_name}.parquet"
            
            # Load LiDAR points
            lidar_df = pd.read_parquet(lidar_file)
            
            # Load labels
            label_df = pd.read_parquet(label_file)
            
            # Group by frame (timestamp)
            frame_groups = lidar_df.groupby('key.frame_timestamp_micros')
            
            for timestamp, frame_points in frame_groups:
                # Get labels for this frame
                frame_labels = label_df[
                    label_df['key.frame_timestamp_micros'] == timestamp
                ]
                
                self.frames.append({
                    'segment': segment_name,
                    'timestamp': timestamp,
                    'points': frame_points,
                    'labels': frame_labels
                })
        
        print(f"Loaded {len(self.frames)} frames total")
    
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        frame_data = self.frames[idx]
        
        # Extract point cloud (x, y, z, intensity)
        points_df = frame_data['points']
        points = np.column_stack([
            points_df['lidar.x'].values,
            points_df['lidar.y'].values,
            points_df['lidar.z'].values,
            points_df['lidar.intensity'].values
        ]).astype(np.float32)
        
        # Filter points to range
        mask = self._filter_points_to_range(points)
        points = points[mask]
        
        # Extract 3D bounding boxes
        labels_df = frame_data['labels']
        gt_boxes, gt_classes = self._extract_boxes(labels_df)
        
        # Data augmentation (training only)
        if self.augment:
            points, gt_boxes = self._augment(points, gt_boxes)
        
        return {
            'frame_id': f"{frame_data['segment']}_{frame_data['timestamp']}",
            'points': points,
            'gt_boxes': gt_boxes,
            'gt_classes': gt_classes,
        }
    
    def _filter_points_to_range(self, points):
        """Filter points to be within detection range."""
        pc_range = np.array(self.POINT_CLOUD_RANGE)
        mask = (
            (points[:, 0] >= pc_range[0]) & (points[:, 0] <= pc_range[3]) &
            (points[:, 1] >= pc_range[1]) & (points[:, 1] <= pc_range[4]) &
            (points[:, 2] >= pc_range[2]) & (points[:, 2] <= pc_range[5])
        )
        return mask
    
    def _extract_boxes(self, labels_df):
        """Convert label dataframe to box arrays."""
        if len(labels_df) == 0:
            return np.zeros((0, 7), dtype=np.float32), np.zeros((0,), dtype=np.int64)
        
        gt_boxes = []
        gt_classes = []
        
        for _, label in labels_df.iterrows():
            # Only keep Vehicle, Pedestrian, Cyclist
            label_type = label['lidar_box.type']
            if label_type not in self.TYPE_TO_CLASS:
                continue
            
            # Box format: [x, y, z, length, width, height, heading]
            box = np.array([
                label['lidar_box.center.x'],
                label['lidar_box.center.y'],
                label['lidar_box.center.z'],
                label['lidar_box.size.x'],  # length
                label['lidar_box.size.y'],  # width
                label['lidar_box.size.z'],  # height
                label['lidar_box.heading']
            ], dtype=np.float32)
            
            gt_boxes.append(box)
            gt_classes.append(self.TYPE_TO_CLASS[label_type])
        
        if len(gt_boxes) == 0:
            return np.zeros((0, 7), dtype=np.float32), np.zeros((0,), dtype=np.int64)
        
        return np.array(gt_boxes, dtype=np.float32), np.array(gt_classes, dtype=np.int64)
    
    def _augment(self, points, gt_boxes):
        """Apply data augmentation."""
        # Random flip along Y axis
        if np.random.rand() > 0.5:
            points[:, 1] = -points[:, 1]
            if len(gt_boxes) > 0:
                gt_boxes[:, 1] = -gt_boxes[:, 1]
                gt_boxes[:, 6] = -gt_boxes[:, 6]  # Flip heading
        
        # Random rotation around Z axis (-45 to +45 degrees)
        rotation = np.random.uniform(-np.pi/4, np.pi/4)
        cos_r, sin_r = np.cos(rotation), np.sin(rotation)
        rotation_matrix = np.array([
            [cos_r, -sin_r, 0],
            [sin_r, cos_r, 0],
            [0, 0, 1]
        ])
        points[:, :3] = points[:, :3] @ rotation_matrix.T
        if len(gt_boxes) > 0:
            gt_boxes[:, :3] = gt_boxes[:, :3] @ rotation_matrix.T
            gt_boxes[:, 6] += rotation
        
        # Random scaling (0.95 to 1.05)
        scale = np.random.uniform(0.95, 1.05)
        points[:, :3] *= scale
        if len(gt_boxes) > 0:
            gt_boxes[:, :6] *= scale
        
        return points, gt_boxes
    
    @staticmethod
    def collate_fn(batch):
        """Custom collate function for batching."""
        frame_ids = [item['frame_id'] for item in batch]
        
        # Merge points with batch index
        points_list = []
        for i, item in enumerate(batch):
            points = item['points']
            batch_idx = np.full((points.shape[0], 1), i, dtype=np.float32)
            points_with_batch = np.concatenate([batch_idx, points], axis=1)
            points_list.append(points_with_batch)
        
        points = np.concatenate(points_list, axis=0)
        
        # Pad gt_boxes to same length
        max_boxes = max(item['gt_boxes'].shape[0] for item in batch)
        gt_boxes = np.zeros((len(batch), max_boxes, 7), dtype=np.float32)
        gt_classes = np.zeros((len(batch), max_boxes), dtype=np.int64)
        
        for i, item in enumerate(batch):
            n = item['gt_boxes'].shape[0]
            if n > 0:
                gt_boxes[i, :n] = item['gt_boxes']
                gt_classes[i, :n] = item['gt_classes']
        
        return {
            'frame_ids': frame_ids,
            'points': torch.from_numpy(points),
            'gt_boxes': torch.from_numpy(gt_boxes),
            'gt_classes': torch.from_numpy(gt_classes),
        }


# Test the dataset
if __name__ == '__main__':
    # Test locally
    dataset = WaymoDataset(
        data_dir='data/raw',
        augment=True
    )
    
    print(f"\nDataset size: {len(dataset)} frames")
    
    # Load one sample
    sample = dataset[0]
    print(f"\nSample 0:")
    print(f"  Frame ID: {sample['frame_id']}")
    print(f"  Points shape: {sample['points'].shape}")
    print(f"  GT boxes shape: {sample['gt_boxes'].shape}")
    print(f"  GT classes: {sample['gt_classes']}")
    
    # Test dataloader
    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=WaymoDataset.collate_fn,
        num_workers=0  # 0 for testing
    )
    
    batch = next(iter(dataloader))
    print(f"\nBatch:")
    print(f"  Points shape: {batch['points'].shape}")
    print(f"  GT boxes shape: {batch['gt_boxes'].shape}")
    print(f"  Frame IDs: {batch['frame_ids']}")