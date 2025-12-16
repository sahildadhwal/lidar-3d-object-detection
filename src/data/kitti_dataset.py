"""
KITTI dataset loader for 3D object detection.
Handles velodyne point clouds and 3D bounding box labels.
"""

import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset


class KITTIDataset(Dataset):
    """KITTI 3D Object Detection Dataset"""
    
    CLASS_NAMES = ['Car', 'Pedestrian', 'Cyclist']
    
    def __init__(self, data_dir, split='training', augment=False):
        """
        Args:
            data_dir: Path to KITTI root directory
            split: 'training' or 'testing'
            augment: Apply data augmentation
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.augment = augment
        
        self.velodyne_dir = self.data_dir / split / 'velodyne'
        self.label_dir = self.data_dir / split / 'label_2'
        self.calib_dir = self.data_dir / split / 'calib'
        
        # Get all sample IDs
        self.sample_ids = [f.stem for f in sorted(self.velodyne_dir.glob('*.bin'))]
        
        print(f"Loaded {len(self.sample_ids)} samples from {split} set")
    
    def __len__(self):
        return len(self.sample_ids)
    
    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        
        # Load point cloud
        points = self.load_velodyne(sample_id)
        
        # Load labels
        if self.split == 'training':
            labels = self.load_labels(sample_id)
            calib = self.load_calib(sample_id)
            
            # Convert labels to LiDAR coordinate system
            gt_boxes, gt_classes = self.labels_to_boxes(labels, calib)
        else:
            gt_boxes = np.zeros((0, 7), dtype=np.float32)
            gt_classes = np.zeros((0,), dtype=np.int64)
        
        return {
            'sample_id': sample_id,
            'points': points,
            'gt_boxes': gt_boxes,
            'gt_classes': gt_classes
        }
    
    def load_velodyne(self, sample_id):
        """Load velodyne point cloud"""
        velo_file = self.velodyne_dir / f'{sample_id}.bin'
        points = np.fromfile(velo_file, dtype=np.float32).reshape(-1, 4)
        return points
    
    def load_labels(self, sample_id):
        """Load 3D bounding box labels"""
        label_file = self.label_dir / f'{sample_id}.txt'
        
        if not label_file.exists():
            return []
        
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        labels = []
        for line in lines:
            parts = line.strip().split(' ')
            
            obj_type = parts[0]
            if obj_type not in self.CLASS_NAMES:
                continue
            
            # Parse label (KITTI format)
            label = {
                'type': obj_type,
                'truncated': float(parts[1]),
                'occluded': int(parts[2]),
                'alpha': float(parts[3]),
                'bbox': [float(x) for x in parts[4:8]],  # 2D bbox
                'dimensions': [float(x) for x in parts[8:11]],  # h, w, l
                'location': [float(x) for x in parts[11:14]],  # x, y, z in camera
                'rotation_y': float(parts[14])
            }
            labels.append(label)
        
        return labels
    
    def load_calib(self, sample_id):
        """Load calibration matrices"""
        calib_file = self.calib_dir / f'{sample_id}.txt'
        
        calib = {}
        with open(calib_file, 'r') as f:
            for line in f.readlines():
                if ':' not in line:
                    continue
                key, value = line.split(':', 1)
                calib[key] = np.array([float(x) for x in value.split()])
        
        # Reshape matrices
        calib['P2'] = calib['P2'].reshape(3, 4)
        calib['R0_rect'] = calib['R0_rect'].reshape(3, 3)
        calib['Tr_velo_to_cam'] = calib['Tr_velo_to_cam'].reshape(3, 4)
        
        return calib
    
    def labels_to_boxes(self, labels, calib):
        """Convert camera coordinate labels to LiDAR boxes"""
        if len(labels) == 0:
            return np.zeros((0, 7), dtype=np.float32), np.zeros((0,), dtype=np.int64)
        
        gt_boxes = []
        gt_classes = []
        
        for label in labels:
            # Get rotation matrix from camera to LiDAR
            R0 = np.eye(4)
            R0[:3, :3] = calib['R0_rect']
            
            Tr_velo_to_cam = np.eye(4)
            Tr_velo_to_cam[:3, :4] = calib['Tr_velo_to_cam']
            
            # Transform: camera -> rect -> velo
            # (Simplified - full implementation needs proper transformation)
            
            h, w, l = label['dimensions']
            x, y, z = label['location']
            ry = label['rotation_y']
            
            # Box format: [x, y, z, l, w, h, heading]
            # Note: KITTI uses camera coordinates, need transformation for real use
            box = np.array([x, y, z, l, w, h, ry], dtype=np.float32)
            gt_boxes.append(box)
            
            class_idx = self.CLASS_NAMES.index(label['type'])
            gt_classes.append(class_idx)
        
        return np.array(gt_boxes, dtype=np.float32), np.array(gt_classes, dtype=np.int64)
    
    @staticmethod
    def collate_fn(batch):
        """Collate function for DataLoader"""
        sample_ids = [item['sample_id'] for item in batch]
        
        # Stack points with batch index
        points_list = []
        for i, item in enumerate(batch):
            points = item['points']
            batch_idx = np.full((points.shape[0], 1), i, dtype=np.float32)
            points_with_batch = np.concatenate([batch_idx, points], axis=1)
            points_list.append(points_with_batch)
        
        points = np.concatenate(points_list, axis=0)
        
        # Pad boxes
        max_boxes = max(item['gt_boxes'].shape[0] for item in batch)
        gt_boxes = np.zeros((len(batch), max_boxes, 7), dtype=np.float32)
        gt_classes = np.zeros((len(batch), max_boxes), dtype=np.int64)
        
        for i, item in enumerate(batch):
            n = item['gt_boxes'].shape[0]
            if n > 0:
                gt_boxes[i, :n] = item['gt_boxes']
                gt_classes[i, :n] = item['gt_classes']
        
        return {
            'sample_ids': sample_ids,
            'points': torch.from_numpy(points),
            'gt_boxes': torch.from_numpy(gt_boxes),
            'gt_classes': torch.from_numpy(gt_classes)
        }


if __name__ == '__main__':
    # Test the dataset
    dataset = KITTIDataset(data_dir='data/kitti', split='training')
    
    print(f"Dataset size: {len(dataset)}")
    
    # Load first sample
    sample = dataset[0]
    print(f"\nSample 0:")
    print(f"  Points: {sample['points'].shape}")
    print(f"  Boxes: {sample['gt_boxes'].shape}")
    print(f"  Classes: {sample['gt_classes']}")