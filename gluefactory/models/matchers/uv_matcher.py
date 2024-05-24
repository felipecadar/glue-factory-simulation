"""
Nearest neighbor matcher for normalized descriptors.
Optionally apply the mutual check and threshold the distance or ratio.
"""

import logging

import torch
import torch.nn.functional as F

from ..base_model import BaseModel

from scipy.spatial import KDTree
import numpy as np

def warp_keypoints(
        keypoints1: np.ndarray,
        coords1: np.ndarray,
        coords2: np.ndarray,
        segmentation1: np.ndarray,
        segmentation2: np.ndarray,
        threshold: float = 300,
):
    '''
    coords1: (H, W, 3) # Object coordinates
    coords2: (H, W, 3) # Object coordinates
    segmentation1: (H, W) # instance segmentation
    segmentation2: (H, W) # instance segmentation
    keypoints1: (N, 2) # keypoints coordinates in the first image
    threshold: float # distance threshold to consider a correspondence    

    return:
        keypoints2: (N, 2) # GT keypoint coordinates.
        distances: (N,) # to measure how reliable is the GT. Lower is better
    '''
    H, W, _ = coords1.shape
    instances1 = np.unique(segmentation1.reshape(-1), axis=0)
    instances2 = np.unique(segmentation2.reshape(-1), axis=0)
    shared_instances = list(set(instances1).intersection(set(instances2)))
    # shared_instances = list(filter(lambda x: x > 1, shared_instances))

    # from int to float
    coords1 = coords1.astype(np.float32)
    coords2 = coords2.astype(np.float32)

    kps_gt = np.zeros_like(keypoints1) - 1
    gt_dist = np.zeros(len(keypoints1)) + np.inf

    for instance_id in shared_instances:
        # print(f"Instance {instance_id}")
        seg_mask1 = (segmentation1 == instance_id).astype(np.uint8)
        seg_mask2 = (segmentation2 == instance_id).astype(np.uint8)
        coords1_masked = coords1 * seg_mask1[:, :, None]
        coords2_masked = coords2 * seg_mask2[:, :, None]

        tree2 = KDTree(coords2_masked.reshape(-1, 3))

        # if seg_mask1.sum() == 0 or seg_mask2.sum() == 0:
        #     continue
        
        kps_count = 0
        for kp_idx in range(len(keypoints1)):
            # eval just the keypoints that belong to the same instance
            # to avoid wrong correspondences
            source_kps = keypoints1[kp_idx]

            if not seg_mask1[int(source_kps[1]), int(source_kps[0])]:
                continue

            kps_count += 1
            NOCS1 = coords1_masked[int(source_kps[1]), int(source_kps[0])]
            dists2, indexes2 = tree2.query(NOCS1, k=1)

            if dists2 > threshold:
                continue
            else:
                # find coordinates of the NOCS2 in the image
                y2, x2 = np.unravel_index(indexes2, (H, W))
                assert np.allclose(coords2_masked[y2, x2], coords2_masked.reshape(-1, 3)[indexes2])


            kps_gt[kp_idx] = [x2, y2]
            gt_dist[kp_idx] = dists2
        # print(f"Instance {instance_id}: {kps_count} keypoints")
    kps_gt = kps_gt.astype(float)

    return {
        'keypoints': kps_gt,
        'distances': gt_dist,
    }

class UVMatcher(BaseModel):
    default_conf = {
        'gt_thresh': 300,
        'pixel_thresh': 2,
    }
    required_data_keys = ["keypoints0", "keypoints1", "segmentation0", "segmentation1", "uv_coords0", "uv_coords1"]

    def _init(self, conf):
        pass
        
    def _forward(self, data):
        device = data["keypoints0"].device

        keypoints0 = data["keypoints0"]
        keypoints1 = data["keypoints1"]
        uv_coords0 = data["uv_coords0"]
        uv_coords1 = data["uv_coords1"]
        segmentation0 = data["segmentation0"]
        segmentation1 = data["segmentation1"]
        
        matches0 = []
        matches1 = []
        assignment = []
        
        for b in range(keypoints0.shape[0]):
            gt_0to1 = warp_keypoints(
                keypoints0[b].detach().cpu().numpy(),
                uv_coords0[b].detach().cpu().numpy(),
                uv_coords1[b].detach().cpu().numpy(),
                segmentation0[b].detach().cpu().numpy(),
                segmentation1[b].detach().cpu().numpy(),
                threshold=self.conf.gt_thresh,
            )['keypoints']

            gt_1to0 = warp_keypoints(
                keypoints1[b].detach().cpu().numpy(),
                uv_coords1[b].detach().cpu().numpy(),
                uv_coords0[b].detach().cpu().numpy(),
                segmentation1[b].detach().cpu().numpy(),
                segmentation0[b].detach().cpu().numpy(),
                threshold=self.conf.gt_thresh,
            )['keypoints']
            
            gt_0to1 = torch.tensor(gt_0to1, device=device).to(keypoints0.dtype)
            gt_1to0 = torch.tensor(gt_1to0, device=device).to(keypoints0.dtype)
            
            distmat0 = torch.cdist(keypoints1[b], gt_0to1)
            distmat1 = torch.cdist(keypoints0[b], gt_1to0)
            
            m0_scores, m0 = distmat0.min(dim=0)
            m1_scores, m1 = distmat1.min(dim=0)
            
            valid0 = m0_scores < self.conf.pixel_thresh
            valid1 = m1_scores < self.conf.pixel_thresh
            
            m0[~valid0] = -1
            m1[~valid1] = -1
            
            assignment_mat = torch.zeros((keypoints0.shape[1], keypoints1.shape[1]), device=device)
            assignment_mat[m1, m0] = 1
            
            matches0.append(m0)
            matches1.append(m1)
            assignment.append(assignment_mat)
            
        matches0 = torch.stack(matches0, dim=0)
        matches1 = torch.stack(matches1, dim=0)
        assignment = torch.stack(assignment, dim=0)
        
        return {
            "matches0": matches0,
            "matches1": matches1,
            "assignment": assignment,
        }

    def loss(self, pred, data):
        raise NotImplementedError
