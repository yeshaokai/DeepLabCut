#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#

import sys
import os
import subprocess
import deeplabcut
import json
import numpy as np 
dlc_path = os.path.dirname(deeplabcut.__file__)
root = os.path.join(dlc_path, "superanimal_pytorch")
sys.path.append(os.path.join(root, 'third_party'))


from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         process_mmdet_results, vis_pose_result)
from mmpose.datasets import DatasetInfo

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

pose_names = ['superquadruped_hrnetw32',
              'superquadruped_tokenpose']

det_names = ['superquadruped_faster_rcnn']


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

class PoseModel:
    def __init__(self, name, device):
        self.pose_config = None
        self.pose_checkpoint = None
        self.device = device
        assert name in pose_names
        if name == 'superquadruped_hrnetw32':
            self.pose_config = os.path.join(root, 'third_party', 'hrnet_pose_config_SA_quadruped.py')
            self.dataset_info_path = os.path.join(root, "third_party", "quadruped_dataset.json")            
        elif name == 'superquadruped_tokenpose':
            self.pose_config = os.path.join(root, 'third_party', 'tokenpose_pose_config_SA_quadruped.py')
            self.dataset_info_path = os.path.join(root, "third_party", "quadruped_dataset.json")                                               
    def initialize_model(self, ckpt_path):
        self.pose_checkpoint = ckpt_path        
        model = init_pose_model(self.pose_config,
                                     self.pose_checkpoint,
                                     self.device)
        
        return model
class DetModel:
    def __init__(self, name, device, det_thr = 0.3):
        self.det_config = None
        self.det_checkpoint = None
        self.device = device
        self.det_thr = det_thr
        assert name in det_names        
        if name == 'superquadruped_faster_rcnn':        
            self.det_config = os.path.join(root, 'third_party', 'det_config_SA_quadruped.py')

    def initialize_model(self, ckpt_path):
        self.det_checkpoint = ckpt_path
        model = init_detector(self.det_config,
                                   self.det_checkpoint,
                                   self.device)

        return model
class SuperAnimalPytorchInference:
    def __init__(self, det_model_name, pose_model_name,  device, det_thr = 0.3): 
        self.det_model_factory = DetModel(det_model_name, device)        
        self.pose_model_factory = PoseModel(pose_model_name, device)
        self.det_cat_id = 1
        dataset_info_path = self.pose_model_factory.dataset_info_path
        
        with open(dataset_info_path, 'r') as f:
            dataset_info = json.load(f)['dataset_info']

            self.dataset_info = DatasetInfo(dataset_info)
            self.keypoint_names = [v['name'] for v in self.dataset_info.keypoint_info.values()]
            print (self.keypoint_names)
    def initialize_model(self, det_ckpt, pose_ckpt):

        self.det_model = self.det_model_factory.initialize_model(det_ckpt)        
        self.pose_model = self.pose_model_factory.initialize_model(pose_ckpt)
        self.dataset = self.pose_model.cfg.data['test']['type']
        
    def predict_frame(self, image_paths):
        ret = {}
        ret['joint_names'] = self.keypoint_names
        ret['predictions'] = {}
        for image_path in image_paths:
            mmdet_results = inference_detector(self.det_model, image_path)
            person_results = process_mmdet_results(mmdet_results, self.det_cat_id)
            pose_results, returned_outputs = inference_top_down_pose_model(
                self.pose_model,
                image_path,
                person_results,
                bbox_thr=self.det_model_factory.det_thr,
                format='xyxy',
                dataset=self.dataset,
                dataset_info=self.dataset_info,
                return_heatmap=False,
                outputs=None)
            ret['predictions'][image_path] = {}
            ret['predictions'][image_path]['bboxes']  = []
            ret['predictions'][image_path]['markers']  = []            
            for pred in pose_results:
                ret['predictions'][image_path]['bboxes'].append(pred['bbox'])
                ret['predictions'][image_path]['markers'].append(pred['keypoints'].reshape(-1))
        return ret

if __name__ == '__main__':

    from argparse import ArgumentParser    

    parser = ArgumentParser()
    
    parser.add_argument('det_model_name', help = 'det model name')
    parser.add_argument('pose_model_name', help = 'pose model name')
    
    parser.add_argument('det_checkpoint', help = 'path to checkpoint of the detector')

    parser.add_argument('pose_checkpoint', help = 'path to the checkpoint of the pose estimator')

    parser.add_argument('image_path', help ='path to the image')
    
    parser.add_argument('--device', type = str, default = 'cuda')

    parser.add_argument('--out', type = str, help = 'output path (json) of the prediction', default = 'temp.json')


    args = parser.parse_args()
            
    inference_obj = SuperAnimalPytorchInference(args.det_model_name,
                                                args.pose_model_name,
                                                args.device)

    inference_obj.initialize_model(args.det_checkpoint, args.pose_checkpoint)

    image_path = [args.image_path]
        
    res = inference_obj.predict_frame(image_path)
    
    with open(args.out, 'w') as f:
        json.dump(res, f, indent = 4, cls = NpEncoder)