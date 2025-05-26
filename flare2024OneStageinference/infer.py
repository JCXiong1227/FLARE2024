import os
import torch
import time
import glob
import argparse
import numpy as np
from pathlib import Path
from models.phase2_newnetwork import SMT
from collections import OrderedDict
from monai.inferers import sliding_window_inference
from inferencetransformer import image_transform, infer_post_transform
from monai.transforms import Compose, SaveImaged 


'''set hyperparameters'''
parser = argparse.ArgumentParser()
parser.add_argument('--numclasses', type=int, default=14, help='numclasses')
parser.add_argument('--inputs_dir', type=str, default=r'./inputs', help='dir of output')
parser.add_argument('--ModelName', type=str,  default='SMT', help='model name')
parser.add_argument('--stage2_output_dir', type=str, default=r'./outputs', help='dir of output')
parser.add_argument('--stage2_model_weight', type=str, default=r'./weights/bestweights.pth', help='weight2')

def convert_state_dict(state_dict):
    """Converts a state dict saved from a dataParallel module to normal
        module state_dict inplace
        :param state_dict is the loaded DataParallel model_state
        You probably saved the model using nn.DataParallel, which stores the model in module, and now you are trying to load it
        without DataParallel. You can either add a nn.DataParallel temporarily in your network for loading purposes, or you can
        load the weights file, create a new ordered dict without the module prefix, and load it back
    """
    state_dict_new = OrderedDict()
    # print(type(state_dict))
    for k, v in state_dict.items():

        name = k[7:] # remove the prefix module.
        # My heart is broken, the pytorch have no ability to do with the problem.
        state_dict_new[name] = v
        if name == 'linear_e.weight':
            np.save('weight_matrix.npy', v.cpu().numpy())
    return state_dict_new

'''##################################load models##################################'''
def GetNetwork(args):
    if args.ModelName == 'SMT':


        net = SMT(img_size=96, in_chans=1, num_classes=14,
        embed_dims=[20*2, 40*2, 80*2, 160*2], ca_num_heads=[2, 2, 2, -1], sa_num_heads=[-1, -1, 8, 16], mlp_ratios=[2, 2, 2, 2], 
        qkv_bias=True, depths=[2, 2, 2, 2], ca_attentions=[1, 1, 1, 0], head_conv=3, expand_ratio=2)
        # net = SMT(embed_dims=[30*2, 60*2, 120*2, 240*2], num_classes=14, ca_num_heads=[3, 3, 3, -1], 
        #           sa_num_heads=[-1, -1, 8, 16], mlp_ratios=[2, 2, 2, 2], qkv_bias=True, depths=[2, 4, 6, 2],
        #             ca_attentions=[1, 1, 1, 0], head_conv=3, expand_ratio=2)
        net.load_state_dict(torch.load(args.stage2_model_weight, map_location='cpu'))
        net.eval()
        print('Model has been loaded')
    return net



def Infer(args, net, imagefile):
    test_image_files=[{"image": imagefile}]
    input_image = image_transform(test_image_files)
    post_transforms = infer_post_transform(test_transforms=image_transform)
    starttime = time.time()
    with torch.no_grad():      
        for test_data in input_image:

            images = test_data['image'].unsqueeze(dim=0)#add batch dim
            test_data['pred'] = sliding_window_inference(
                                images, (96, 96, 96), 1, net, 
                                overlap=0.1,sw_device='cpu',device='cpu')
            test_data['pred'] = test_data['pred'].squeeze(dim=0)
            test_data = post_transforms(test_data)
            print(torch.unique(test_data['pred']))


          
            os.makedirs(os.path.join(args.stage2_output_dir), exist_ok=True)


            savetransformer = Compose([SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir=os.path.join(args.stage2_output_dir),
                   output_postfix="", output_ext=".nii.gz", resample=False, separate_folder = False,output_dtype = np.uint8),])
            savetransformer(test_data)

            filename = Path(imagefile).name
            intermediate_path = os.path.join(args.stage2_output_dir, filename)
            new_file_name = filename.split('_0000.nii.gz')[0] + '.nii.gz'
            save_dir = os.path.join(args.stage2_output_dir, new_file_name)
            os.replace(intermediate_path, save_dir)  
            print(time.time()-starttime)


if __name__=="__main__":
    args = parser.parse_args()
    net = GetNetwork(args)
    net.eval()
    all_nii_list = sorted(glob.glob(os.path.join(args.inputs_dir, '*.nii.gz')))
    for NiiFile in all_nii_list:
        
        Infer(args, net, NiiFile)

        
        


