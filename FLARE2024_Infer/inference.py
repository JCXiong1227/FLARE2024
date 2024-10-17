import numpy as np
import torch
from pathlib import Path
import os
from models.partial3D import FasterNet
from models.phase2_newnetwork import  SMT
from data_transform import stage2_tolabel,image_initial_transforms,stage1_test_transforms,stage2_test_transforms,stage1_post_transforms,stage2_post_transforms,restore_transforms
import argparse
from monai.inferers import sliding_window_inference
import time
import glob
from collections import OrderedDict

'''set hyperparameters'''

parser = argparse.ArgumentParser()
parser.add_argument('--inputs_dir', type=str, default=r'./inputs', help='dir of output')
parser.add_argument('--stage2_output_dir', type=str, default=r'./outputs', help='dir of output')
parser.add_argument('--stage1_model_weight', type=str, default='./weights/model_stage1.pth', help='weight1')
parser.add_argument('--stage2_model_weight', type=str, default='./weights/model_stage2.pth', help='weight2')
args = parser.parse_args()



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

#

net_stage1 = FasterNet(in_chans=1,
                    num_classes=2,
                    embed_dim=32,
                    decoder_dim=64,
                    dims=(32, 64, 128, 256)
                    )
net_stage1.load_state_dict(torch.load(args.stage1_model_weight, map_location='cpu'))
net_stage1.eval()
print('net_stage1 load')


net_stage2 = SMT(
embed_dims=[30*2, 60*2, 120*2, 240*2], num_classes=14, ca_num_heads=[3, 3, 3, -1], sa_num_heads=[-1, -1, 8, 16], mlp_ratios=[2, 2, 2, 2], 
qkv_bias=True, depths=[2, 4, 6, 2], ca_attentions=[1, 1, 1, 0], head_conv=3, expand_ratio=2)


net_stage2.load_state_dict(convert_state_dict(torch.load(args.stage2_model_weight, map_location='cpu')))
net_stage2.eval()
print('net_stage2 load')

'''##################################find all test data##################################'''
all_nii_list = sorted(glob.glob(os.path.join(args.inputs_dir, '*.nii.gz')))
''' predict one by one'''

global_start_time = time.time()
for nii_name in all_nii_list:
        print('prediction starts!')
        '''---------------------阶段一数据加载、模型加载及推理-------------------------------------------------'''
        start_time = time.time()

        test_image_files=[{"image": nii_name}]
        input_tensor = image_initial_transforms(test_image_files)
        input_stage1 = stage1_test_transforms(input_tensor)
        # print((input_tensor)[0]['image'].data)
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                for test_data in input_stage1:
                    images = test_data['image'].unsqueeze(dim=1)
                    test_data['pred_ROI'] = net_stage1(images).squeeze(dim=0).cpu()
                    test_data = stage1_post_transforms(test_data)
                    test_data['image'] =  input_tensor[0]['image']

                # print('stage1 prediction finished')

                # ''' determine if any forground label exist'''

                if not torch.any(test_data['pred_ROI']==1):
                    """不存在前景的情况下"""
                    print('no abdominal organs detected!!!')
                    input_tensor[0]['pred'] = test_data['pred_ROI']
                    output_tensor = restore_transforms(input_tensor[0])
                   
                else:
                    #----------------------阶段二数据加载、模型加载及推理-------------------------------------------------------------------
                        input_stage2 = stage2_test_transforms([test_data])
                        for out_data in input_stage2:
                            images = out_data['image'].unsqueeze(dim=1)
                            out_data['pred'] = sliding_window_inference(
                                        images, (96, 96, 96), 1, net_stage2, overlap=0.05,sw_device='cpu',device='cpu')
                            out_data['pred'] = out_data['pred'].squeeze(dim=0) # c, h,w,d

                                     
                            out_data = stage2_tolabel(out_data)
                                                      
                            out_data = stage2_post_transforms(out_data)

        filename = Path(nii_name).name
        intermediate_path = os.path.join(args.stage2_output_dir, filename)
        new_file_name = filename.split('_0000.nii.gz')[0] + '.nii.gz'
        save_dir = os.path.join(args.stage2_output_dir, new_file_name)
        os.replace(intermediate_path, save_dir)                  
                
        t2 = time.time()
        print("total推理时间：{}".format(t2 - start_time))
        print('final prediction finished,next')
print('all done')
print(time.time()-global_start_time)
   















