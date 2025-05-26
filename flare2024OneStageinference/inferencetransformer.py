import warnings
warnings.filterwarnings('ignore')

import numpy as np

from monai import transforms
from monai.transforms import (
    Orientationd,
    ToTensord,
    AsDiscreted,
    Compose,
    CropForegroundd,
    LoadImaged,   
    EnsureChannelFirstd,
    ScaleIntensityRangePercentilesd,
    KeepLargestConnectedComponentd,
    Spacingd,   
    EnsureTyped,
    Invertd,
    NormalizeIntensityd,
    SaveImaged,
    Activationsd,
    RemoveSmallObjectsd,
    SpatialPadd
   
)

sample_patch = (96,96,96)
# image_transform = Compose(
#                 [
#                     LoadImaged(keys=["image"] ),
#                     EnsureChannelFirstd(keys=["image"]),  
#                     ScaleIntensityRangePercentilesd(keys=["image"],lower = 5, upper = 95, b_min =0, b_max=1, clip=True),                  
#                     Spacingd(
#                                     keys=["image"],
#                                     pixdim=(1.5, 1.5, 1.5),
#                                     mode=("bilinear"),
#                                 ),                   

#                     CropForegroundd(keys=["image"], source_key="image"), 
#                     # ToTensord(keys=["image"]),
#                     NormalizeIntensityd(keys="image", nonzero=True),
#                     SpatialPadd(keys=['image'], spatial_size=sample_patch,method ='end',mode='constant')
#                 ]
#     ) 

image_transform = transforms.Compose(
                [
                transforms.LoadImaged(keys=["image"]),
                transforms.EnsureChannelFirstd(keys=["image"]),
                transforms.ScaleIntensityRangePercentilesd(keys=["image"],lower= 5, upper= 95, b_min=0, b_max=255, clip=True), #channel_wise=True for MRI
                
                transforms.CropForegroundd(keys=["image"], source_key="image"),
                transforms.Spacingd(keys=["image"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear")),
                transforms.NormalizeIntensityd(keys=['image'], nonzero=True,channel_wise=True),
                transforms.SpatialPadd(keys=["image"],method ='end',mode='constant', spatial_size=[96, 96, 96]),

                ]
            )



def infer_post_transform(test_transforms):
    post_transforms = Compose([
       
        Activationsd(keys="pred", softmax=True),
        AsDiscreted(keys="pred", argmax=True),
        KeepLargestConnectedComponentd(keys='pred', applied_labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                                       independent=True,num_components=1), 
        
        # AsDiscreted(keys="pred", argmax=True), 
        Invertd(
            keys="pred", 
            transform=test_transforms,
            orig_keys="image", 
            meta_keys="pred_meta_dict", 
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict", 
            nearest_interp=True,  
        ),
        # SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir='./outputs',
        #            output_postfix="", output_ext=".nii.gz", resample=False, separate_folder = False,output_dtype = np.uint8),

    ])
    return post_transforms