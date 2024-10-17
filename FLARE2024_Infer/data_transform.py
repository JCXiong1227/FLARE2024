
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from monai.transforms import (
    AsDiscreted,
    Compose,
    CropForegroundd,
    LoadImaged,
    NormalizeIntensityd,
    EnsureChannelFirstd,
    ScaleIntensityRangePercentilesd,
    KeepLargestConnectedComponentd,
    Spacingd,
    Resized,
    EnsureTyped,
    Invertd,
    SaveImaged,
    Activationsd,
    SpatialPadd,
RemoveSmallObjectsd,
)

image_initial_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            ScaleIntensityRangePercentilesd(keys=["image"], lower=5, upper=95, b_min=0, b_max=255, clip=True),
            
            # CropForegroundd(keys=["image"], source_key="image"),
        ])

resize_coarse = (128, 128, 128)
stage1_test_transforms =  Compose(            
        [ 
            Resized(keys=["image"], spatial_size=resize_coarse, mode=("trilinear")),
            NormalizeIntensityd(keys=['image'], nonzero=True),
        ])

sample_patch = (96,96,96)
stage2_test_transforms = Compose(
        [
            CropForegroundd(keys=["image"], source_key="pred_ROI"),  # 取label 的 value > 0
            Spacingd(
                keys=["image"],
                pixdim=(1.5, 1.5, 2),
                mode=("bilinear"),
            ),
            
            
            NormalizeIntensityd(keys=['image'], nonzero=True),
            SpatialPadd(keys=['image'], spatial_size=sample_patch, method='end', mode='constant'),
            # 对低于spatial_size 的维度pad
        ]
    )

stage1_post_transforms = Compose([
        # EnsureTyped(keys="pred_ROI"),
        Activationsd(keys="pred_ROI", softmax=True),
        AsDiscreted(keys="pred_ROI", argmax=True),

        Invertd(
            keys="pred_ROI",  # invert the `pred` data field, also support multiple fields
            transform=stage1_test_transforms,
            orig_keys="image",  # get the previously applied pre_transforms information on the `img` data field,
            # then invert `pred` based on this information. we can use same info
            # for multiple fields, also support different orig_keys for different fields
            meta_keys="pred_ROI_meta_dict",  # key field to save inverted meta data, every item maps to `keys`
            orig_meta_keys="image_meta_dict",  # get the meta data from `img_meta_dict` field when inverting,
            # for example, may need the `affine` to invert `Spacingd` transform,
            # multiple fields can use the same meta data to invert
            meta_key_postfix="meta_dict",  # if `meta_keys=None`, use "{keys}_{meta_key_postfix}" as the meta key,
            # if `orig_meta_keys=None`, use "{orig_keys}_{meta_key_postfix}",
            # otherwise, no need this arg during inverting
            nearest_interp=True,  # don't change the interpolation mode to "nearest" when inverting transforms
            # to ensure a smooth output, then execute `AsDiscreted` transform
            # to_tensor=True,  # convert to PyTorch Tensor after inverting
        ),
        RemoveSmallObjectsd(keys="pred_ROI", min_size=20*20*20,independent_channels=True),

        # SaveImaged(keys="pred_ROI", meta_keys="pred_ROI_meta_dict", output_dir='intermediate_output',
        #            output_postfix="stage1", output_ext=".nii.gz", resample=False),

    ])

stage2_tolabel = Compose([
        # EnsureTyped(keys="pred"),
        Activationsd(keys="pred", softmax=True),
        AsDiscreted(keys="pred", argmax=True),
        # print(11111)
        KeepLargestConnectedComponentd(keys='pred', applied_labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                                       independent=True,num_components=1),  # 最大连通域 for all organs
        # SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir='final_output',
        #            output_postfix="stage2——raw", output_ext=".nii.gz", resample=False), #在invertd 已经 resample回去了
])
stage2_post_transforms =   Compose(
    [Invertd(
            keys="pred",  # invert the `pred` data field, also support multiple fields
            transform=stage2_test_transforms,
            orig_keys="image",  # get the previously applied pre_transforms information on the `img` data field,
            # then invert `pred` based on this information. we can use same info
            # for multiple fields, also support different orig_keys for different fields
            meta_keys="pred_meta_dict",  # key field to save inverted meta data, every item maps to `keys`
            orig_meta_keys="image_meta_dict",  # get the meta data from `img_meta_dict` field when inverting,
            # for example, may need the `affine` to invert `Spacingd` transform,
            # multiple fields can use the same meta data to invert
            meta_key_postfix="meta_dict",  # if `meta_keys=None`, use "{keys}_{meta_key_postfix}" as the meta key,
            # if `orig_meta_keys=None`, use "{orig_keys}_{meta_key_postfix}",
            # otherwise, no need this arg during inverting
            nearest_interp=True,  # don't change the interpolation mode to "nearest" when inverting transforms
            # to ensure a smooth output, then execute `AsDiscreted` transform
            # to_tensor=True,  # convert to PyTorch Tensor after inverting![](../../../AppData/Local/Temp/64b79613b0a31b0001b9a90b.png)
        ),
       
        
        SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir='./outputs',
                   output_postfix="", output_ext=".nii.gz", resample=False, separate_folder = False,output_dtype = np.uint8), #在invertd 已经 resample回去了
        ])

restore_transforms = Compose([
        Invertd(
            keys="pred",  # invert the `pred` data field, also support multiple fields
            transform=image_initial_transforms,
            orig_keys="image",  # get the previously applied pre_transforms information on the `img` data field,
            # then invert `pred` based on this information. we can use same info
            # for multiple fields, also support different orig_keys for different fields
            meta_keys="pred_meta_dict",  # key field to save inverted meta data, every item maps to `keys`
            orig_meta_keys="image_meta_dict",  # get the meta data from `img_meta_dict` field when inverting,
            # for example, may need the `affine` to invert `Spacingd` transform,
            # multiple fields can use the same meta data to invert
            meta_key_postfix="meta_dict",  # if `meta_keys=None`, use "{keys}_{meta_key_postfix}" as the meta key,
            # if `orig_meta_keys=None`, use "{orig_keys}_{meta_key_postfix}",
            # otherwise, no need this arg during inverting
            nearest_interp=True,  # don't change the interpolation mode to "nearest" when inverting transforms
            # to ensure a smooth output, then execute `AsDiscreted` transform
            # to_tensor=True,  # convert to PyTorch Tensor after inverting![](../../../AppData/Local/Temp/64b79613b0a31b0001b9a90b.png)
        ),      
        SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir='./outputs',
                   output_postfix="", output_ext=".nii.gz", resample=False,separate_folder = False,output_dtype = np.uint8), #在invertd 已经 resample回去了
    ])



