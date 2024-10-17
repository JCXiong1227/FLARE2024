import os
import glob
import re
from pathlib import Path
import numpy as np
# from prefetch_generator import BackgroundGenerator
from monai.transforms import (
    AsDiscrete,
    NormalizeIntensityd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Transposed,
    LabelFilterd,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    EnsureTyped,
    RandAffined,
    SpatialPadd,
    Resized,
    RandScaleIntensityd,
    HistogramNormalized,
    CenterSpatialCrop,
    RandCropByLabelClassesd,
    EnsureChannelFirstd,
ScaleIntensityRangePercentilesd,    RandSpatialCropd,
    ToTensord,
    ThresholdIntensityd,
    CenterSpatialCropd)

from monai.data import (
    ThreadDataLoader,
    DataLoader,
    CacheDataset,
    Dataset,
    load_decathlon_datalist,
    decollate_batch,
    set_track_meta
    
)

import warnings
warnings.filterwarnings('ignore')   

def organ_data_module(task_dir, dataset = 'flare2024', phase = 'phase_1', batch_size=1,cache=False,train_mode=False):
   
    data_root = task_dir
    train_files=[]
    val_files=[]
    if dataset == 'flare2024':
        

        trainimagesroot = os.path.join(data_root, 'imagesTr')
        trainlabelsroot = os.path.join(data_root, 'finallabels')

        valimagesroot = os.path.join(data_root, 'imagesVal', 'images')
        vallabelsroot = os.path.join(data_root, 'imagesVal', 'labels')

        for train_imagename in os.listdir(trainimagesroot):
            train_labelname = train_imagename.split('_0000.nii.gz')[0]+r'.nii.gz'

            train_imagedir = os.path.join(trainimagesroot, train_imagename)
            train_labeldir = os.path.join(trainlabelsroot, train_labelname)
            if os.path.exists(train_labeldir):

                train_files.append({'image': train_imagedir, 'label': train_labeldir})

        for val_imagename in os.listdir(valimagesroot):
            val_labelname = val_imagename.split('_0000.nii.gz')[0]+r'.nii.gz'
            val_imagedir = os.path.join(valimagesroot, val_imagename)
            val_labeldir = os.path.join(vallabelsroot, val_labelname)
            val_files.append({'image': val_imagedir, 'label': val_labeldir})

    

    if train_mode:
        
        resize_coarse = (128,128,128)

        if dataset == 'flare2024' and phase == 'phase_1':
            train_transforms = Compose(
                [
                    LoadImaged(keys=["image", "label"]), #若是nibabel读取（nii.gz文件）， 转化为sitk 格式
                    EnsureChannelFirstd(keys=["image", "label"]),                    
                    # ScaleIntensityRangePercentilesd(keys=["image"],lower = 5, upper = 95, b_min =0, b_max=255, clip=True),
                    # CropForegroundd(keys=["image", "label"], source_key="image"), 
                    # Spacingd(
                    #     keys=["image", "label"],
                    #     pixdim=(2, 2, 2.5),   #plyu设置的是1.5，1.5，2             
                    #     mode=("bilinear", "nearest"),
                    # ),
                    
                    Resized( keys=["image", "label"], spatial_size=resize_coarse , mode=("trilinear", "nearest")),
                    NormalizeIntensityd(keys='image',nonzero =True), 

                    
                    # SpatialPadd(keys=['image', 'label'], spatial_size=sample_patch,method ='end',mode='constant'), #对低于spatial_size 的维度pad

                    RandFlipd(
                        keys=["image", "label"],
                        spatial_axis=[0],
                        prob=0.10,
                    ),
                    RandFlipd(
                        keys=["image", "label"],
                        spatial_axis=[1],
                        prob=0.10,
                    ),
                    RandFlipd(
                        keys=["image", "label"],
                        spatial_axis=[2],
                        prob=0.10,
                    ),
                    RandRotate90d(
                        keys=["image", "label"],
                        prob=0.10,
                        max_k=3,
                    ),
                    # RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.1), plyu中使用，被jcxiong注释
                    RandShiftIntensityd(
                        keys=["image"],
                        offsets=0.10,
                        prob=0.1,
                    ),
                    RandAffined(
                        keys=['image', 'label'],
                        mode=('bilinear', 'nearest'),
                        prob=0.1, spatial_size=resize_coarse,
                        rotate_range=(0, 0, np.pi / 15),
                        scale_range=(0.1, 0.1, 0.1)),

                    ToTensord(keys=["image", "label"]),            
                ]
            )
          
            val_transforms = Compose(
                [
                    LoadImaged(keys=["image", "label"] ),
                    EnsureChannelFirstd(keys=["image", "label"]),  
                    # KeepLargestConnectedComponentd(keys='label',independent=True, applied_labels=[1,2,3,4,5,6,7,8,9,10,11,12,13],),
   
                    # ScaleIntensityRangePercentilesd(keys=["image"],lower = 5, upper = 95, b_min =0, b_max=255, clip=True),
                    # CropForegroundd(keys=["image", "label"], source_key="image"),

                    # Spacingd(
                    #                 keys=["image", "label"],
                    #                 pixdim=(2, 2, 2.5),#plyu设置的是1.5，1.5，2
                    #                 mode=("bilinear", "nearest"),
                    #             ),
                    
                    Resized( keys=["image", "label"], spatial_size=resize_coarse , mode=("trilinear", "nearest")),
                    NormalizeIntensityd(keys='image',nonzero=True), 
                    ToTensord(keys=["image", "label"]),

                ]
            )


        elif dataset == 'flare2024' and phase == 'phase_2':
          
            sample_patch = (96,96,96)
            num_samples= 6
            train_transforms = Compose(
                [
                    LoadImaged(keys=["image", "label"] ),
                    EnsureChannelFirstd(keys=["image", "label"]),  
                    #  KeepLargestConnectedComponentd(keys='label', applied_labels=[1,2,3,4,5,6,7,8,9,10,11,12,13],independent= True), #最大连通域
                    ScaleIntensityRangePercentilesd(keys=["image"],lower = 5, upper = 95, b_min =0, b_max=255, clip=True),
                    # CropForegroundd(keys=["image", "label"], source_key="label",margin = 20), #取label 的 value > 0 R# ROI croped 
                    Spacingd(
                        keys=["image", "label"],
                        pixdim=(2, 2, 2.0),                
                        mode=("bilinear", "nearest"),
                    ),
                    CropForegroundd(keys=["image", "label"], source_key="label"), #取label 的 value > 0

                    NormalizeIntensityd(keys='image',nonzero =True), 
                    SpatialPadd(keys=['image', 'label'], spatial_size=sample_patch,method ='end',mode='constant'), #对低于spatial_size 的维度pad

                    RandCropByPosNegLabeld(
                                        keys=["image", "label"],
                                        label_key="label",
                                        spatial_size=sample_patch,
                                        pos=5,
                                        neg=1,
                                        num_samples=num_samples,
                                    ),
                                   
                                    
                    RandFlipd(
                        keys=["image", "label"],
                        spatial_axis=[0],
                        prob=0.10,
                    ),
                    RandFlipd(
                        keys=["image", "label"],
                        spatial_axis=[1],
                        prob=0.10,
                    ),
                    RandFlipd(
                        keys=["image", "label"],
                        spatial_axis=[2],
                        prob=0.10,
                    ),
                    RandRotate90d(
                        keys=["image", "label"],
                        prob=0.10,
                        max_k=3,
                    ),
                    RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.1),
                    RandShiftIntensityd(
                        keys=["image"],
                        offsets=0.10,
                        prob=0.1,
                    ),
                    RandAffined(
                        keys=['image', 'label'],
                        mode=('bilinear', 'nearest'),
                        prob=0.1, spatial_size=  sample_patch,
                        rotate_range=(0, 0, np.pi / 15),
                        scale_range=(0.1, 0.1, 0.1)),

                    ToTensord(keys=["image", "label"]), 

                ]
            )

            
            val_transforms = Compose(
                [
                    LoadImaged(keys=["image", "label"] ),
                    EnsureChannelFirstd(keys=["image", "label"]),  
                    # KeepLargestConnectedComponentd(keys='label',independent=True, applied_labels=[1,2,3,4,5,6,7,8,9,10,11,12,13],),
   
                    ScaleIntensityRangePercentilesd(keys=["image"],lower = 5, upper = 95, b_min =0.0, b_max=255.0, clip=True),
                    # ScaleIntensityRangePercentilesd(keys=["image"],lower = 5, upper = 95, b_min =0, b_max=255, clip=True),
                    # CropForegroundd(keys=["image", "label"], source_key="image"), 
                    
                    Spacingd(
                                keys=["image", "label"],
                                pixdim=(2, 2, 2.0),  
                                mode=("bilinear", "nearest"),
                            ),
                    CropForegroundd(keys=["image", "label"], source_key="label"), #取label 的 value > 0
                    
                    # ScaleIntensityRangePercentilesd(keys=["image"],lower = 5, upper = 95, b_min =0, b_max=255, clip=True),
                    

                    

                    NormalizeIntensityd(keys='image',nonzero=True), 
                    
                    
                    
                    SpatialPadd(keys=['image', 'label'], spatial_size=sample_patch,method ='end',mode='constant'), #对低于spatial_size 的维度pad
                    ToTensord(keys=["image", "label"]),

                ]
            )

        #先把数据cache起来，减少训练时间, 但是容易爆显存
        if cache:
            train_ds = CacheDataset(
                data=train_files,
                transform=train_transforms,
                cache_num=len(train_files),
                cache_rate=1.0,
                num_workers=8,
                copy_cache=False
            )

            # disable multi-workers because `ThreadDataLoader` works with multi-threads
            train_loader = DataLoader(train_ds, num_workers=6, batch_size=batch_size, shuffle=True, drop_last=True)
            val_ds = CacheDataset(
                data=val_files, transform=val_transforms, cache_num=len(val_files), cache_rate=1.0, num_workers=0,copy_cache=False
            )
            val_loader = DataLoader(val_ds, num_workers=0, batch_size=1,shuffle=False,)

        else:
            train_ds = Dataset(
                data=train_files,
                transform=train_transforms,   
            )
            
            train_loader =    DataLoader(train_ds, num_workers=0, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True) 
            val_ds = Dataset(
                data=val_files, transform=val_transforms
            )
            val_loader = DataLoader(val_ds, num_workers=0, batch_size=1,pin_memory=True)

            # for i in  val_ds:
            #      print(i["image"].meta["filename_or_obj"])
        return train_ds,train_loader,val_ds,val_loader 
        #return train_ds,val_ds 


