#  Position-aware Guided Point Cloud Completion with CLIP Model

we propose a rapid and efficient method to expand an unimodal framework into a multimodal framework.This approach incorporates a position-aware module designed to enhance the spatial information of the missing parts through a weighted map learning mechanism.In addition, we establish a Point-Text-Image triplet corpus PCN-TI and MVP-TI based on the existing unimodal point cloud completion dataset and use the pre-trained vision-language model CLIP to provide richer detail information for 3D shapes, thereby enhancing performance.

![datasets](https://github.com/user-attachments/assets/fc9ca4d6-521b-4fb2-826b-5453c485ce17)

## Usage

### Requirements

- PyTorch >= 1.7.0
- python >= 3.7
- CUDA >= 9.0
- GCC >= 4.9 
- torchvision
- open3d
- tensorboardX

Dependent on backbone network


### Dataset

| dataset  | url|
| --- | --- |
| PCN-TI | [Quark](https://pan.quark.cn/s/20c69b4d2f69)  |
| MVP-TI | [Quark](https://pan.quark.cn/s/20c69b4d2f69) |


## Acknowledgements

Our code is inspired by [PointCLIP](https://github.com/ZrrSkywalker/PointCLIP) and [PoinTr](https://github.com/yuxumin/PoinTr).

## Citation
If you find our work useful in your research, please consider citing: 
```
@inproceedings{yu2021pointr,
  title={PoinTr: Diverse Point Cloud Completion with Geometry-Aware Transformers},
  author={Yu, Xumin and Rao, Yongming and Wang, Ziyi and Liu, Zuyan and Lu, Jiwen and Zhou, Jie},
  booktitle={ICCV},
  year={2021}
}
```
