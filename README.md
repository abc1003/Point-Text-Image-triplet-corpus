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


### Our Point-Text-Image-triplet-corpus Dataset

| dataset  | url|
| --- | --- |
| PCN-TI | [Quark](https://pan.quark.cn/s/20c69b4d2f69)  |
| MVP-TI | [Quark](https://pan.quark.cn/s/20c69b4d2f69) |

## Project Structure 🗂️
root/
├── Dataset generation #MVP&PCN
│ 
├── examples #Using the model samples from the dataset we generated
│ 
└── utils/clip #CLIP files required to generate the dataset

## Acknowledgements

Our code is inspired by [PointCLIP](https://github.com/ZrrSkywalker/PointCLIP) and [PoinTr/AdaPointr](https://github.com/yuxumin/PoinTr).

## Citation
If you find our work useful in your research, please consider citing: 
```
@misc{zhou2024positionawareguidedpointcloud,
      title={Position-aware Guided Point Cloud Completion with CLIP Model}, 
      author={Feng Zhou and Qi Zhang and Ju Dai and Lei Li and Qing Fan and Junliang Xing},
      year={2024},
      eprint={2412.08271},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.08271}, 
}
```
