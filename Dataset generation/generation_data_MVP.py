import h5py
import numpy as np
import torch
import torch.nn as nn
import yaml
from easydict import EasyDict
from matplotlib import pyplot as plt
from tqdm import tqdm

import projection
from clip import clip
from torch.utils.data import DataLoader
from PIL import Image
from datasets.PCNDataset import PCNClip
from realistic_projection import Realistic_Projection
from utils.config import merge_new_config
import model_utils
import itertools

import os
import open3d as o3d
from torch.utils.data import Dataset, DataLoader

pos_list = ['Top Left', 'Top Right', 'Bottom Left', 'Bottom Right']
def get_miss_prompt_idx():
    miss_prompt = []
    # 生成miss_prompt列表
    for r in range(1, len(pos_list) + 1):
        combinations = list(itertools.combinations(pos_list, r))
        for combination in combinations:
            combination = list(combination)
            miss_prompt.append(combination)
    return miss_prompt

def get_idx(textlist):
    miss_prompt = get_miss_prompt_idx()
    for idx,i in enumerate(miss_prompt):
        text = set(i)
        if text == textlist:
            return idx
def get_class(taxonomy_id):
    idx = category.index(taxonomy_id)
    return idx


def cutimg(image):
    # 获取图像的宽度和高度
    image = image.permute(0,2,3,1)
    _,height, width, _ = image.shape
    # 划分为四个格子
    grid_width = width // 2
    grid_height = height // 2
    # 提取四个格子的区域
    top_left = image[:,:grid_height, :grid_width]
    top_right = image[:,:grid_height, grid_width:]
    bottom_left = image[:,grid_height:, :grid_width,:]
    bottom_right = image[:,grid_height:, grid_width:,:]
    # 统计每个格子中非零元素的数量
    top_left = top_left.permute(0, 3, 1, 2)
    top_right = top_right.permute(0, 3, 1, 2)
    bottom_left = bottom_left.permute(0, 3, 1, 2)
    bottom_right = bottom_right.permute(0, 3, 1, 2)
    top_left1 = torch.nn.functional.interpolate(top_left, size=(224, 224), mode='bilinear', align_corners=True)
    top_right1 = torch.nn.functional.interpolate(top_right, size=(224, 224), mode='bilinear', align_corners=True)
    bottom_left1 = torch.nn.functional.interpolate(bottom_left, size=(224, 224), mode='bilinear', align_corners=True)
    bottom_right1 = torch.nn.functional.interpolate(bottom_right, size=(224, 224), mode='bilinear', align_corners=True)
    cut_imgs=(top_left1,top_right1,bottom_left1,bottom_right1)

    return cut_imgs


def check_loss_place(part,gt):
    pos_list = ['Top Left','Top Right','Bottom Left','Bottom Right']
    text = []
    part_cng = list(count_non_grid(part).values())
    gt_cng = list(count_non_grid(gt).values())
    for i,x in enumerate(gt_cng):
        grid_text = pos_list[i]
        if x == part_cng[i] and x!=0:

            text.append(grid_text)
    max_count = max(count_non_grid(part).values())
    grid_text = [grid for grid, count in count_non_grid(part).items() if count == max_count]
    text.append(grid_text[0])
    text = set(text)

    return text

def splitmvp():
    mvp_train_path = 'data/mvp/MVP_Train_CP.h5'
    input_file = h5py.File(mvp_train_path, 'r')
    data = np.array(input_file['incomplete_pcds'][()])
    datasets = torch.from_numpy(data)
    d1 = datasets[:15000]
    d2 = datasets[15000:30000]
    d3 = datasets[30000:45000]
    d4 = datasets[45000:]
    datatuple = (d1, d2, d3, d4)
    for idx, i in enumerate(datatuple):
        print(i)
def save_pic(tensor, picpath):
    tensor = np.transpose(tensor.numpy(), (1, 2, 0))
    tensor = (tensor * 255).astype(np.uint8)
    image = Image.fromarray(tensor)
    image.save(picpath)
def cfg_from_yaml_file(cfg_file):
    config = EasyDict()
    with open(cfg_file, 'r') as f:
        try:
            new_config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            new_config = yaml.load(f)
    merge_new_config(config=config, new_config=new_config)
    return config

class ImagePartAttention(nn.Module):
    def __init__(self, embed_dim=512, num_heads=1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):

        B, N, C = x.shape  # N=4表示4个图像块

        q = self.q_proj(x)  # (B,4,512)
        k = self.k_proj(x)  # (B,4,512)
        v = self.v_proj(x)  # (B,4,512)

        scale = 1.0 / (self.head_dim ** 0.5)
        attn_logits = torch.bmm(q, k.transpose(-2, -1)) * scale  # (B,4,4)

        attn_weights = F.softmax(attn_logits, dim=-1)  # (B,4,4)

        out = torch.bmm(attn_weights, v)  # (B,4,512)

        out = self.out_proj(out)

        return out
class Extractor(torch.nn.Module):
    def __init__(self, model):
        super(Extractor, self).__init__()

        self.model = model
        self.pc_views = Realistic_Projection()
        self.get_img = self.pc_views.get_img


    def mv_projmap(self,pc):

        pc_views = model_utils.PCViews(-0.7,224)
        PCVIEW = pc_views.get_img
        img = PCVIEW(pc).cuda()
        img = img.unsqueeze(1).repeat(1, 3, 1, 1)
        img = torch.nn.functional.interpolate(img, size=(224, 224), mode='bilinear', align_corners=True)

        return img


    def get_all_text_feat(self):
        with open('PCN_text_prompt.json','r') as file:
             json_data = json.load(file)
             categories = list(json_data.keys())
             miss_prompt_data = list(json_data.values())
             n = len(categories)
             m = len(miss_prompt_data[0])
             text_data = np.empty((n, m, 1), dtype=object)
             text_feat_data = torch.empty((n, m, 512)).cuda()
             for i, prompts in enumerate(miss_prompt_data):
                 prompts1 = torch.cat([clip.tokenize(p) for p in prompts])
                 prompts1 = prompts1.cuda()
                 text_feat = self.model.encode_text(prompts1)
                 text_data[i] = np.array(prompts).reshape(m, 1)
                 text_feat_data[i] = text_feat

        return text_feat_data
    def forward(self, pc):

        img = self.mv_projmap(pc)
        cutimgs = cutimg(img)
        img__class_feats, img_feats = self.model.encode_image(img)
        img__class_feats_b0, _ = self.model.encode_image(cutimgs[0])
        img__class_feats_b1, _ = self.model.encode_image(cutimgs[1])
        img__class_feats_b2, _ = self.model.encode_image(cutimgs[2])
        img__class_feats_b3, _ = self.model.encode_image(cutimgs[3])

        img__class_feats_b0 = img__class_feats_b0.unsqueeze(1)
        img__class_feats_b1 = img__class_feats_b1.unsqueeze(1)
        img__class_feats_b2 = img__class_feats_b2.unsqueeze(1)
        img__class_feats_b3 = img__class_feats_b3.unsqueeze(1)
        img_feat_block = torch.cat((img__class_feats_b0,img__class_feats_b1,img__class_feats_b2,img__class_feats_b3),dim=1)

        img__class_feats = img__class_feats / img__class_feats.norm(dim=-1, keepdim=True)
        img_feat_block = img_feat_block / img_feat_block.norm(dim=-1, keepdim=True)

        attn = ImagePartAttention()
        img_feat_pa = attn(img_feat_block.float())

        return img__class_feats,img_feat_pa

def extract_feature_maps():
    device = "cuda:0"
    model, _ = clip.load('ViT-B/16', device=device)
    model.to(device)

    extractor = Extractor(model)
    extractor = extractor.to(device)
    extractor.eval()
    # 数据集路径
    mvp_train_path ='data/mvp/MVP_Train_CP.h5'
    mvp_test_path ='data/mvp/MVP_Test_CP.h5'
    trainpath1 = 'data/mvp/MVP_Test_CP_2.pt'
    # input_file = h5py.File(mvp_test_path, 'r')
    # data = np.array(input_file['incomplete_pcds'][()])
    datasets = torch.load(trainpath1)
    clip_featlist = []
    for points in tqdm(datasets):

        points = points.reshape(1,2048,3).cuda()

        with torch.no_grad():
            img_feat,img_feat_block = extractor(points.float())

            clip_feat = (img_feat,img_feat_block)

            clip_featlist.append(clip_feat)

            # print(name1)
    torch.save(clip_featlist,'clip_featlist.pt')


if __name__ == '__main__':
    extract_feature_maps()




