import os
import glob
import json
import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import warnings
import collections


warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)
class VPADataset(Dataset):
    def __init__(self, anno_dir, img_dir, transform=None):

        self.anno_list = list(json.load(open(i)) for i in \
                              glob.glob(os.path.join(anno_dir, '*.json')))
        self.attr_score = np.load('/home/chen/work/visual_privacy/vpa/user_studies/attr_score.npy', allow_pickle=True).item()
        self.img_dir = img_dir
        self.transform = transform
        # self.missing_list = [14, 15, 22, 28, 34, 36, 40, 42, 44, 45, 47, 50, 51, 52, 53, 54, 63, 71, 72, 76, 77, 80, 81, 83, 84, 86, 87, 88, 89, 91, 93, 94, 95, 96, 98]
    def __getitem__(self, idx):

        img_path = os.path.join(self.img_dir,
                                self.anno_list[idx]['image_path'])
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        target_list = list(float(self.attr_score[l]) if l in self.attr_score else 0 for l in self.anno_list[idx]['labels'])
        target = sum(target_list)/49

        return img, target, img_path

    def __len__(self):
        return len(self.anno_list)

    def getMSE(self, d1, d2):
        mse = 0.0
        for a,b in zip(d1, d2):
            mse += (a-b)**2
        return mse / len(d1)

    def getRankCorrelationWithMSE(self, predicted, gt=None):

        if gt is None:
            gt = self.labels.copy()

        gt = np.array(gt).tolist()
        predicted = np.array(predicted).squeeze().tolist()

        n = min(len(predicted), len(gt))
        if n < 2:
            return 0

        gt = gt[:n]
        predicted = predicted[:n]
        mse = self.getMSE(gt, predicted)

        def get_rank(list_a):
            # get GT rank
            rank_list = np.zeros(len(list_a))
            idxs = np.array(list_a).argsort()

            # Record the GT rank
            for rank, i in enumerate(idxs):
                rank_list[i] = rank

            return rank_list

        gt_rank = get_rank(gt)
        predicted_rank = get_rank(predicted)

        #-------------------------------------------------------
        ssd = 0
        for i in range(len(predicted_rank)):
            ssd += (gt_rank[i] -  predicted_rank[i])**2

        rc = 1-(6*ssd/(n*n*n - n))

        # spearmanr() from scipy package produces the same result
        # rc, _ = stats.spearmanr(a=predicted, b=gt, axis=0)

        return rc, mse
# python3 main.py --train-batch-size 512 --test-batch-size 512 --cnn ResNet50FC --dataset vpa 
# python3 main.py --cnn ResNet50FC --model-weights data/vpa_ResNet50FC_lstm3_score/best_weights.pkl --eval-images images/high --att-maps-out att_maps_score --csv-out memorabilities_score.csv --dataset vpa
# anno_dir = '/home/chen/work/visual_privacy/vpa/data/annos/train2017_anno/train2017'
# img_dir = '/home/chen/work/visual_privacy/vpa/data'

# train_data_transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225])])

# my_ds = VPADataset(anno_dir, img_dir, train_data_transform)
# score_list = []
# for i in range(my_ds.__len__()):
#     _, t = my_ds.__getitem__(i)
#     score_list.append(t)
# print('mean: ', sum(score_list)/float(len(score_list)))
# print('max: ', max(score_list))
#     print(t)
#     input('enter')
#     for l in t:
#         l = int(l[0]) if '_' in l else int(l)
#         if l not in label_dict:
#             label_dict[l] = 0
#         else:
#             label_dict[l] += 1

# od = collections.OrderedDict(sorted(label_dict.items()))
# print(len(od))
# print(od)

# miss_list = []
# for i in range(100):
#     if i not in od:
#         miss_list.append(i)
# print(miss_list)