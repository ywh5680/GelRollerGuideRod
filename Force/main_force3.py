import matplotlib.pyplot as plt
plt.switch_backend('agg')
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
import argparse
import progressbar
import numpy as np
from utils.progbar import *
from utils.config import *
from utils.loss_function import *
from utils.RollingMeasure import *
from utils.module import *
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
from torchvision.utils import *
from PIL import Image

from sklearn.linear_model import LinearRegression
import csv
from tensorboardX import SummaryWriter
from torchvision import datasets, transforms
from sklearn.metrics import r2_score
from math import sqrt
from sklearn.metrics import mean_absolute_error  # MAE
from sklearn.metrics import mean_squared_error  # MSE
from torchvision.utils import save_image
import shutil
import math

# reproducibility
init_seed = 1
torch.manual_seed(init_seed)
torch.cuda.manual_seed(init_seed)
torch.cuda.manual_seed_all(init_seed)
np.random.seed(init_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

transform1 = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.1),
    transforms.RandomRotation(30),
    transforms.RandomVerticalFlip(p=0.1),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
transform2 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class Datasets(Dataset):
    def __init__(self, flist_path=None, config=None, AugFlag=None):
        self.config = config
        self.data_path = self.config.DATA_PATH
        self.flist_path = flist_path
        self.all_filelist = self.get_all_filelist()
        self.transform = AugFlag
        self.label_file = None
        if hasattr(self.config, 'LABEL_FLIST') and self.config.LABEL_FLIST:
            try:
                self.label_file = self.get_label(self.config.LABEL_FLIST)
            except Exception:
                self.label_file = None

    def get_all_filelist(self):
        with open(self.flist_path, 'rt') as f:
            filelist = f.read().splitlines()
        return filelist

    def get_label(self, path):
        label_dict = {}
        with open(path, "r") as f:
            for line in f.readlines():
                line = line.strip('\n')
                if not line:
                    continue
                group, label = line.split(' ')
                label_dict[group] = label
        return label_dict

    def __len__(self):
        return len(self.all_filelist)

    def __getitem__(self, idx):
        par_list = self.all_filelist[idx].strip()
        parts = par_list.split()
        sensor, folder_name, image_name, Ft, Fn = None, None, None, None, None

        if len(parts) == 4:  # "0.jpg Fx Fy Fz"
            image_name = parts[0]
            fx, fy, fz = float(parts[1]), float(parts[2]), float(parts[3])
            Ft, Fn = math.sqrt(fx * fx + fy * fy), fz
            img_candidates = [
                os.path.join(self.data_path, 'images', image_name),
                os.path.join(self.data_path, image_name),
            ]
        elif len(parts) >= 6:  # "sensor05 session 584.jpg Fx Fy Fz"
            sensor, folder_name, image_name = parts[0], parts[1], parts[2]
            fx, fy, fz = float(parts[3]), float(parts[4]), float(parts[5])
            if len(parts) >= 8:
                try:
                    Ft, Fn = float(parts[6]), float(parts[7])
                except:
                    Ft, Fn = math.sqrt(fx * fx + fy * fy), fz
            else:
                Ft, Fn = math.sqrt(fx * fx + fy * fy), fz
            img_candidates = [
                os.path.join(self.data_path, sensor + '_256', folder_name, 'tactile', image_name),
                os.path.join(self.data_path, folder_name, 'tactile', image_name),
                os.path.join(self.data_path, 'images', image_name),
            ]
        else:
            raise ValueError(f"Unrecognized line format in flist: {par_list}")

        image_tar_path = None
        for c in img_candidates:
            if c and os.path.exists(c):
                image_tar_path = c
                break
        if image_tar_path is None:
            flist_dir = os.path.dirname(self.flist_path)
            candidate = os.path.join(flist_dir, image_name)
            if os.path.exists(candidate):
                image_tar_path = candidate
            else:
                raise FileNotFoundError(f"Image not found: {par_list}")

        image_tar = self.get_data(image_tar_path)
        if self.transform:
            image_tar = self.transform(image_tar)

        force = np.array([Ft, Fn], dtype=np.float32).reshape(2,)
        h_obj = np.array([0.], dtype=np.float32)
        if self.label_file is not None and folder_name is not None:
            try:
                key = str(int(folder_name.split('_')[1]))
                h_obj = np.array(np.float32(self.label_file[key])).reshape(1,)
            except Exception:
                h_obj = np.array([0.], dtype=np.float32)

        if sensor is None:
            h_sensor = np.array([4.], dtype=np.float32)
        else:
            if sensor == 'sensor01': h_sensor = np.array([4.])
            elif sensor == 'sensor02': h_sensor = np.array([8.])
            elif sensor == 'sensor03': h_sensor = np.array([12.])
            else: h_sensor = np.array([4.])

        return image_tar, torch.from_numpy(force).float(), torch.from_numpy(h_obj).float(), torch.from_numpy(h_sensor).float()

    def get_data(self, filepath):
        data = Image.open(filepath).convert('RGB')
        data = data.resize((128, 128))
        return data


class ResNetMultiImageInput(models.ResNet):
    def __init__(self, block, layers, num_input_images=1):
        super(ResNetMultiImageInput, self).__init__(block, layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class ForceEstimation(nn.Module):
    def __init__(self, args):
        super(ForceEstimation, self).__init__()
        depth = args.Resnet_depth
        blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[depth]
        block_type = {18: models.resnet.BasicBlock, 50: models.resnet.Bottleneck}[depth]
        self.resnet = ResNetMultiImageInput(block_type, blocks, num_input_images=1)

        if depth == 18:
            pretrained_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        elif depth == 50:
            pretrained_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        else:
            raise NotImplementedError(f"ResNet-{depth} not implemented here")

        # 载入预训练权重
        self.resnet.load_state_dict(pretrained_model.state_dict(), strict=False)

        # encoder 输出通道数要根据 depth 调整
        if depth == 18:
            self.encoder = torch.nn.Sequential(*list(self.resnet.children())[:6])  # layer2输出128
            feat_dim = 128
        elif depth == 50:
            self.encoder = torch.nn.Sequential(*list(self.resnet.children())[:6])  # layer2输出512
            feat_dim = 512

        self.maxpool = nn.MaxPool2d(kernel_size=5, stride=3, padding=1)
        self.fc1 = nn.Linear(feat_dim * 5 * 5, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 2)


    def forward(self, imgs):
        out = self.encoder(imgs)
        out = self.maxpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out


class Trainer():
    def __init__(self, args):
        self.config = args
        self.epoch = 0
        self.best_score = 1e9
        self.model = ForceEstimation(self.config).to(self.config.DEVICE)
        self.model = nn.DataParallel(self.model)
        self.solver_Stiffness = optim.Adam(list(self.model.parameters()), lr=self.config.LR)
        self.train_dataset = Datasets(flist_path=self.config.Stiffness_TRAIN_FLIST, config=self.config, AugFlag=transform1)
        self.test_datasets = Datasets(flist_path=self.config.Stiffness_TEST_FLIST, config=self.config, AugFlag=transform1)
        self.val_datasets = Datasets(flist_path=self.config.Stiffness_VAL_FLIST, config=self.config, AugFlag=transform2)
        if self.config.LOAD_MODEL:
            try: self.load()
            except Exception as e: print("Warning: load failed:", e)
        if self.config.mode == 'test':
            try: self.load()
            except Exception: pass
            self.test()

    def train(self):
        writer = SummaryWriter(log_dir=self.config.save_path, comment="Train")
        train_loader = DataLoader(dataset=self.train_dataset, batch_size=self.config.BATCH_SIZE,
                                  num_workers=self.config.NUM_WORKS, drop_last=False, shuffle=True, pin_memory=True)
        total_batch = len(train_loader)
        while self.epoch < self.config.EPOCH:
            self.epoch += 1
            print('====== Epoch: ' + str(self.epoch) + '======')
            bar = progressbar.ProgressBar(maxval=total_batch).start()

            losses_reg_train, losses_reg_test = RollingMeasure(), RollingMeasure()
            for index, batch in enumerate(train_loader):
                bar.update(index+1)
                imgs, force, _, _ = (item.to(self.config.DEVICE) for item in batch)
                F_pred = self.model(imgs)
                loss = torch.mean(torch.abs(F_pred - force))
                self.solver_Stiffness.zero_grad()
                loss.backward()
                self.solver_Stiffness.step()
                losses_reg_train(torch.mean(loss).data.cpu().numpy())
            losses_reg_test = self.validation(losses_reg_test)
            with open(self.config.save_path + 'loss_log.txt', 'a+') as f:
                f.write(f'Ep: {self.epoch} -- train_reg: {losses_reg_train.measure} -- test_reg: {losses_reg_test.measure}; -- LR: {self.solver_Stiffness.param_groups[0]["lr"]}\n')
            writer.add_scalars('losses_reg',
                               {'losses_reg': losses_reg_train.measure, 'losses_test': losses_reg_test.measure},
                               self.epoch)
            if losses_reg_test.measure < self.best_score:
                self.best_score = losses_reg_test.measure
                with open(self.config.save_path + 'loss_log.txt', 'a+') as f: f.write('saving...\n')
                self.save()

    def validation(self, losses_test):
        test_loader = DataLoader(dataset=self.val_datasets, batch_size=self.config.BATCH_SIZE,
                                 num_workers=self.config.NUM_WORKS, drop_last=True)
        self.model.eval()
        bar = progressbar.ProgressBar(maxval=len(test_loader)).start()
        with torch.no_grad():
            for index, batch in enumerate(test_loader):
                bar.update(index+1)
                imgs, force, _, _ = (item.to(self.config.DEVICE) for item in batch)
                F_pred = self.model(imgs)
                loss = torch.mean(torch.abs(F_pred - force))
                losses_test(torch.mean(loss).data.cpu().numpy())
        self.model.train()
        return losses_test

    def test(self):
        test_loader = DataLoader(dataset=self.test_datasets, batch_size=self.config.BATCH_SIZE,
                                 num_workers=self.config.NUM_WORKS, drop_last=True)
        self.model.eval()
        F_estim_list, F_gt_list, record = [], [], []
        bar = progressbar.ProgressBar(maxval=len(test_loader)).start()
        with torch.no_grad():
            for index, batch in enumerate(test_loader):
                bar.update(index+1)
                imgs, force, _, _ = (item.to(self.config.DEVICE) for item in batch)
                F_pred = self.model(imgs)
                F_pred_np, F_gt_np = F_pred.cpu().numpy(), force.cpu().numpy()
                F_estim_list.append(F_pred_np)
                F_gt_list.append(F_gt_np)
                for i in range(F_pred_np.shape[0]):
                    record.append(f"GT: {F_gt_np[i,0]:.4f}\t{F_gt_np[i,1]:.4f}\tPred: {F_pred_np[i,0]:.4f}\t{F_pred_np[i,1]:.4f}")

        if len(F_estim_list) == 0:
            print("No test samples found."); return

        F_estim, F_gt = np.vstack(F_estim_list), np.vstack(F_gt_list)

        txt_name = 'Raw_' + os.path.basename(self.config.Stiffness_TRAIN_FLIST).replace('.flist','') + '_' + os.path.basename(self.config.Stiffness_TEST_FLIST).replace('.flist','') + '.txt'
        with open(txt_name, 'w') as file:
            file.write('Ft_gt\tFn_gt\tFt_pred\tFn_pred\n')
            for i in range(F_estim.shape[0]):
                file.write(f"{F_gt[i,0]:.6f}\t{F_gt[i,1]:.6f}\t{F_estim[i,0]:.6f}\t{F_estim[i,1]:.6f}\n")

        mae_ft, mae_fn = mean_absolute_error(F_gt[:,0], F_estim[:,0]), mean_absolute_error(F_gt[:,1], F_estim[:,1])
        rmse_ft, rmse_fn = sqrt(mean_squared_error(F_gt[:,0], F_estim[:,0])), sqrt(mean_squared_error(F_gt[:,1], F_estim[:,1]))
        mae_overall = mean_absolute_error(F_gt.reshape(-1), F_estim.reshape(-1))
        rmse_overall = sqrt(mean_squared_error(F_gt.reshape(-1), F_estim.reshape(-1)))
        try:
            r2_ft, r2_fn = r2_score(F_gt[:,0], F_estim[:,0]), r2_score(F_gt[:,1], F_estim[:,1])
        except: r2_ft, r2_fn = float('nan'), float('nan')

        print("==========Test Results=============")
        print(f"MAE Ft: {mae_ft:.6f}, MAE Fn: {mae_fn:.6f}")
        print(f"RMSE Ft: {rmse_ft:.6f}, RMSE Fn: {rmse_fn:.6f}")
        print(f"MAE overall: {mae_overall:.6f}, RMSE overall: {rmse_overall:.6f}")
        print(f"R2 Ft: {r2_ft:.6f}, R2 Fn: {r2_fn:.6f}")
        print("==================================")

        # ============ 绘图 =============
        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        plt.plot(F_gt[:,0], label="Ft Ground Truth", color="blue")
        plt.plot(F_estim[:,0], label="Ft Prediction", color="red", alpha=0.7)
        plt.xlabel("Sample Index"); plt.ylabel("Ft")
        plt.legend(); plt.title("Ft Prediction vs Ground Truth")

        plt.subplot(1,2,2)
        plt.plot(F_gt[:,1], label="Fn Ground Truth", color="blue")
        plt.plot(F_estim[:,1], label="Fn Prediction", color="red", alpha=0.7)
        plt.xlabel("Sample Index"); plt.ylabel("Fn")
        plt.legend(); plt.title("Fn Prediction vs Ground Truth")

        plt.tight_layout()
        fig_path = os.path.join(self.config.save_path, "Pred_vs_GT.png")
        plt.savefig(fig_path, dpi=300)
        print(f"预测对比图已保存到: {fig_path}")

    def save(self):
        print('\nSaving ...\n')
        torch.save({'epoch': self.epoch,
                    'main_task_model': self.model.state_dict(),
                    'solver_main': self.solver_Stiffness.state_dict(),}, self.config.save_path + 'checkpoints.pt')
        if self.epoch == 1:
            argsDict = self.config.__dict__
            with open(self.config.save_path + 'setting.txt', 'w') as f:
                f.writelines('------------------ start ------------------\n')
                for eachArg, value in argsDict.items():
                    f.writelines(eachArg + ' : ' + str(value) + '\n')
                f.writelines('------------------- end -------------------')

    def load(self):
        model_path = self.config.load_path + 'checkpoints.pt'
        print('\nLoading ...\n', model_path)
        checkpoint = torch.load(model_path, map_location=self.config.DEVICE)
        self.model.load_state_dict(checkpoint['main_task_model'])
        self.solver_Stiffness.load_state_dict(checkpoint['solver_main'])
        self.epoch = checkpoint['epoch']
        print('Epoch: ' + str(self.epoch))


# ------------------ main ------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--Resnet_depth', type=str, default=50, choices=[10,18,34,50,101,152,200])
    parser.add_argument('--DEVICE', type=str, default='cuda')
    parser.add_argument('--LABEL_FLIST', type=str, default='/media/disk1/ylt/Force/label_hardness1.txt')
    parser.add_argument('--LR', type=float, default=1e-04)
    parser.add_argument('--NUM_WORKS', type=int, default=12)
    parser.add_argument('--EPOCH', type=int, default=2000)
    parser.add_argument('--BATCH_SIZE', type=int, default=32)
    parser.add_argument('--AugTrain', type=bool, default=True)
    parser.add_argument('--AugTest', type=bool, default=False)
    parser.add_argument('--AugVal', type=bool, default=True)
    parser.add_argument('--LOAD_MODEL', type=bool, default=False)
    parser.add_argument('--mode', type=str, default='train', choices=['train','test'])
    parser.add_argument('--sensor', type=str, default='sensor05')
    parser.add_argument('--flist', type=str, default='')

    args = parser.parse_args()
    args.DATA_PATH = 'session/'  # 数据根目录
    args.save_path = 'Res_Force3/Raw/' + args.sensor + '/'
    args.load_path = args.save_path
    args.Stiffness_TRAIN_FLIST = os.path.join('session', args.sensor, 'train_with_force.flist')
    args.Stiffness_VAL_FLIST = os.path.join('session', args.sensor, 'val_with_force.flist')
    args.Stiffness_TEST_FLIST = os.path.join('session', args.sensor, 'test_with_force.flist')

    if args.mode == 'train':
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        shutil.copy(sys.argv[0], args.save_path + '/copy.py')

    trainer = Trainer(args)
    if args.mode == 'train':
        trainer.train()
    elif args.mode == 'test':
        trainer.test()

