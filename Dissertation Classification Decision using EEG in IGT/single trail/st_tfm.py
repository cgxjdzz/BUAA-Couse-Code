from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_fscore_support,accuracy_score
import torch
import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset, random_split, DataLoader
import torchvision
from torch import nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from tsai.all import *
import pickle
import torchvision.transforms as transforms
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epoches, start_decline_epochs =61, 30
weight_decay = 1e-4
batch_size = 512
k = 5
orin_data_path = '/gs/home/dakezhang/eeg_data/0226newdata/single_trail/'

labels_map = {
     "A":0,
    "B":1,
    "C":2,
    "D":3,
}


class EEGDataset(Dataset):
    def __init__(self,samples, labels , transform=None, target_transform=None):
        if isinstance(labels[0], str):
            labels = np.array([labels_map[label] for label in labels])
        self.labels = torch.from_numpy(labels).long()
        self.samples = (torch.from_numpy(samples*1e7).float())
        self.transform = transform
        self.target_transform = target_transform
        self.num_classes = 5

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        mean = 0.0
        std = 1e-5
        # window = window *1e10
        X = self.samples[index] + torch.from_numpy(np.random.normal(mean, std, self.samples[index].shape)).float()
        one_hot_y = torch.nn.functional.one_hot(self.labels[index], num_classes=self.num_classes).float()
        return X, one_hot_y





class TfmModel(nn.Module):
    def __init__(self,  num_classes):
        super(TfmModel, self).__init__()
        self.tfm = TransformerModel(c_in =63, c_out = 128)
        self.fc = nn.Linear(128, num_classes)
    def forward(self, x):

        x = self.tfm(x)
        x = self.fc(x)
        return x


if __name__ == '__main__':

    patient_data = []
    for patient in range(2,6):
        if patient < 10:
            id = '00' + str(patient)
        else:
            id = '0' + str(patient)
        print(f'patient : {patient}  \n')            
        data_dir =  orin_data_path+ '0226data/' + id
    
        with open(data_dir + '/0227_data.pkl', 'rb') as f:
            samples = pickle.load(f)
            labels = pickle.load(f)
            file_names = pickle.load(f)
    
        train_split = [i for i, x in enumerate(file_names) if x <= 80]
        test_split = [i for i, x in enumerate(file_names) if x > 80 and x<=100]
        train_samples, test_samples = samples[train_split], samples[test_split]
        train_labels, test_labels = labels[train_split], labels[test_split]
        train_dataset, test_dataset = EEGDataset(train_samples,train_labels), EEGDataset(test_samples,test_labels)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)

    
        model = TfmModel(num_classes = 5)
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=weight_decay, eps=1e-08)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=0)
    
    
        for epoch in range(num_epoches):
            running_loss = 0.0
            # 记录预测的label与真实的label
            train_max_classnum, train_y_decode = [], []
            # Train
            model.train()
            for X, y in train_dataloader:
                X = X.to(device)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                outputs = model(X)
                outputs = outputs.to('cpu')
                loss = criterion(outputs, y)
                loss.backward()
                running_loss += loss.item()
                y_decode_slice = np.array(torch.argmax(y, dim=-1))
                max_classnum_slice = np.array(torch.argmax(outputs, dim=1))
                train_max_classnum.append(max_classnum_slice)
                train_y_decode.append(y_decode_slice)
            max_classnum, y_decode = np.concatenate(train_max_classnum), np.concatenate(train_y_decode)
            precision_recall_fscore = precision_recall_fscore_support(y_decode, max_classnum, average=None)
            precision = precision_recall_fscore[0]
            recall = precision_recall_fscore[1]
            f1 = precision_recall_fscore[2]
            accuracy = accuracy_score(y_decode, max_classnum)
            print(f'epoch {epoch}: train loss: {running_loss}, class precision: {precision}, all_accuracy:{accuracy}\n')
    
            if epoch % 5 == 0:
                model.eval()
                test_loss = 0.0
                #记录预测的y与真实的y
                test_max_classnum, test_y_decode = [], []
                for X, y in test_dataloader:
                    X = X.to(device)
                    # X = X.view(512, 1, 126, 126)
                    outputs = model(X)
                    outputs = outputs.to('cpu')
                    loss = criterion(outputs, y)
                    test_loss += loss.item()
                    y_decode_slice = np.array(torch.argmax(y, dim=-1))
                    max_classnum_slice = np.array(torch.argmax(outputs, dim=1))
                    test_max_classnum.append(max_classnum_slice)
                    test_y_decode.append(y_decode_slice)
    
                max_classnum, y_decode = np.concatenate(test_max_classnum), np.concatenate(test_y_decode)
                precision_recall_fscore = precision_recall_fscore_support(y_decode, max_classnum, average=None)
                precision = precision_recall_fscore[0]
                accuracy = accuracy_score(y_decode, max_classnum)
                print(f'epoch {epoch}: test loss: {test_loss}, class precision: {precision}, all_accuracy:{accuracy}\n')
                if epoch == num_epoches-1:
                    patient_data.append((patient, test_loss, precision, accuracy))
        print('Finished Training')
        model_path = orin_data_path+'/model_file'+'/tfm'
        torch.save(model, os.path.join(model_path,id+'_model.pt'))
        torch.save(model.state_dict(), os.path.join(model_path,id+'_model_statedict.pth'))
    for (patient, test_loss, precision, accuracy) in patient_data:
        print(f'patient {patient}: test loss: {test_loss}, class precision: {precision}, all_accuracy:{accuracy}\n')




