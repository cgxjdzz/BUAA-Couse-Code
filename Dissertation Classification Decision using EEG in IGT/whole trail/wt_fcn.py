from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_fscore_support
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_fscore_support,accuracy_score
import torch
import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset, random_split,DataLoader
import torchvision
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import pickle
import random
import torch.optim as optim
from tsai.all import *
import re
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epoches, start_decline_epochs =61, 30
weight_decay = 20
batch_size = 1
k = 5
lag = 1
# labels_map = {     "A":0,    "B":1,    "C":2,    "D":3}
Noimpaired = [5, 6, 7, 10, 11, 12, 14, 19, 22, 23, 25]
Impaired = [2, 3, 4, 8, 9, 13, 15, 16, 18, 20, 21, 24]
orin_data_path = '/gs/home/dakezhang/eeg_data/whole_trail/0319data/'
#orin_data_path = 'E:/eeg_dataset/0319data/'

class EEGDataset(Dataset):
    def __init__(self, samples_list, labels, transform=True, target_transform=None):

        self.labels = torch.tensor(labels).long()
        self.samples =  samples_list
        self.transform = transform
        self.target_transform = target_transform
        self.num_classes = 2

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        # window = window *1e10
        X = self.samples[index]
        X = torch.tensor(X.values).float()
        if self.transform:
            X = (X)
        one_hot_y = torch.nn.functional.one_hot(self.labels[index], num_classes=self.num_classes).float()
        return X, one_hot_y


class TfmModel(nn.Module):
    def __init__(self, num_classes):
        super(TfmModel, self).__init__()
        self.tfm = TransformerModel(c_in=63, c_out=128)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # x = x.view(-1,1,15145200)
        x = x.permute(0, 2, 1)
        x = self.tfm(x)
        x = self.fc(x)
        return x


class CnnModel(nn.Module):
    def __init__(self, num_classes=2):
        super(CnnModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(3757200, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = x.view(-1, 3757200)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x


class LstmModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LstmModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True)
        self.fc1 = nn.Linear(hidden_size * 2, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        output = lstm_out[:, -1, :]
        x = self.fc1(output)
        x = torch.relu(x)
        x = self.fc2(x)
        return x


class AutoEncoderModel(nn.Module):

    def __init__(self):
        super(AutoEncoderModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(128 * 2, 256 * 4),
            nn.ReLU(),
            nn.Linear(256 * 4, 256 // 2),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(256 // 2, 256 * 4),
            nn.ReLU(),
            nn.Linear(256 * 4, 128 * 2)
        )
        self.decoder.weight = self.encoder[-2].weight.T

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class FcnModel(nn.Module):
    def __init__(self):
        super(FcnModel, self).__init__()
        self.fcn = nn.Sequential(
            nn.Linear(256//2, 96),
            nn.ReLU(),
            nn.Linear(96, 96),
            nn.ReLU(),
            nn.Linear(96, 2)
        )

    def forward(self, x):
        x = self.fcn(x)
        x = F.softmax(x, dim=1) 
        return x


if __name__ == '__main__':
    patient_data = []
    test_acc = []
    for patient in range(2, 26):
        if patient==17:
            continue

        print(f'patient : {patient}  \n')
        listdir =  os.listdir(orin_data_path)
        test_data = pd.read_csv(orin_data_path + str(patient) + '.csv', header=0,index_col=0).transpose()
        test_dataset = EEGDataset(samples_list=[test_data],labels=[1 if patient in Impaired else 0])
        listdir.remove(str(patient) + '.csv')

        train_data = []
        train_label = []
        for file_name in os.listdir(orin_data_path):
            slice_data= pd.read_csv(orin_data_path + file_name, header=0,index_col=0).transpose()
            train_data.append(slice_data)
            if int(file_name[:-4]) in Impaired:
                train_label.append(1)
            else:
                train_label.append(0)
        train_dataset = EEGDataset(samples_list=train_data, labels=train_label)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)

        id = str(patient)
        orin_model_path = '/gs/home/dakezhang/eeg_data/whole_trail/' + '/model_file'
        cnn_model = CnnModel()
        cnn_model.load_state_dict(torch.load(orin_model_path + '/cnn/' + id + '_model_statedict.pth'))
        cnn_model.fc2 = nn.Sequential()
        lstm_model = LstmModel(input_size=63, hidden_size=128, num_layers=3, num_classes=2)
        lstm_model.load_state_dict(torch.load(orin_model_path + '/lstm/' + id + '_model_statedict.pth'))
        lstm_model.fc2 = nn.Sequential()
        dae_model = AutoEncoderModel()
        dae_model.load_state_dict(torch.load(orin_model_path+ '/dae/'+id+'_model_statedict.pth'))
        dae_model.decoder = nn.Sequential()
        dae_model.to(device)
        #tfm_model = TfmModel(num_classes=2)
        #tfm_model.load_state_dict(torch.load(orin_model_path + '/tfm/' + id + '_model_statedict.pth'))
        #tfm_model.fc = nn.Sequential()
        #tfm_model.eval()
        lstm_model.eval()
        cnn_model.eval()
        dae_model.eval()
        #tfm_model = tfm_model.to(device)
        lstm_model = lstm_model.to(device)
        cnn_model = cnn_model.to(device)
        model = FcnModel()
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=weight_decay, eps=1e-08)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=0)
        for epoch in range(num_epoches):
            torch.backends.cudnn.enabled = False
            running_loss = 0.0
            # 记录预测的label与真实的label
            train_max_classnum, train_y_decode = [], []
            # Train
            model.train()
            for X, y in train_dataloader:
                X = X.to(device)
                cnn_output = cnn_model(X)
                lstm_output = lstm_model(X)
                concate_X = torch.cat([cnn_output, lstm_output], dim=1)
                dae_feature = dae_model(concate_X)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                outputs = model(dae_feature)
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

            max_acc = 0
            max_precision = []
            if epoch % 5 == 0:
                model.eval()
                test_loss = 0.0
                # 记录预测的y与真实的y
                test_max_classnum, test_y_decode = [], []
                for X, y in test_dataloader:
                    X = X.to(device)
                    cnn_output = cnn_model(X)
                    lstm_output = lstm_model(X)
                    concate_X = torch.cat([cnn_output, lstm_output], dim=1)
                    dae_feature = dae_model(concate_X)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    outputs = model(dae_feature)
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
                if accuracy > max_acc:
                    max_acc = accuracy
                    max_precision = precision
                if epoch == num_epoches - 1:
                    test_acc.append(accuracy)
                    patient_data.append((patient, test_loss, max_precision, max_acc))
        print('Finished Training')
        id = str(patient)
        model_path = '/gs/home/dakezhang/eeg_data/whole_trail/' + '/model_file' + '/fcn'
        torch.save(model, os.path.join(model_path, id + '_model.pt'))
        torch.save(model.state_dict(), os.path.join(model_path, id + '_model_statedict.pth'))
        for (patient, test_loss, precision, accuracy) in patient_data:
            print(f'patient {patient}: test loss: {test_loss}, class precision: {precision}, all_accuracy:{accuracy}\n')
    print(test_acc,'\n')
    print(sum(test_acc) / len(test_acc))