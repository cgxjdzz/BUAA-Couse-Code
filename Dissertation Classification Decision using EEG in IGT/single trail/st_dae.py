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
num_epoches, start_decline_epochs =1, 30
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

class CnnModel(nn.Module):
    def __init__(self, num_classes = 5):
        super(CnnModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(4 * 250 * 31, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = x.view(-1, 4 * 250 * 31)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x

class LstmModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LstmModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True)
        self.fc1 = nn.Linear(hidden_size*2, 128)
        self.fc2 = nn.Linear(128, num_classes)
    def forward(self, x):
        x = x.permute(0,2,1)
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
            nn.Linear(128*3, 256*4),
            nn.ReLU(),
            nn.Linear(256*4, 256//2),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(256//2, 256*4),
            nn.ReLU(),
            nn.Linear(256*4, 128*3)
        )
        self.decoder.weight = self.encoder[-2].weight.T

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
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
    
        orin_model_path = orin_data_path+'/model_file'
        cnn_model = CnnModel()
        cnn_model.load_state_dict(torch.load(orin_model_path+ '/cnn/'+id+'_model_statedict.pth'))
        cnn_model.fc2 = nn.Sequential()
        lstm_model = LstmModel(input_size=63, hidden_size=128, num_layers=3, num_classes = 5)
        lstm_model.load_state_dict(torch.load(orin_model_path+ '/lstm/'+id+'_model_statedict.pth'))
        lstm_model.fc2 = nn.Sequential()
        tfm_model = TfmModel( num_classes=5)
        tfm_model.load_state_dict(torch.load(orin_model_path+ '/tfm/'+id+'_model_statedict.pth'))
        tfm_model.fc = nn.Sequential()
        tfm_model.eval()
        lstm_model.eval()
        cnn_model.eval()
        tfm_model = tfm_model.to(device)
        lstm_model = lstm_model.to(device)
        cnn_model=cnn_model.to(device)
        model = AutoEncoderModel()
        model = model.to(device)
        criterion = nn.MSELoss()
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
                tfm_output = tfm_model(X)
                concate_X = torch.cat([cnn_output, lstm_output,tfm_output], dim=1)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                outputs = model(concate_X)
                #outputs = outputs.to('cpu')
                loss = criterion(outputs, concate_X)
                loss.backward()
                running_loss += loss.item()
            print(f'epoch {epoch}: train loss: {running_loss}\n')
    
            if epoch % 5 == 0:
                model.eval()
                test_loss = 0.0
                #记录预测的y与真实的y
                test_max_classnum, test_y_decode = [], []
                for X, y in test_dataloader:
                    X = X.to(device)
                    cnn_output = cnn_model(X)
                    lstm_output = lstm_model(X)
                    tfm_output = tfm_model(X)
                    concate_X = torch.cat([cnn_output, lstm_output, tfm_output], dim=1)
                    outputs = model(concate_X)
                    #outputs = outputs.to('cpu')
                    loss = criterion(outputs, concate_X)
                    test_loss += loss.item()
                print(f'epoch {epoch}: test loss: {test_loss}\n')
                if epoch == num_epoches-1:
                    patient_data.append((patient, test_loss))    
        print('Finished Training')
        model_path = orin_data_path+'/model_file'+'/dae'
        torch.save(model, os.path.join(model_path,id+'_model.pt'))
        torch.save(model.state_dict(), os.path.join(model_path,id+'_model_statedict.pth'))
    for (patient, test_loss) in patient_data:
        print(f'patient {patient}: test loss: {test_loss}\n')




