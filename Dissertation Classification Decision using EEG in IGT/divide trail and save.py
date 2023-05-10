import mne
import pandas as pd
import os
#sfreq = 1000
for patient in range(2,26):

    if patient < 10:
        id = '00' + str(patient)
    else:
        id = '0' + str(patient)
    patient_path = 'E:/eeg_dataset/0226data/'+id
    if not os.path.exists(patient_path):
        os.makedirs(patient_path)
        os.makedirs(patient_path+'\A')
        os.makedirs(patient_path + '\B')
        os.makedirs(patient_path + '\C')
        os.makedirs(patient_path + '\D')
    # raw = mne.io.read_raw_eeglab("E:/eeg_dataset/1205data/002_1205no_epoch.set",uint16_codec='utf-8')
    raw = mne.io.read_epochs_eeglab("E:/eeg_dataset/1208data/"+id+"_epochs.set",uint16_codec="latin-1")
    # raw = mne.io.read_raw_cnt("D:/桌面/深度学习论文/eeg/EEG数据/EEG/002.cnt")
    label = pd.read_csv('E:/eeg_dataset/'+id+'_label.csv')
    label = label.iloc[:,1]
    data = raw._data
    time = raw.times
    channel = raw.ch_names
    for trail in range(400):
        trail_data = data[trail]
        trail_data = pd.DataFrame(trail_data,index=channel,columns=time)
        trail_label = label[trail]
        trail_data.to_csv(patient_path+'/'+trail_label+'/'+str(trail)+'.csv')
