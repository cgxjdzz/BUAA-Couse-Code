import mne
import pandas as pd
import numpy as np
for patient in range(2,26):

    cnt_path = 'D:/桌面/深度学习论文/eeg/EEG数据/EEG/'
    save_path = 'E:/eeg_dataset/0226data/'


    if patient < 10:
        id = '00' + str(patient)
    else:
        id = '0' + str(patient)

    raw = mne.io.read_raw_cnt(cnt_path+id+'.cnt', preload= True)

    #去除EOG
    raw.drop_channels(['VEO','HEO'])

    ch_name = ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'M2', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2']
    # 重参考
    raw.set_eeg_reference( ref_channels='average')

    #划分epoch
    events = mne.events_from_annotations(raw)
    epochs = mne.Epochs(raw, (events[0]), event_id ={'start':2}, tmin=-0.5, tmax=3,
                        baseline=(-0.5, 0), preload=True)

    #基线矫正
    epochs = epochs.apply_baseline()

    #重采样
    epochs.resample(sfreq=200)

    #带通滤波
    lowpass, highpass = 45, 0.1
    epochs.filter(highpass, lowpass)

    #时频分析
    epochs_spectrum = epochs.compute_psd(tmin = 0,tmax = 3.0)

    # X = epochs.get_data()

    mne.export.export_epochs(fname = save_path+id+'_epochs.set', epochs = epochs, overwrite=True)
    spectrum = (epochs_spectrum.get_data())
    np.save(save_path+id+'_epochs_spectrum.csv', spectrum)


