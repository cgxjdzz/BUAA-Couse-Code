import pandas as pd
import os
# Event_label = {34:'A',35:'B', 31:'C', 32:'D' }
Event_label = {44:'A',45:'B', 41:'C', 42:'D' }

for i in range(2,26):
    if i < 10:
        id = '00'+str(i)
    else:
        id = '0' + str(i)
    event_data = pd.read_table('E:/eeg_dataset/'+id+'_event.txt')
    event = event_data.iloc[:,1]
    label = []
    for event_slice in event:
        # if event_slice in [31,32,34,35]:
        #     label.append(Event_label[event_slice])
        if event_slice in [44,45,41,42]:
            label.append(Event_label[event_slice])
    label = pd.DataFrame(label)
    if len(label) < 400:
        print(id)
    label.to_csv('E:/eeg_dataset/'+id+'_label.csv')
