import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
orin_path = 'E:/eeg_dataset/0226data/'
all_df = pd.DataFrame()
all_df_block = pd.DataFrame()
score_mapping = {'A':-1,'B':-1,'C':1,'D':1}
Noimpaired = []
Belowaver = []
Impaired = []
for patient in range(2,26):
    if patient < 10:
        id = '00' + str(patient)
    else:
        id = '0' + str(patient)
    folder_path = orin_path+id+'/'
    col_names = ['A', 'B', 'C', 'D']

    # 创建一个空的DataFrame
    df = pd.DataFrame()

    # 循环遍历文件夹和文件，并将每个文件读取为一个DataFrame，并将它们添加到主DataFrame中
    for col_name in col_names:
        col_path = os.path.join(folder_path, col_name)
        file_list = sorted([f for f in os.listdir(col_path) if f.endswith('.csv')])
        time_list = [int(x.split('.')[0]) for x in file_list]
        label_list = [col_name] * len(time_list)
        # if col_name in ['A', 'B']:
        #     label_list = [0] * len(time_list)
        # else:
        #     label_list = [1] * len(time_list)
        file_df = pd.DataFrame({'time':time_list,'label':label_list})
        df = pd.concat([df, file_df])
    df.sort_values(by=['time'], inplace=True)
    df = df.set_index('time')
    df = df.reset_index()
    df_block = pd.DataFrame()
    score = 0
    for time in range(0,df.shape[0],10):
        label_block = df['label'][time:time+10]
        label_block = label_block.replace(score_mapping)
        score += sum(label_block)
        new_row = {'time': time, 'score': score}

        df_block = df_block.append(new_row, ignore_index=True)
    # 按数字列排序DataFrame

    df['patient'] = [patient]*df.shape[0]
    # 将DataFrame转换为array
    # array = df[col_names].to_numpy()
    all_df = pd.concat([all_df, df])
    plt.plot(df_block['time'], df_block['score'])
    plt.title( str(patient) +' score')
    plt.xlabel('time')
    plt.ylabel('score')
    plt.savefig(str(patient) + ' NET score.png')
    plt.clf()

    if list(df_block['score'])[-1]>=45:
        Noimpaired.append(patient)
    elif list(df_block['score'])[-1]>=40:
        Belowaver.append(patient)
    else :
        Impaired.append(patient)
print(Noimpaired,Belowaver,Impaired)
print(all_df)

# matrix = all_df.pivot(index='patient', columns='time', values='label')
#
# plt.figure(figsize=(40, 8))
# sns.set(font_scale=1.2)
# sns.heatmap(matrix, cmap='coolwarm', linewidths=0.5,yticklabels=True)
#
# # 设置坐标轴标签
# # plt.xlabel('Time Label')
# plt.ylabel('Patient')
#
# # 添加空白行
# plt.yticks(rotation=0)
# plt.yticks(fontsize=5)
# plt.xticks(fontsize=0)
# # plt.axhline(y=0, color='k', linewidth=2)
# # plt.axhline(y=matrix.shape[0], color='k', linewidth=2)
# plt.tight_layout()
#
# # 显示图形
# plt.show()
# # plt.savefig('E:/eeg_dataset/0226data/four_class_label_distribution.pdf', dpi=150, bbox_inches='tight')