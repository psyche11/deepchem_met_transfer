import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

samp_nums = [1000, 3000, 6000, 10000, 30000, 60000]
max_accuracy=[]
min_accuracy=[]
mean_accuracy=[]
max_precision=[]
min_precision=[]
mean_precision=[]
max_roc=[]
min_roc=[]
mean_roc=[]
max_prc=[]
min_prc=[]
mean_prc=[]
max_recall=[]
min_recall=[]
mean_recall=[]


for i in samp_nums:
    test_data = pd.read_csv('deepchem_train.csv')
    test_data_samp = test_data[test_data['samp_num']==i]
    
    #samp_std1 = test_data_samp['accuracy_score'].std()
    samp_mean1 = test_data_samp['accuracy_score'].mean()
    samp_max1 = test_data_samp['accuracy_score'].max()
    samp_min1 = test_data_samp['accuracy_score'].min()
    max_accuracy.append(samp_max1)
    min_accuracy.append(samp_min1)
    mean_accuracy.append(samp_mean1)
    
    #samp_std2 = test_data_samp['precision_score'].std()
    samp_mean2 = test_data_samp['precision_score'].mean()
    samp_max2 = test_data_samp['precision_score'].max()
    samp_min2 = test_data_samp['precision_score'].min()
    max_precision.append(samp_max2)
    min_precision.append(samp_min2)
    mean_precision.append(samp_mean2)
    
    #samp_std3 = test_data_samp['roc_auc_score'].std()
    samp_mean3 = test_data_samp['roc_auc_score'].mean()
    samp_max3 = test_data_samp['roc_auc_score'].max()
    samp_min3 = test_data_samp['roc_auc_score'].min()
    max_roc.append(samp_max3)
    min_roc.append(samp_min3)
    mean_roc.append(samp_mean3)
    
    #samp_std4 = test_data_samp['prc_auc_score'].std()
    samp_mean4 = test_data_samp['prc_auc_score'].mean()
    samp_max4 = test_data_samp['prc_auc_score'].max()
    samp_min4 = test_data_samp['prc_auc_score'].min()
    max_prc.append(samp_max4)
    min_prc.append(samp_min4)
    mean_prc.append(samp_mean4)
    
    #samp_std5 = test_data_samp['recall_score'].std()
    samp_mean5 = test_data_samp['recall_score'].mean()
    samp_max5 = test_data_samp['recall_score'].max()
    samp_min5 = test_data_samp['recall_score'].min()
    max_recall.append(samp_max5)
    min_recall.append(samp_min5)
    mean_recall.append(samp_mean5)

nsamp_nums = [1,2,3,4,5,6]
### Figure 1 ###
plt.figure(1,figsize=(10,8))
plt.title('Train Dataset')
plt.xlabel('Sample numbers(k)')
plt.ylabel('Prediction indexes(%)')
   
plt.plot(nsamp_nums, mean_accuracy, color='red',label='accuracy')
plt.plot(nsamp_nums, mean_precision, color='blue', label='precision')
plt.plot(nsamp_nums, mean_roc, color='green', label='roc_auc')
plt.plot(nsamp_nums,  mean_prc, color='cyan', label='prc_auc')
plt.plot(nsamp_nums,  mean_recall, color='magenta', label='recall')

'''plt.fill_between(nsamp_nums, max_accuracy, min_accuracy, alpha=0.25, facecolor='red')
plt.fill_between(nsamp_nums, max_precision, min_precision, alpha=0.25, facecolor='blue')
plt.fill_between(nsamp_nums, max_roc, min_roc, alpha=0.25, facecolor='green')
plt.fill_between(nsamp_nums, max_prc, min_prc, alpha=0.25, facecolor='cyan')
plt.fill_between(nsamp_nums, max_recall, min_recall, alpha=0.25, facecolor='magenta')'''

plt.legend(loc='lower left')

plt.xticks([1,2,3,4,5,6],[1,3,6,10,30,60])

plt.savefig('train_figure_total.png',dpi=300,alpha=0.3)

### Figure 2 ###
plt.figure(2,figsize=(14,10))
plt.suptitle('Train Dataset')

plt.subplot(231)
plt.xlabel('Sample numbers(k)')
plt.ylabel('Prediction accuracy(%)')
plt.plot(nsamp_nums, mean_accuracy, color='red',label='accuracy')
plt.fill_between(nsamp_nums, max_accuracy, min_accuracy, alpha=0.25, facecolor='red')
plt.legend(loc='lower left')
plt.xticks([1,2,3,4,5,6],[1,3,6,10,30,60])

plt.subplot(232)
plt.xlabel('Sample numbers(k)')
plt.ylabel('Prediction precision(%)')
plt.plot(nsamp_nums, mean_precision, color='blue', label='precision')
plt.fill_between(nsamp_nums, max_precision, min_precision, alpha=0.25, facecolor='blue')
plt.legend(loc='lower left')
plt.xticks([1,2,3,4,5,6],[1,3,6,10,30,60])

plt.subplot(233)
plt.xlabel('Sample numbers(k)')
plt.ylabel('Prediction roc_auc(%)')
plt.plot(nsamp_nums, mean_roc, color='green', label='roc_auc')
plt.fill_between(nsamp_nums, max_roc, min_roc, alpha=0.25, facecolor='green')
plt.legend(loc='lower left')
plt.xticks([1,2,3,4,5,6],[1,3,6,10,30,60])

plt.subplot(234)
plt.xlabel('Sample numbers(k)')
plt.ylabel('Prediction prc_auc(%)')
plt.plot(nsamp_nums, mean_prc, color='cyan', label='prc_auc')
plt.fill_between(nsamp_nums, max_prc, min_prc, alpha=0.25, facecolor='cyan')
plt.legend(loc='lower left')
plt.xticks([1,2,3,4,5,6],[1,3,6,10,30,60])

plt.subplot(235)
plt.xlabel('Sample numbers(k)')
plt.ylabel('Prediction recall(%)')
plt.plot(nsamp_nums, mean_recall, color='magenta', label='recall')
plt.fill_between(nsamp_nums, max_recall, min_recall, alpha=0.25, facecolor='magenta')
plt.legend(loc='lower left')
plt.xticks([1,2,3,4,5,6],[1,3,6,10,30,60])


plt.savefig('train_figure_sep.png',dpi=300,alpha=0.3)
    
    



