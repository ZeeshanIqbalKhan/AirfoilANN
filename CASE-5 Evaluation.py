# from AirfoilModel import AirfoilModel
from AirfoilModel import DATASET
from AirfoilModel import r2score
import numpy as np
# import datetime as dt
import pandas as pd
import pickle as pkl
from keras.models import load_model
import matplotlib.pyplot as plt
from scipy.io import savemat
plt.style.use('seaborn-whitegrid')
%matplotlib qt


def EVAL(MDOEL, DS, SCALER):
    ReqData_1 = DS.Data.loc[((DS.Data['ReynoldsNumber'] == 250000) \
                           & (DS.Data['MachNumber'] == 0.25))]
    ReqData_2 = DS.Data.loc[((DS.Data['ReynoldsNumber'] == 200000) \
                           & (DS.Data['MachNumber'] == 0.20))]
        
    AoA = DS.Data['alpha'].loc[((DS.Data['ReynoldsNumber'] == 250000) \
                           & (DS.Data['MachNumber'] == 0.25))]    
    
    TestX_1 = SCALER.transform(ReqData_1.loc[:, 'yU_1':'alpha'])
    TestY_1 = ReqData_1.loc[:, 'Cl':'Cm']
    YPRED_1  = MDOEL.predict(TestX_1)
    
    TestX_2 = SCALER.transform(ReqData_2.loc[:, 'yU_1':'alpha'])
    TestY_2 = ReqData_2.loc[:, 'Cl':'Cm']
    YPRED_2  = MDOEL.predict(TestX_2)
    
    return YPRED_1, TestY_1, YPRED_2, TestY_2, AoA


# Create Dataset
NACA0045 = DATASET('NACA0045', r'Data\NACA0045.csv')
NACA2412 = DATASET('NACA2412', r'Data\NACA2412.csv')
NACA6408 = DATASET('NACA6408', r'Data\NACA6408.csv')
NACA136138 = DATASET('NACA136138', r'Data\NACA136138.csv')

scaler = pkl.load(open('Models\SCALER_D_all_10.pkl', 'rb'))
model = load_model('Models\MODEL_512.256.128.3_D_all_10.h5', custom_objects={'r2score':r2score})


YP_0045_1, YT_0045_1, YP_0045_2, YT_0045_2, AoA = EVAL(model,NACA0045,scaler)
YP_2412_1, YT_2412_1, YP_2412_2, YT_2412_2, _ = EVAL(model,NACA2412,scaler)
YP_6408_1, YT_6408_1, YP_6408_2, YT_6408_2, _ = EVAL(model,NACA6408,scaler)
YP_136138_1, YT_136138_1, YP_136138_2, YT_136138_2, _ = EVAL(model,NACA136138,scaler)

YT_0045_1 = YT_0045_1.to_numpy()
YT_2412_1 = YT_2412_1.to_numpy()
YT_6408_1 = YT_6408_1.to_numpy()
YT_136138_1 = YT_136138_1.to_numpy()

YT_0045_2 = YT_0045_2.to_numpy()
YT_2412_2 = YT_2412_2.to_numpy()
YT_6408_2 = YT_6408_2.to_numpy()
YT_136138_2 = YT_136138_2.to_numpy()

M_20 = np.ones(len(AoA))*0.20
M_25 = np.ones(len(AoA))*0.25

fig1 = plt.figure(figsize=(16, 14))
plt.rcParams['xtick.major.pad']='10'
plt.rcParams['ytick.major.pad']='10'
plt.rcParams['legend.fontsize']='12'
plt.plot(AoA, YT_0045_1[:,0], color='tab:red', marker='o', linestyle='-', linewidth=2, markersize=5)
plt.plot(AoA, YP_0045_1[:,0], color='tab:red', marker='o', linestyle='--', linewidth=2, markersize=5)
plt.plot(AoA, YT_2412_1[:,0], color='tab:blue', marker='D', linestyle='-', linewidth=2, markersize=5)
plt.plot(AoA, YP_2412_1[:,0], color='tab:blue', marker='D', linestyle='--', linewidth=2, markersize=5)
plt.plot(AoA, YT_6408_1[:,0], color='tab:orange', marker='<', linestyle='-', linewidth=2, markersize=5)
plt.plot(AoA, YP_6408_1[:,0], color='tab:orange', marker='<', linestyle='--', linewidth=2, markersize=5)
plt.plot(AoA, YT_136138_1[:,0], color='black', marker='*', linestyle='-', linewidth=2, markersize=7)
plt.plot(AoA, YP_136138_1[:,0], color='black', marker='*', linestyle='--', linewidth=2, markersize=7)
plt.xlim((-10, 10))
plt.ylim((-1.5, 2.5))
plt.xlabel(r'Angle of Attack ($\alpha$)',fontsize=24)
plt.ylabel('$C_{L}$',fontsize=24)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(['NACA 0045 (Actual)','NACA 0045 (Predicted)',\
            'NACA 2412 (Actual)','NACA 2412 (Predicted)',\
            'NACA 6408 (Actual)','NACA 6408 (Predicted)',\
            'NACA 136138 (Actual)','NACA 136138 (Predicted)'], frameon=True, prop={'size': 20})
# plt.grid(False)
plt.savefig('CL.eps', format='eps')

fig2 = plt.figure(figsize=(16, 14))
plt.rcParams['xtick.major.pad']='10'
plt.rcParams['ytick.major.pad']='10'
plt.rcParams['legend.fontsize']='12'
plt.plot(AoA, YT_0045_1[:,1], color='tab:red', marker='o', linestyle='-', linewidth=2, markersize=5)
plt.plot(AoA, YP_0045_1[:,1], color='tab:red', marker='o', linestyle='--', linewidth=2, markersize=5)
plt.plot(AoA, YT_2412_1[:,1], color='tab:blue', marker='D', linestyle='-', linewidth=2, markersize=5)
plt.plot(AoA, YP_2412_1[:,1], color='tab:blue', marker='D', linestyle='--', linewidth=2, markersize=5)
plt.plot(AoA, YT_6408_1[:,1], color='tab:orange', marker='<', linestyle='-', linewidth=2, markersize=5)
plt.plot(AoA, YP_6408_1[:,1], color='tab:orange', marker='<', linestyle='--', linewidth=2, markersize=5)
plt.plot(AoA, YT_136138_1[:,1], color='black', marker='*', linestyle='-', linewidth=2, markersize=7)
plt.plot(AoA, YP_136138_1[:,1], color='black', marker='*', linestyle='--', linewidth=2, markersize=7)
plt.xlim((-10, 10))
plt.ylim((0, 0.12))
plt.xlabel(r'Angle of Attack ($\alpha$)',fontsize=24)
plt.ylabel('$C_{D}$',fontsize=24)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(['NACA 0045 (Actual)','NACA 0045 (Predicted)',\
            'NACA 2412 (Actual)','NACA 2412 (Predicted)',\
            'NACA 6408 (Actual)','NACA 6408 (Predicted)',\
            'NACA 136138 (Actual)','NACA 136138 (Predicted)'], frameon=True, prop={'size': 20})    
# plt.grid(False)
plt.savefig('CD.eps', format='eps')

    
fig3 = plt.figure(figsize=(16, 14))
plt.rcParams['xtick.major.pad']='10'
plt.rcParams['ytick.major.pad']='10'
plt.rcParams['legend.fontsize']='12'
plt.plot(AoA, YT_0045_1[:,2], color='tab:red', marker='o', linestyle='-', linewidth=2, markersize=5)
plt.plot(AoA, YP_0045_1[:,2], color='tab:red', marker='o', linestyle='--', linewidth=2, markersize=5)
plt.plot(AoA, YT_2412_1[:,2], color='tab:blue', marker='D', linestyle='-', linewidth=2, markersize=5)
plt.plot(AoA, YP_2412_1[:,2], color='tab:blue', marker='D', linestyle='--', linewidth=2, markersize=5)
plt.plot(AoA, YT_6408_1[:,2], color='tab:orange', marker='<', linestyle='-', linewidth=2, markersize=5)
plt.plot(AoA, YP_6408_1[:,2], color='tab:orange', marker='<', linestyle='--', linewidth=2, markersize=5)
plt.plot(AoA, YT_136138_1[:,2], color='black', marker='*', linestyle='-', linewidth=2, markersize=7)
plt.plot(AoA, YP_136138_1[:,2], color='black', marker='*', linestyle='--', linewidth=2, markersize=7)
plt.xlim((-10, 10))
plt.ylim((-0.175, 0.075))
plt.xlabel(r'Angle of Attack ($\alpha$)',fontsize=24)
plt.ylabel('$C_{M}$',fontsize=24)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(['NACA 0045 (Actual)','NACA 0045 (Predicted)',\
            'NACA 2412 (Actual)','NACA 2412 (Predicted)',\
            'NACA 6408 (Actual)','NACA 6408 (Predicted)',\
            'NACA 136138 (Actual)','NACA 136138 (Predicted)'], frameon=True, prop={'size': 20})
# plt.grid(False)
plt.savefig('CM.eps', format='eps')

def RMSE(y_true, y_pred):
    temp = np.sqrt(np.mean(np.square(y_true - y_pred),axis=0))
    return temp

from sklearn.metrics import r2_score

a = r2_score(YT_0045_1[:,0], YP_0045_1[:,0])  

# savemat('PRED_NACA0045.mat',mdict=df_0045_P.to_dict())