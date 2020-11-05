from AirfoilModel import AirfoilModel
from AirfoilModel import DATASET
import datetime as dt

def myPrint(A,B):
    _A = '%11.5e'%(A[0]) + '\t' + '%11.5e'%(A[1]) + '\t' + '%11.5e'%(A[2])
    _B = '%11.5e'%(B[0]) + '\t' + '%11.5e'%(B[1]) + '\t' + '%11.5e'%(B[2])
    return _A + '\t' + _B


# Create Dataset
D405 = DATASET('D4_05', 'Datasets.zip', 'NACA4Digit_Dataset05Point.csv',
               TVT_ratio = [0.7,0.15,0.15], RANDOM_SEED = [42,30])
D410 = DATASET('D4_10', 'Datasets.zip', 'NACA4Digit_Dataset10Point.csv',
               TVT_ratio = [0.7,0.15,0.15], RANDOM_SEED = [42,30])
D415 = DATASET('D4_15', 'Datasets.zip', 'NACA4Digit_Dataset15Point.csv',
               TVT_ratio = [0.7,0.15,0.15], RANDOM_SEED = [42,30])

D405.SaveScaler(r'Models/')
D410.SaveScaler(r'Models/')
D415.SaveScaler(r'Models/')

DATA = [D405, D410, D415]

MODEL_SHAPE = [512,256,128,3]
#%%
F = open(r'Results\Results_CASE-3 Datasets.txt',"w")
F.write("// File Written on " + str(dt.datetime.now()) + "\n")

for ind, DS in enumerate(DATA):
    AF = AirfoilModel(MODEL_SHAPE)
    AF.model_train_eval(DS, RUNS=1, BATCHSIZE=128, EPOCHS=1)
    AF.save_model(r'Models/')
    
    F.write(DS.Label + ':\t\t' + myPrint(AF.best_RMSE,AF.best_R2) + '\n')
    
    if(ind < len(DATA) - 1):
        del AF 
    
F.write("\n// ------------------------------------------------- //\n");
F.close()

# Plots
# Plot history for Acc
AF.plot('acc',label='Accuracy')

# Plot history for loss
AF.plot('loss',label='MSE')

