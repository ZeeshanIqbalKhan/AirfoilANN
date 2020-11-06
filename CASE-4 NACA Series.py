from AirfoilModel import AirfoilModel
from AirfoilModel import DATASET
import datetime as dt

def myPrint(A,B):
    _A = '%11.5e'%(A[0]) + '\t' + '%11.5e'%(A[1]) + '\t' + '%11.5e'%(A[2])
    _B = '%11.5e'%(B[0]) + '\t' + '%11.5e'%(B[1]) + '\t' + '%11.5e'%(B[2])
    return _A + '\t' + _B


# Create Dataset
D4 = DATASET('D4_10', 'Datasets.zip', 'NACA4Digit_Dataset10Point.csv',
               TVT_ratio = [0.7,0.15,0.15], RANDOM_SEED = [42,30])
D55r = DATASET('D5&5r_10', 'Datasets.zip', ('NACA5Digit_Dataset10Point.csv','NACA5rDigit_Dataset10Point.csv'),
               TVT_ratio = [0.7,0.15,0.15], RANDOM_SEED = [42,30])
DU = DATASET('D_all_10', 'Datasets.zip', ('NACA4Digit_Dataset10Point.csv','NACA5Digit_Dataset10Point.csv','NACA5rDigit_Dataset10Point.csv'),
               TVT_ratio = [0.7,0.15,0.15], RANDOM_SEED = [42,30])

D4.SaveScaler(r'Models/')
D55r.SaveScaler(r'Models/')
DU.SaveScaler(r'Models/')

MODEL_SHAPE = [512,256,128,3]

# Create & Train Models
F = open(r'Results\Results_CASE-4 NACA Series.txt',"w")
F.write("// File Written on " + str(dt.datetime.now()) + "\n")

AF4 = AirfoilModel(MODEL_SHAPE)
AF4.model_train_eval(D4, RUNS=5, BATCHSIZE=128, EPOCHS=50)
AF4.save_model(r'Models/')

AF55r = AirfoilModel(MODEL_SHAPE)
AF55r.model_train_eval(D55r, RUNS=5, BATCHSIZE=128, EPOCHS=50)
AF55r.save_model(r'Models/')
    
AFU = AirfoilModel(MODEL_SHAPE)
AFU.model_train_eval(DU, RUNS=5, BATCHSIZE=128, EPOCHS=50)
AFU.save_model(r'Models/')

RMSE_a, R2_a = AF4.evaluate_model(D55r)
F.write(D4.Label + ',\t' + D55r.Label + ':\t\t' + myPrint(RMSE_a,R2_a) + '\n')

RMSE_b, R2_b = AF55r.evaluate_model(D4)
F.write(D55r.Label + ',\t' + D4.Label + ':\t\t' + myPrint(RMSE_b,R2_b) + '\n')

RMSE_c, R2_c = AFU.evaluate_model(DU)
F.write(DU.Label + ',\t' + DU.Label + ':\t\t' + myPrint(RMSE_c,R2_c) + '\n')
 
F.write("\n// ------------------------------------------------- //\n");
F.close()


# Plots
# Plot history for Acc
AFU.plot('acc',label='Accuracy')

# Plot history for loss
AFU.plot('loss',label='MSE')

