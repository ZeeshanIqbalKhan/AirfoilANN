from AirfoilModel import AirfoilModel
from AirfoilModel import DATASET
import datetime as dt

def myPrint(A,B):
    _A = '%11.5e'%(A[0]) + '\t' + '%11.5e'%(A[1]) + '\t' + '%11.5e'%(A[2])
    _B = '%11.5e'%(B[0]) + '\t' + '%11.5e'%(B[1]) + '\t' + '%11.5e'%(B[2])
    return _A + '\t' + _B


# Create Dataset
D410 = DATASET('D4_10', 'NACA4Digit_Dataset10Point.csv', zipfolder='Datasets.zip',
               TVT_ratio = [0.7,0.15,0.15], RANDOM_SEED = [42,30])
D410.SaveScaler(r'Models/')

MODEL_SHAPES = [[64,3],
                [64,32,3],
                [64,32,16,3],
                [64,32,16,8,3],
                [128,64,32,3],
                [256,128,64,3],
                [512,256,128,3],
                [1024,512,256,3]]

F = open(r'Results\Results_CASE-1,2 Wide vs. Deep.txt',"w")
F.write("// File Written on " + str(dt.datetime.now()) + "\n")

for ind,MODEL_SHAPE in enumerate(MODEL_SHAPES):
    AF = AirfoilModel(MODEL_SHAPE)
    AF.model_train_eval(D410, RUNS=5, BATCHSIZE=128, EPOCHS=50)
    AF.save_model(r'Models/')
    
    F.write(AF.MODEL_NAME + ':\t\t' + myPrint(AF.best_RMSE,AF.best_R2) + '\n')
    
    if(ind < len(MODEL_SHAPES) - 1):
        del AF 
    
F.write("\n// ------------------------------------------------- //\n");
F.close()

# Plots
# Plot history for Acc
AF.plot('acc',label='Accuracy')

# Plot history for loss
AF.plot('loss',label='MSE')

