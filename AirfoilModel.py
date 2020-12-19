import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras import backend as K
from sklearn.metrics import r2_score
import pickle as pkl
import numpy as np
from zipfile import ZipFile
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')


def RMSE(y_true, y_pred):
    temp = np.sqrt(np.mean(np.square(y_true.to_numpy() - y_pred),axis=0))
    return temp

def R2(y_true, y_pred):
    return r2_score(y_true, y_pred, multioutput='raw_values')

def r2score(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred )) 
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

#%% Class to Create and train ANN Model
class AirfoilModel():
    def __init__(self,MODEL_SHAPE):
         self.hist = dict();
         self.model_array = [];
         self.ValResults = [];
         self.TestResults = [];
         self.losses = []; 
         self.TestR2scores = [];
         self.TestRMSE = [];
         self.MODEL_SHAPE = MODEL_SHAPE 
         
    def model_train_eval(self, DATASET, RUNS=5, BATCHSIZE=128, EPOCHS=50):
        self.Dataset = DATASET
        inputShape = DATASET.InputShape
        TrainX, TrainY, ValX, ValY, TestX, TestY =  DATASET.getTrainValTest()
        
        for kk in np.arange(0,RUNS):
            model = self._create_model(inputShape)
            earlyStopping = EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='min')
            reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, epsilon=1e-4, mode='min')
            model.summary()
            history = model.fit(TrainX, TrainY, validation_data=(ValX,ValY),
                            epochs=EPOCHS, batch_size=BATCHSIZE,
                            callbacks=[earlyStopping, reduce_lr_loss])
     
            trRes   = model.evaluate(TrainX, TrainY, batch_size=BATCHSIZE)
            valRes  = model.evaluate(  ValX,   ValY, batch_size=BATCHSIZE)
            testRes = model.evaluate( TestX,  TestY, batch_size=BATCHSIZE)    
            ypred   = model.predict(TestX)
            testR2  = R2(TestY,ypred);
            test_rmse = RMSE(TestY,ypred);
            new_loss = testRes[0];
            tvtloss = [trRes[0],valRes[0],testRes[0]];
            tvtR2   = [trRes[3],valRes[3],testRes[3]];
            
            self.hist[kk] = history.history;
            self.model_array.append(model);
            self.ValResults.append(valRes);
            self.TestResults.append(testRes);
            self.losses.append(new_loss);
            self.TestR2scores.append(testR2);
            self.TestRMSE.append(test_rmse);
        
            if(kk == 0):
                self.best_loss  = new_loss;
                self.best_model = model;
                self.best_hist  = history.history;
                self.best_R2    = testR2;
                self.best_RMSE  = test_rmse;
                self.best_tvtloss = tvtloss;
                self.best_tvtR2   = tvtR2;
            
            if(new_loss < self.best_loss):
                self.best_loss  = new_loss;
                self.best_model = model;
                self.best_hist  = history.history;
                self.best_R2    = testR2;
                self.best_RMSE  = test_rmse;
                self.best_tvtloss = tvtloss;
                self.best_tvtR2   = tvtR2;
        
            del model, earlyStopping, reduce_lr_loss, history, trRes, valRes, testRes,
            ypred, testR2, test_rmse, new_loss, tvtloss, tvtR2
            
    def save_model(self,path):
        # save paramters and model
        objects = (self.best_hist, self.ValResults, self.TestResults, self.losses, self.TestR2scores, self.TestRMSE,
                   self.best_loss, self.best_R2, self.best_RMSE)   
        pkl.dump(objects,open(path + 'DATA_' + self.MODEL_NAME + '_' + self.Dataset.Label + '.pkl', 'wb'))
        self.best_model.save(path + 'MODEL_' + self.MODEL_NAME + '_' + self.Dataset.Label + '.h5')

    def _create_model(self, inputShape):

        ADAM = optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        LAYERS = [];
        self.MODEL_NAME  = ''
        
        for jj,N_neuron in enumerate(self.MODEL_SHAPE):
            if(jj == 0):
                LAYERS.append(Dense(N_neuron, activation='relu', name='layer1', input_dim=inputShape))
            elif(jj < len(self.MODEL_SHAPE) - 1):
                LAYERS.append(Dense(N_neuron, activation='relu', name='layer'+str(jj+1)))
            else:
                LAYERS.append(Dense(3, name='output'))
            if(jj < len(self.MODEL_SHAPE) - 1):
                self.MODEL_NAME = self.MODEL_NAME + str(N_neuron) + '.'
            else:
                self.MODEL_NAME = self.MODEL_NAME + str(N_neuron)      
        model = Sequential(LAYERS)
        model.compile(optimizer = ADAM, loss = 'mean_squared_error',
                      metrics=['acc','mse', r2score])
        return model        
    
    def evaluate_model(self, DS):
        _TestX = self.Dataset.scaler.transform(DS.Data.loc[:, 'yU_1':'alpha'])
        _TestY = DS.Data.loc[:, 'Cl':'Cm']
        ypred  = self.best_model.predict(_TestX)
        _R2    = R2(_TestY,ypred)
        _RMSE  = RMSE(_TestY,ypred)
        return _RMSE, _R2
    
    def plot(self,key,label='label'):
        plt.plot(self.best_hist[key],'k',linewidth=2)
        plt.plot(self.best_hist['val_' + key],'b',linewidth=2)
        for k,mm in enumerate(self.model_array):
            plt.plot(self.hist[k][key],':k',linewidth=0.5)
            plt.plot(self.hist[k]['val_' + key],':b',linewidth=0.5)
        plt.title('Model ' + label)
        plt.ylabel(label)
        plt.xlabel('Epoch')
        plt.legend(['Train','Validation'],loc='best')
        plt.show()

#%% Class to load and scale datasets 
class DATASET():
    def __init__(self, DataLabel, filename, zipfolder = '', TVT_ratio = [0.7,0.15,0.15], RANDOM_SEED = [42,30]):
        assert sum(TVT_ratio) == 1
        self.Label = DataLabel
        self._zip_path = zipfolder
        if(zipfolder != ''):
            zf = ZipFile(zipfolder) 
            
        if(type(filename)==str):
            self._file_name = filename
            if(zipfolder == ''):
                self.Data = pd.read_csv(filename)
            else:
                self.Data = pd.read_csv(zf.open(filename))
                
        elif(type(filename)==tuple):
            self._file_name = []
            temp = []
            for n,file in enumerate(filename):
                self._file_name.append(file)
                if(zipfolder == ''):
                    _df = pd.read_csv(file, index_col=None, header=0)
                else:
                    _df = pd.read_csv(zf.open(file), index_col=None, header=0)
                    
                temp.append(_df)
            
            df = pd.concat(temp, axis=0, ignore_index=True)
            self.Data = df.dropna().reset_index().drop(columns=['index'])
            
        train, temp = train_test_split(self.Data, test_size=TVT_ratio[1]+TVT_ratio[2], random_state=RANDOM_SEED[0], shuffle=True)
        _ratio = TVT_ratio[2]/(TVT_ratio[1]+TVT_ratio[2]) 
        val, test   = train_test_split(temp, test_size=_ratio, random_state=RANDOM_SEED[1], shuffle=True)
        
        self.TrainX = train.loc[:, 'yU_1':'alpha']
        self.TrainY = train.loc[:, 'Cl':'Cm']
        self.ValX = val.loc[:, 'yU_1':'alpha']
        self.ValY = val.loc[:, 'Cl':'Cm']
        self.TestX = test.loc[:, 'yU_1':'alpha']
        self.TestY = test.loc[:, 'Cl':'Cm']
        
        self._ScaleInputs()
            
        self.InputShape = self.TrainX.shape[1]
        
    def _ScaleInputs(self):
        self.scaler = StandardScaler().fit(self.TrainX)
        self.TrainX = self.scaler.transform(self.TrainX)
        self.ValX   = self.scaler.transform(self.ValX)
        self.TestX  = self.scaler.transform(self.TestX)
        
    def SaveScaler(self, path):
        pkl.dump(self.scaler,open(path + 'SCALER_' + self.Label + '.pkl', 'wb'))
    
    def getTrainValTest(self):
        return self.TrainX, self.TrainY, self.ValX, self.ValY, self.TestX, self.TestY 

