"""
===================== This file generates Javafoil Script =====================
Created on Sun Sep 13 23:06:56 2020

@author: Zeeshan Iqbal Khan
"""

import numpy as np
import os as os
import datetime as dt
#%%
flag = '5r' # 4  for NACA 4-Digit Series
           # 5  for NACA 5-Digit Series
           # 5r for NACA 4-Digit Series with Reflexed TE
           
FOLDER_NAME = 'NACA ' + flag + ' Digit'       
if(not(os.path.isdir(FOLDER_NAME))):
    os.mkdir(FOLDER_NAME)
    os.mkdir(FOLDER_NAME + '\\JavaFoilData')
    os.mkdir(FOLDER_NAME + '\\AirfoilData')
    
if(not(os.path.isdir('Scripts'))):    
    os.mkdir('Scripts')
#%% Airfoil Parameters     
PointCount = 101;
locMaxThickness = 0.3;

# Arrays of airfoil parameters
thickness    = np.arange(0.05, 0.40, 0.05);
locMaxCamber = np.arange(0.05, 0.85, 0.1);

           
if(flag=='4'):
    AirfoilType = 0; #NACA four digit
    P    = np.arange(0, 10, 1)  # Maximum Camber [%]
   
elif((flag=='5') or (flag=='5r')):
    P    = np.arange(0, 2, 0.2)    # Design Lift Coeff.
    if(flag=='5'):   
        AirfoilType = 2; # NACA five digit
        rflag = '0';
    elif(flag=='5r'):
        AirfoilType = 3; # NACA five digit with reflexed TE
        rflag = '1';

N_AFs = len(thickness)*len(locMaxCamber)*len(P);
#%% Flight Conditions 

# Array of mach number
MachNumber   = np.array([0.1,0.2,0.3]);

# Reylond number range
RE_min = 100000;
RE_max = 500000;
RE_del = 100000;

# Angle of Attack range
AOA_min = -10;
AOA_max =  10;
AOA_del =   1;

N_FCs = len(MachNumber)*len(np.arange(RE_min,RE_max+RE_del,RE_del))*len(np.arange(AOA_min,AOA_max+AOA_del,AOA_del)) 

#%% Path and Filenames

# Path to folders where data will be saved on execution of generated script file 
Data_dirJ = os.path.dirname(__file__) + '\\' + FOLDER_NAME + '\\JavaFoilData';
Data_dirA = os.path.dirname(__file__) + '\\' + FOLDER_NAME + '\\AirfoilData';

SaveFileType = '.txt'; # '.txt' or '.xml'

for n,thk in enumerate(thickness):
    # Path & Name of the script file to be generated
    fname = os.path.dirname(__file__) + '\\Scripts\\JavaFoilScript_' + flag +'_'+ str(int(thk*100)) + '.jfscript';
    
    #%% Write Script file 
    F = open(fname,"w")
    F.write("// File Written on " + str(dt.datetime.now()) + "\n")
    F.write("// Will Generate " + str(N_AFs/len(thickness)) + " Airfoils and " + str(N_AFs*3/len(thickness)) + " JavaFoil files." + "\n")
    
    for n,loc in enumerate(locMaxCamber):
        for n,pp in enumerate(P):
            # Airfoil Label 
            if(flag=='4'):
                label = 'NACA_'+ str(int(pp)) + str(int(loc*10)) + (str(int(thk*100))).zfill(2);
            elif((flag=='5') or (flag=='5r')):
                label = 'NACA_'+ str(int(pp*20/3)) + str(int(loc*10*2)) + rflag + (str(int(thk*100))).zfill(2);
                
            F.write("\n//" + label + "--------------------------------- // \n");
            
            #%% Create and save airfoil
            #CreateAirfoil(AirfoilType,PointCount,arrParams,ClosedFlag)
            F.write("Geometry.CreateAirfoil(" + str(AirfoilType) + "," 
                    + str(PointCount) + "," + str(int(thk*100)) + "," + str(int(locMaxThickness*100)) + "," 
                    + str(pp) + "," + str(int(loc*100)) + ",0,0,1)\n");
            
            #Save(FileName)
            filenameA = os.path.join(Data_dirA, label + SaveFileType);
            F.write("Geometry.Save(\"" + filenameA.replace("\\" , "\\\\") + "\"" + ")\n");
            
            #%% Analyze airfoil for each mach and save data
            for n,mach in enumerate(MachNumber):
                flabel = label + '_M0p' + str(int(mach*10)) + SaveFileType;
                filenameJ = os.path.join(Data_dirJ, flabel);
                # Mach Number
                F.write("\nOptions.MachNumber(" + str(mach) + ")\n");
            
                #Analyze(ReFirst,ReLast,ReDelta,AngleFirst,AngleLast,AngleDelta,TransUpper,TransLower,Roughness,AddToPlotsFlag)
                F.write("Polar.Analyze(" + str(RE_min) + "," + str(RE_max) + ","
                    + str(RE_del) + "," + str(AOA_min) + "," + str(AOA_max) + ","
                    + str(AOA_del) + ",100,100,0,0)\n");
            
                #Save(FileName)
                F.write("Polar.Save(\"" + filenameJ.replace("\\" , "\\\\") + "\"" + ")\n")
                
    #%% Close script file
    F.write("\n// --------------------------------- //\n"); 
    F.write("//        End of Script File         // \n");
    F.close()

print('All Script Files written')