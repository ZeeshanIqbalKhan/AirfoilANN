clear; close all; clc
% SERIES = 'NACA 4 Digit';
% SERIES = 'NACA 5 Digit';
SERIES = 'NACA 5r Digit';

N = 15;

pathJ = sprintf('%s\\Data\\%s\\JavaFoilData\\',cd,SERIES);
pathA = sprintf('%s\\Data\\%s\\AirfoilData\\',cd,SERIES);
pathS = sprintf('%s\\Data\\%s\\CombinedData_%02i\\',cd,SERIES,N);
if(exist(sprintf('%s\\Data\\%s\\CombinedData_%02i',cd,SERIES,N),'dir'))
    rmdir(sprintf('%s\\Data\\%s\\CombinedData_%02i',cd,SERIES,N),'s')
end
mkdir(sprintf('%s\\Data\\%s\\CombinedData_%02i',cd,SERIES,N))

F = dir([pathJ '*.txt']);
F = {F.name}';
temp = regexp(F,'NACA_\d+','match');
files = unique([temp{:}]');
HeadersJ = {'ReynoldsNumber','MachNumber','alpha','Cl','Cd','Cm'};
for k=1:N
    HU{k} = sprintf('yU_%i',k);
    HL{k} = sprintf('yL_%i',k);
end
HeadersA = [HU HL];
ColHeads = [HeadersA HeadersJ];
nF=numel(files);
ERR_COUNT = 0;
for k=1:nF
    label = files{k};
    try
        [DataJ,Headers] = ReadJavaFoilData(pathJ,label);
        JAV_ERR = 0;
    catch ME
        warning('Error: Unable to read Javafoil data of %s so ignored...\n',label);
        JAV_ERR = 1;
    end
    if ~all(cellfun(@(x,y)(strcmp(x,y)),Headers,HeadersJ))
        error('Headers mismatch in Javafoil data');
    end
    
    [DataA, ERR_FLAG] = ReadAirfoilData(pathA,label,N);
    
    if(or(ERR_FLAG,JAV_ERR))
        ERR_COUNT = ERR_COUNT + 1;
        warning('Error in %s so ignored...\n',label);
    else
        DATA = [repmat(DataA,size(DataJ,1),1), DataJ];
        T = array2table(DATA);
        T.Properties.VariableNames = ColHeads;
        writetable(T,[pathS label '.csv']);
        fprintf('[%3.0f/%3.0f] %s.csv Generated...\n',k-ERR_COUNT,nF-ERR_COUNT,label);
    end
end
