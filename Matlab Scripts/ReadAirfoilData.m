function [Data, ERR_FLAG] = ReadAirfoilData(path,label,N)
th = linspace(-pi/2,pi/2,N+2);
xn = sin(th'); xn = (xn+1)/2;

T = importdata([path label '.txt']);

id = find(T.data(:,1)==0);
pU = flip(T.data(1:id,:),1);
pL = T.data(id:end,:);

ERR_FLAG = abs(pU(end,2)) > 0.01;

yU = interp1(pU(:,1),pU(:,2),xn);
yL = interp1(pL(:,1),pL(:,2),xn);

yU = yU(2:end-1);
yL = yL(2:end-1);
Data = [yU' yL'];
end
