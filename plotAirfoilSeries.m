clearvars; close all; clc
D(1) = load('NACA4Digit.mat');
D(2) = load('NACA5Digit.mat');
D(3) = load('NACA5rDigit.mat');
%%
for k=1:numel(D)
    X{k} = D(k).XL(:,1);
    for i=1:size(D(k).XL,2)
        YYL(:,i)=interp1(D(k).XL(:,i),D(k).YL(:,i),X{k});
        YYU(:,i)=interp1(D(k).XU(:,i),D(k).YU(:,i),X{k});
    end
    YmaxL{k} = max(YYL,[],2);
    YminL{k} = min(YYL,[],2);
    YmaxU{k} = max(YYU,[],2);
    YminU{k} = min(YYU,[],2);
end
%%
LABEL = {'NACA 4 Digit','NACA 5 Digit','NACA 5 Digit Reflexed'};
figure(1),clf
for k = 1:numel(D)
subplot(numel(D),1,k)
fill([X{k};flip(X{k})],[YmaxL{k};flip(YminL{k})],'b','FaceAlpha',0.2),hold on
fill([X{k};flip(X{k})],[YmaxU{k};flip(YminU{k})],'g','FaceAlpha',0.2),grid on
title(LABEL{k})
end

for k = 1:numel(D)
ID{k} = ~(abs(D(k).YU(end,:)) > 0.01);
ERR{k} = D(k).NAME(~ID{k})';
figure(1+k),plot(D(k).XL(:,ID{k}),D(k).YL(:,ID{k}),D(k).XU(:,ID{k}),D(k).YU(:,ID{k})),title(LABEL{k}),grid on
end
%%
% 5 Digit
EN5 = cellfun(@(x)regexp(x,'\d*','match'),ERR{2});
thickness5 = cellfun(@(x)str2double(x(end-1:end)),EN5);

% 5r Digit
EN5r = cellfun(@(x)regexp(x,'\d*','match'),ERR{3});
thickness5r = cellfun(@(x)str2double(x(end-1:end)),EN5r);
