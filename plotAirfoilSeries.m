clearvars; close all; clc
D(1) = load('NACA4Digit.mat');
D(2) = load('NACA5Digit.mat');
D(3) = load('NACA5rDigit.mat');

D(4).NAME = [D(1).NAME D(2).NAME D(3).NAME];
D(4).SERIES = 'NACA 4,5 Digit [Comb.]';
D(4).XL = [D(1).XL D(2).XL D(3).XL];
D(4).XU = [D(1).XU D(2).XU D(3).XU];
D(4).YL = [D(1).YL D(2).YL D(3).YL];
D(4).YU = [D(1).YU D(2).YU D(3).YU];

D([1 2 3]) = [];

% Load Test Airfoils
Airfoils = {'NACA2412','NACA0045','NACA6408','NACA136138'};
for i = 1:numel(Airfoils)
    A = load(['Data\' Airfoils{i} '.mat']);
    AF{i} = A.(Airfoils{i});
end

N = 10;
th = linspace(-pi/2,pi/2,N+2);
xn = sin(th'); xn = (xn+1)/2;

for i = 1:numel(Airfoils)
    id = find(AF{i}(:,1)==0);
    AF_yu{i} = [0;interp1(AF{i}(1:id,1),AF{i}(1:id,2),xn(2:end-1));0];
    AF_yl{i} = [0;interp1(AF{i}(id:end,1),AF{i}(id:end,2),xn(2:end-1));0];
end

for k=1:numel(D)
    ID{k} = ~(abs(D(k).YU(end,:)) > 0.01);
    ERR{k} = D(k).NAME(~ID{k})';
    
    X{k} = D(k).XL(:,1);
    for i=1:size(D(k).XL,2)
        Y(k).XL(:,i)=xn;
        Y(k).XU(:,i)=xn;
        Y(k).YL(:,i)=[0;interp1(D(k).XL(:,i),D(k).YL(:,i),xn(2:end-1));0];
        Y(k).YU(:,i)=[0;interp1(D(k).XU(:,i),D(k).YU(:,i),xn(2:end-1));0];
    end
end
%%
CLR = {'b','g','k','r'};
figure(100),clf,set(gcf,'Position',[180 120 1000 500])
for k = 1:numel(D)
XX = reshape([D(k).XU(:,ID{k});D(k).XL(:,ID{k})],[],1);
YY = reshape([D(k).YU(:,ID{k});D(k).YL(:,ID{k})],[],1);
ind = convhull(XX,YY);
% plot(XX,YY,'.','Color',CLR{k}),hold on
h1(k)=fill(XX(ind),YY(ind),':b','FaceColor',CLR{k},'EdgeColor',CLR{k},'FaceAlpha',0.1);
hold on,grid on
XX = reshape([Y(k).XU(:,ID{k});Y(k).XL(:,ID{k})],[],1);
YY = reshape([Y(k).YU(:,ID{k});Y(k).YL(:,ID{k})],[],1);
ind = convhull(XX,YY);
h2(k) = fill(XX(ind),YY(ind),'-b','FaceColor',CLR{k},'EdgeColor',CLR{k},...
          'FaceAlpha',0.15,'LineWidth',2,'MarkerSize',10);
end

ylim([-0.3 0.4]),xlim([-0.1 1.1])
set(gca,'PlotBoxAspectRatio',[1.2 0.7 1])

%%
CLRAF = {'k','r','g','y'};
for i=1:numel(Airfoils)
h3(i) = plot(AF{i}(:,1),AF{i}(:,2),'-','Color',CLRAF{i},'LineWidth',1.5);
plot(xn,AF_yu{i},'.k',xn,AF_yl{i},'.','Color',CLRAF{i},'MarkerSize',12);
end
hold off

legend([h2 h3],[{'$\;\mathcal{D}_{10}^4 \cup \mathcal{D}_{10}^5$: Design Space'},...
    strrep(Airfoils,'NACA','NACA ')],'Interpreter','latex','Location','NE','FontSize',12)

xlabel('Normalized x-axis') 
ylabel('Normalized y-axis') 
 shg