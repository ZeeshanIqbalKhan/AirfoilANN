function [DATA,Headers] = ReadJavaFoilData(path,label)
f = dir([path label '_M*.txt']);
for n = 1:length(f)
    fname = f(n).name;
    TT = readtable([path fname]);
    II = find(cell2mat(cellfun(@(x)(strcmp(x,'Mach')),TT.Var1,'UniformOutput',false)) == 1);
    II = [II; height(TT)+2];
    for k=1:length(II)-1
        M(n,k).Name = cell2mat(table2cell(TT(II(2)-1,3:4)));
        temp = cell2mat(TT(II(k),3).Var3);
        MachNumber = str2num(temp(:,1:end-1));
        temp = cell2mat(TT(II(k),6).Var6);
        ReynoldsNumber = str2num(temp(:,1:end-1));
        
        Headers = ['ReynoldsNumber','MachNumber','alpha',TT{II(k)+2,2:4}];
        
        AA = table2cell(TT(II(k)+4:II(k+1)-2,1:4));
        AA(cellfun(@(x)strcmp(x,'?'),AA)) = {'nan'};
        Data = cellfun(@str2num,AA);
        M(n,k).DATA = [repmat([ReynoldsNumber MachNumber],size(Data,1),1) Data];
    end
end
DATA = cell2mat({M.DATA}');
end