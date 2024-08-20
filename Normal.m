function [normalData] = Normal(data)

[s f]=size(data);
xMin = min(data);
xMax=max(data);
% for i = 1: f-1
 for i = 1: f
    if xMin(i) == 0 & xMax(i) ==0
         normalData(1:s,i)= 0;
    else
        normalData(1:s,i)= (data(:,i)- xMin(i)) ./ (xMax(i) - xMin(i)) ;
% kh        y = (ymax-ymin)*(x-xmin)/(xmax-xmin) + ymin;
    end
end
%     normalData=[normalData , data(:,f)];

%%% or 3 line zer
% mindata = min(data);
% maxdata = max(data);
% minmaxdata = bsxfun(@rdivide, bsxfun(@minus, data, mindata), maxdata - mindata);
end

