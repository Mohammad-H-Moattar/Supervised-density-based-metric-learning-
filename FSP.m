%%%%%%% by atena jalali
function FSP(Data,Label)
% feature selection and plot 2 dim

%  load fisheriris
% %[ V, eigvalueSum ] = fld( X, L, n, crit, qrf, r, e, M )
% %[ V, eigvalueSum ] = fld( meas, species,2);
% DT=fitctree(meas,species);
% costs=predictorImportance(DT)
% % f = figure;
% % gscatter(meas(:,1), meas(:,2), species,'rgb','osd');
% gscatter(meas(:,3), meas(:,4), species,'rgb','osd');
% xlabel('Sepal length');
% ylabel('Sepal width');

% %%%%% feature selection with decision tree (gini)
DT=fitctree(Data,Label);
costs=predictorImportance(DT)
% bar(1:numel(costs),costs)
[~,sortorder]=sort(costs,'descend')
firstbestfeature=sortorder(1);
secondbestfeature=sortorder(2);
%%%%%%%%%%%%%% plot dataset by two best feature 
 gscatter(Data(:,firstbestfeature), Data(:,secondbestfeature), Label,'rgb','osd');
 hold on

end
