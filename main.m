
%%%%%%%%% proposed methode DMLdbIm jalali
clc 
clear all
close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  uci
% 
load('dataset-uci/heart');
Data=heart(:,1:13);
Label=heart(:,14);
% % 
% load('dataset-uci/wdbcdata');
% load('dataset-uci/wdbclabel');
% Data=data;
% Label=label;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%      keel 

%  load dataset-keel/pima/pimadata.mat
%  load dataset-keel/pima/pimalabel.mat
%  Data=pimadata;
%  Label=pimalabel;


% load dataset-keel/glass/glass0data.mat
% load dataset-keel/glass/glass0label.mat
% Data=glass0data;
% Label=glass0label;

% load dataset-keel/ecoli/ecoli1data.mat
% load dataset-keel/ecoli/ecoli1label.mat
% Data=ecoli1data;
% Label=ecoli1label;
% 
% load dataset-keel/ecoli/ecoli2data.mat
% load dataset-keel/ecoli/ecoli2label.mat
% Data=ecoli2data;
% Label=ecoli2label;

% load('dataset-keel/newthyroid/newthyroid1');
% Data=X;
% Label=y;
% for i=1:215
%    if Label(i,1)==-1
%        Label(i,1)=1;  %hight negative
%    else
%         Label(i,1)=2; %less positive
%    end
% end

% load dataset-keel/glass/glass6data.mat
% load dataset-keel/glass/glass6label.mat
% 
% load dataset-keel/ecoli/ecoli3data.mat
% load dataset-keel/ecoli/ecoli3label.mat
% Data=ecoli3data;
% Label=ecoli3label;

% load dataset-keel/yeast/yeast2vs4data.mat
% load dataset-keel/yeast/yeast2vs4label.mat
% Data=yeast2vs4data;
% Label=yeast2vs4label;
% 
% load dataset-keel/yeast/yeast1vs7data.mat
% load dataset-keel/yeast/yeast1vs7label.mat
% Data=yeast1vs7data;
% Label=yeast1vs7label;
% 
% load dataset-keel/winequality/winequalityred8vs6data.mat
% load dataset-keel/winequality/winequalityred8vs6label.mat
% Data=winequalityred8vs6data;
% Label=winequalityred8vs6label;

% load dataset-keel/winequality/winequalityred8vs67data.mat
% load dataset-keel/winequality/winequalityred8vs67label.mat
% Data=winequalityred8vs67data;
% Label=winequalityred8vs67label;

% load dataset-keel/winequality/winequalitywhite39vs5data.mat
% load dataset-keel/winequality/winequalitywhite39vs5label.mat
% Data=winequalitywhite39vs5data;
% Label=winequalitywhite39vs5label;
% 
% load dataset-keel/winequality/winequalityred3vs5data.mat
% load dataset-keel/winequality/winequalityred3vs5label.mat
% Data=winequalityred3vs5data;
% Label=winequalityred3vs5label;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%      keel for deep learning

% load dataset-keel/glass/glass1data.mat
% load dataset-keel/glass/glass1label.mat

% load dataset-keel/yeast/yeast6data.mat
% load dataset-keel/yeast/yeast6label.mat

% load dataset-keel/vehicle/vehicle3data.mat
% load dataset-keel/vehicle/vehicle3label.mat

% load dataset-keel/ecoli/ecoli0147vs2356data.mat
% load dataset-keel/ecoli/ecoli0147vs2356label.mat

% load dataset-keel/page/pageblocks0data.mat
% load dataset-keel/page/pageblocks0label.mat
% Data=pageblocks0;
% Label=pageblocks0label;

% load dataset-keel/ecoli/ecoli0147vs2356data.mat
% load dataset-keel/ecoli/ecoli0147vs2356label.mat
% Data=ecoli0147vs2356;
% Label=ecoli0147vs2356label;

% load dataset-keel/ecoli/ecoli01vs235data.mat
% load dataset-keel/ecoli/ecoli01vs235label.mat
% Data=ecoli01vs235data;
% Label=ecoli01vs235label;

% load dataset-keel/ecoli/ecoli0267vs35data.mat
% load dataset-keel/ecoli/ecoli0267vs35label.mat
% Data=ecoli0267vs35data;
% Label=ecoli0267vs35label;

%%%%%%%%%%%%%%%%%%%%%%% START main

 data=Normal(Data);
 n_class=size(unique(Label),1);
 n_dim = size(data, 2);
 n_data = size(data, 1);
 
 L1=0.00001;
 L2=0.00001;
 alpha=0.1;
 tol=1e-5;
 knn=3;
 kf=5;
%  
 epsilonDBscan=0.5;  % heart = wdbc = glass6 = 0.5,5        glass0 = 0.2,5        ecoli1 = ecoli2 = 0.3,3   
 MinPts=5;           % newthroid1 = ecoli3 = yeast2vs4 = yeast1vs7 = 0.3,5  
                     % winequalityred8vs6=winequalityred8vs67=0.5,5
                     % winequalitywhite39vs5 = winequalityred3vs5 = 0.5,3
                     % pima=0.4,5
 disp('%%%%%%%%% imbalance Ratio  %%%%%%%%%%');
 IR=imbalanceRatio(Label,n_data) 
 w = eye(n_dim);

 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for t=1:10
indexmain= crossvalind('kfold',Label,kf);
for jj=1:kf
    
test=(indexmain==jj);
train=~test;
datatrain=data(train,:);
ltrain=Label(train,:);
datatest=data(test,:);
ltest=Label(test,:);
n_datatrain=size(datatrain, 1);

%%%%%%%%%%%%%%%%%%%%%% show training set before projection
close all;
FSP(datatrain,ltrain);
title('before projection');

%%%%%Use only training set and Divide the training set based on the class labels 
tic
for z=1:n_class
     ind=find(ltrain==z);
     datatrainclass{z}=datatrain(ind,1:n_dim);
end
%%%%%Perform DBSCAN on each class

for z=1:n_class
    [labelsDBscan{z}, isnoise]=DBSCAN(datatrainclass{z},epsilonDBscan,MinPts);
end

%-- find cllusters DBscan
for z=1:n_class
numcluster{z}=numel(unique(labelsDBscan{z}));
end

%-- find samples of components per class in DBscan
numcomponent=0;
for i=1:n_class
  for j=1:numcluster{i}-1
     ind=find(labelsDBscan{i}==j);
     components{i,j}=datatrainclass{i}(ind,1:n_dim);
  end
    n_components{i}=j;
    numcomponent=numcomponent+n_components{i};
end

%%%%%%%%%%%%%%%%%%%%%%%% GMM Plot befor linear transformation

for i=1:n_class
[coeff, score] = pca(datatrainclass{i});
data=[score(:,1) score(:,2)]
num_components = n_components{i};  
gmm_model = fitgmdist(data, num_components);
figure;
scatter(data(:,1), data(:,2), 10, 'k');          
x1 = linspace(min(data(:,1)), max(data(:,1)), 100);
x2 = linspace(min(data(:,2)), max(data(:,2)), 100);
[x1Grid, x2Grid] = meshgrid(x1, x2);
for k = 1:num_components
    mu = gmm_model.mu(k, :);                 
    sigma = gmm_model.Sigma(:, :, k);        
    pdfValues = mvnpdf([x1Grid(:), x2Grid(:)], mu, sigma);
    pdfValues = reshape(pdfValues, size(x1Grid));
    contour(x1Grid, x2Grid, pdfValues, 'LineWidth', 2); 
    hold on
end
title('Gaussian Mixture Model Components before projection');
xlabel('Feature 1');
ylabel('Feature 2');
hold off;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% End GMM 


% initial covariance , means ,prior probabilities for per class
sizedataset=0;
for z=1:n_class
  covperclass{z} = cov(datatrainclass{z});  
  meanperclass{z}=mean(datatrainclass{z});
  sizeperclass{z}=size(datatrainclass{z},1); 
  sizedataset=sizedataset+sizeperclass{z};
end

%----------initial covariance , means ,prior probabilities(weights) for each component
K=numcomponent;
weights = ones(1, K)./K;
for i=1:n_class
  for j=1:n_components{i}
     covpercomp{i,j}=cov(components{i,j});
     meanpercomp{i,j}=mean(components{i,j});
     sizepercomp{i,j}=size(components{i,j},1);
     weightspercomp{i,j}=weights(:,j);    
  end
end

%%%%%%%%%%%%%%%%% update cov ¡ update mean and weight only one time

for i=1:n_class
for j=1:n_components{i}
    gh1{i,j} = (covperclass{i}*(sizeperclass{i}-1))+(covpercomp{i,j}*(sizepercomp{i,j}-1));
    gh2{i,j} = (prod(meanperclass{i})*sizeperclass{i})+(prod(meanpercomp{i,j})*sizepercomp{i,j});  
    gh3{i,j} =(gh1{i,j}+gh2{i,j})/(sizeperclass{i}+sizepercomp{i,j});
    ad{i,j}= (sizeperclass{i}+sizepercomp{i,j}) /((sizeperclass{i}+sizepercomp{i,j})-1);   
    ExEy{i,j}=prod((meanperclass{i}*sizeperclass{i}+meanpercomp{i,j}*sizepercomp{i,j})/(sizeperclass{i}+sizepercomp{i,j}));
    covpercomp{i,j}=(gh3{i,j}.*ad{i,j})-(ExEy{i,j}*ad{i,j});
    meanpercomp{i,j}=(meanperclass{i}*sizeperclass{i})+(meanpercomp{i,j}*sizepercomp{i,j});
end
end

%--------calculate  bhattacharyya distance -----------
distbhatend=0;
for x=1:n_class-1
   for y=x+1:n_class
     for i=1:n_components{x}
       for j=1:n_components{y}
           C=(covpercomp{x,i}+covpercomp{y,j})/2;
           [R,P]=chol(C);
           if P==0
           disbhat(i,j)=bhattacharyyakh(meanpercomp{x,i},covpercomp{x,i},meanpercomp{y,j},covpercomp{y,j});
           distbhatend=disbhat;
           else
           disbhat=distbhatend;    
           end
    end
end
   end
end
dis2class=mean(mean(disbhat));

%%  w = comput eigenvector = linear transformation 

covalldatatrain = cov(datatrain);
[eigvector,eigvalue]  = eig( covalldatatrain);
 W= eigvector;
 
%%--------- update W  ------------------------------
   
   max_iter=50;
   iter=1;
   Wold=W;
   Wnew=W;
   Wend=W;
   LAold=0;
   dis2classold=dis2class;
   while iter<= max_iter
       
       [dis2classproject,disBhatproject,Xprojectpercomponent]=objectfunckh(components , Wnew ,n_class ,n_components ,n_dim ,  epsilonDBscan, MinPts);        
       E=normalBhatkh(disBhatproject,n_class ,n_components,Xprojectpercomponent);
       logarithmEA=GeometricMeankh(E,n_class,n_components);
       
 %%-- Compute value of objective function-----
       
       for r=1:n_datatrain-1
           for c=r+1:n_datatrain
               if ltrain(r,1)~= ltrain(c,1)
                   drc=datatrain(r,:)-datatrain(c,:);
                   dA=sqrt(drc*Wnew'*Wnew*drc');
                   dAadd(r,c)=dA;
               end
           end
       end
       
       for r=1:n_datatrain-1
           for c=r+1:n_datatrain
               if ltrain(r,1)== ltrain(c,1)
                   drc=datatrain(r,:)-datatrain(c,:);
                   dA=sqrt(drc*Wnew'*Wnew*drc');
                   dAsub(r,c)=dA;
               end
           end
       end
       
       LAnew = logarithmEA + L1*sum(sum(dAadd))-L2*sum(sum(dAsub));
       grad=LAnew-LAold;
       LAold=LAnew;
       gradianLA=repmat(grad,n_dim,n_dim);
       
       Wnew=Wold + alpha * gradianLA;
       www=Wnew-Wold;
       if Wnew-Wold<tol
           break;
       end
       
       Wold=Wnew;
       dis2classold=dis2classproject;
       iter=iter+1;
   end
% if ismissing(Wnew)
     if isnan(Wnew(1,1))
     Wnew= eigvector;
 end

%% KNN classifier   ----------------------------
% 
[accWnew{jj},precisionWnew{jj},recalWnew{jj},F1Wnew{jj},X,y]=knnValidate(Wnew,datatrain,ltrain,datatest,ltest,knn);

%%%%%%%%%%%%%%%%%%%%%% show training set after projection
figure
 X  = (Wnew * datatrain')';
Xt = (Wnew * datatest')';
 y = ltrain;
FSP(X,y);
title('after projection');

%%%%%%%%%%%%%%%%%%%%%%%% GMM Plot after linear transformation
% 
for z=1:n_class
     ind=find(y==z);
     Datatravsfer{z}=X(ind,1:n_dim);
end
for z=1:n_class
    [labelDBscan{z}, isnoise]=DBSCAN(Datatravsfer{z},epsilonDBscan,MinPts);
end
for z=1:n_class
ncluster{z}=numel(unique(labelDBscan{z}));
end
ncomponent=0;
for i=1:n_class
  for j=1:ncluster{i}-1
     ind=find(labelDBscan{i}==j);
     component{i,j}=Datatravsfer{i}(ind,1:n_dim);
  end
    n_component{i}=j;
    ncomponent=ncomponent+n_component{i};
end
for i=1:n_class
[coeff, score] = pca(Datatravsfer{i});
Data=[score(:,1) score(:,2)];
num_component = n_component{i};  
gmm_model = fitgmdist(Data, num_component);
figure;
scatter(Data(:,1), Data(:,2), 10, 'k');       
x1 = linspace(min(Data(:,1)), max(Data(:,1)), 100);
x2 = linspace(min(Data(:,2)), max(Data(:,2)), 100);
[x1Grid, x2Grid] = meshgrid(x1, x2);
for k = 1:num_component
    mu = gmm_model.mu(k, :);                  
    sigma = gmm_model.Sigma(:, :, k);        
    pdfValue = mvnpdf([x1Grid(:), x2Grid(:)], mu, sigma);
    pdfValue = reshape(pdfValue, size(x1Grid));
    contour(x1Grid, x2Grid, pdfValue, 'LineWidth', 2); 
    hold on
end
title('Gaussian Mixture Model Components after projection');
xlabel('Feature 1');
ylabel('Feature 2');
hold off;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% End GMM 
toc
end
 perfaccWnewknn{t}=mean(cell2mat(accWnew))
 precisionWnew=cell2mat(precisionWnew)
 perfrecalWnewknn{t}=mean(cell2mat(recalWnew))
 F1Wnew=cell2mat(F1Wnew)
end


 



