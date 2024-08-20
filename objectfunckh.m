function [dis2classproject,disbhatproject,Xprojectpercomponent]=objectfunckh(compnents ,W ,n_class ,n_components,n_dim , epsilonDBscan, MinPts )

data=[];
for i=1:n_class
for j=1:n_components{i}
   Xprojectpercomponent{i,j}  = compnents{i,j} * W;
%  Xprojectpercomponent{i}  = W * compnents{i}' ;
    D=Xprojectpercomponent{i,j}(:,:);
    data=[data;D];
end
   datatrainclassproject{i}=data(:,:);
   data=[];
end

%Perform DBSCAN on each class

for z=1:n_class
[labelsDBscanproject{z}, isnoiseproject]=DBSCAN(datatrainclassproject{z},epsilonDBscan,MinPts);
end

%---------- find cllusters DBscan
for z=1:n_class
numclusterproject{z}=numel(unique(labelsDBscanproject{z}));
end

%---------- find samples of components per class in DBscan
numcomponentproject=0;
for i=1:n_class
  for j=1:numclusterproject{i}-1
     ind=find(labelsDBscanproject{i}==j);
     compnentsproject{i,j}=datatrainclassproject{i}(ind,1:n_dim);
  end
    n_componentsproject{i}=j;
    numcomponentproject=numcomponentproject+n_componentsproject{i};
end
% initial covariance , means ,prior probabilities for per class
sizedatasetproject=0;
for z=1:n_class
  covperclass{z} = cov(datatrainclassproject{z});  
  meanperclass{z}=mean(datatrainclassproject{z});
  sizeperclass{z}=size(datatrainclassproject{z},1); 
  sizedataset=sizedatasetproject+sizeperclass{z};
end

%----------initial covariance , means ,prior probabilities(weights) for each component
K=numcomponentproject;
for i=1:n_class
  for j=1:n_components{i}
     covpercomp{i,j}=cov(compnents{i,j});
     meanpercomp{i,j}=mean(compnents{i,j});
     sizepercomp{i,j}=size(compnents{i,j},1);
  end
end

%%%%%%%%%%%%%%%%% update cov ¡ update mean and weight one time

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

%------------------------------calculate bhattacharyya projected data -------------------

for x=1:n_class-1
   for y=x+1:n_class
       distbhatprojectend=zeros(n_components{x},n_components{y});
     for i=1:n_components{x}
       for j=1:n_components{y}
            C=(covpercomp{x,i}+covpercomp{y,j})/2;
           [R,P]=chol(C);
           if P==0
              disbhatproject(i,j)=bhattacharyyakh(meanpercomp{x,i},covpercomp{x,i},meanpercomp{y,j},covpercomp{y,j});
              distbhatprojectend=disbhatproject;
           else
              disbhatproject= distbhatprojectend; 
           end
    end
end
   end
end
 dis2classproject=mean(mean(disbhatproject));