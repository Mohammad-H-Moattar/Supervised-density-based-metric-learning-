function [Accuracy,TPR,TNR,P,R,GM,F1,X,yTr] =  knnValidate(L, xTr, yTr, xTe, yTe, knn)

     X  = L * xTr';
     Xt = L * xTe';  

    preds = knnClassifier(X, yTr, knn, Xt);
    percent = 100 * sum (preds == yTe) / length(yTe);
    
    %%%%%%   compute confusion matrix and compute percentages  %kh
    
    C=confusionmat(yTe,preds);
%     TP=C(1,1);FN=C(1,2);FP=C(2,1);TN=C(2,2);
      TN=C(1,1);FP=C(1,2);FN=C(2,1);TP=C(2,2);
      
    Accuracy = (TP+TN)/(TP+TN+FP+FN);     % Aaccuracy
    TPR   = TP/(TP+FN);                   % ACC+
    TNR   = TN/(TN+FP);                   % Acc-
    P=TP/(TP+FP);                         % precision
    R=TP/(TP+FN);                         % recal
    GM    = sqrt(P*R);                    % Geometric Mean
    F1 = (2*P*R)/(P+R);                   % F1 score
   %F1 = TP/(TP+FN/2+FP/2)
    X=X';
end 
  function perf= KNNperfkh(ltrain,datatrain,datatest,ltest)

knnclasslabeltest=knnclassify(datatest,datatrain,ltrain,3);

perf=((length(find(knnclasslabeltest==ltest)))/length(ltest))*100;

% C=confusionmat(ltest,knnclasslabeltest);
% acc_of_knn=sum(diag(C))/sum(C(:))*100;
 end 