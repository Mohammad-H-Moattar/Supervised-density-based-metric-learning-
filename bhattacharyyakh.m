  function d=bhattacharyyakh(mu1,C1,mu2,C2)
%%%  Bhattacharyya distance between two Gaussian classes

% clc
% clear all
% mu1=[-0.6702 -1.1827]
% C1=10^-5*[0.1966 0.0179;0.0179 0.0024]
% mu2=[-0.6641 -1.1803]
% C2=10^-6*[0.6853 0.3309;0.3309 0.1618]

 C=(C1+C2)/2;
% [R,p]=chol(C);
% if p==0
% ans=mu1-mu2
dmu=(mu1-mu2)/chol(C);

%try
%     d1=0.125*dmu*dmu'+0.5*log(det(C/chol(C1*C2)));
    %%%% kh
     d=0.125*dmu*dmu'+0.5*log(abs(det(C)/(sqrtm(det(C1)*det(C2)))));
%    d=9.0679;
%catch
  % d3=0.125*dmu*dmu'+0.5*log(abs(det(C/sqrtm(C1*C2))))
  % warning('MATLAB:divideByZero','Data are almost linear dependent. The results may not be accurate.');
% end
% d4=0.125*dmu*dmu'+0.25*log(det((C1+C2)/2)^2/(det(C1)*det(C2)))
  end
