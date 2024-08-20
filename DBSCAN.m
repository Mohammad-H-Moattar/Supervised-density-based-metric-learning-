%
% Copyright (c) 2015, Yarpiz (www.yarpiz.com)
% All rights reserved. Please read the "license.txt" for license terms.
%
% Project Code: YPML110
% Project Title: Implementation of DBSCAN Clustering in MATLAB
% Publisher: Yarpiz (www.yarpiz.com)
% 
% Developer: S. Mostapha Kalami Heris (Member of Yarpiz Team)
% 
% Contact Info: sm.kalami@gmail.com, info@yarpiz.com
%

function [IDX, isnoise]=DBSCAN(X,epsilon,MinPts)

    C=0;
    
    n=size(X,1);
    IDX=zeros(n,1);
    
    D=pdist2(X,X); %default euclidian
    
   % D=pdist2(X,X,'hamming');
    
    visited=false(n,1);
    isnoise=false(n,1);
    
    for i=1:n
        if ~visited(i)
            visited(i)=true;
            
            Neighbors=RegionQuery(i);
            if numel(Neighbors)<MinPts
                % X(i,:) is NOISE
                isnoise(i)=true;
            else
                C=C+1;
                ExpandCluster(i,Neighbors,C);
            end
            
        end
    
    end
    
    function ExpandCluster(i,Neighbors,C)
        IDX(i)=C;
        
        k = 1;
        while true
            j = Neighbors(k);
            
            if ~visited(j)
                visited(j)=true;
                Neighbors2=RegionQuery(j);
                if numel(Neighbors2)>=MinPts
                    Neighbors=[Neighbors Neighbors2];   %#ok
                end
            end
            if IDX(j)==0
                IDX(j)=C;
            end
            
            k = k + 1;
            if k > numel(Neighbors)
                break;
            end
        end
    end
    
    function Neighbors=RegionQuery(i)
        Neighbors=find(D(i,:)<=epsilon);
    end

end

% function [ clusters ] = DBSCAN( A, Eps, minPts)
% %DBSCAN Given a (usually sparse) similarity matrix, cluster the items
% %   Aim:
% %       Clustering the data with Density-Based Scan Algorithm with Noise (DBSCAN)
% %
% %   Input:
% %       A - similarity matrix (should be sparse, this is quite slow if it
% %       isn't sparse!)
% %       Eps - Neighbourhood radius, things are considered neighbours if
% %       their similarity is HIGHER than this value
% %       minPts - number of objects in a neighborhood of an object (minimal
% %       to consider the object to be in a cluster)
% % e.g.
% % x=[randn(30,2)*.4;randn(40,2)*.5+ones(40,1)*[4 4]];
% % dbscan(simdist(x), 0.5, 10)
% % == 2 cluster
% % also tested with quite challenging setups:
% % A = sprand(50000,50000,0.01);
% % A = triu(A) + triu(A,1)'; make it diagonal, more realistic
% % clusters = dbscan(A,0.1,5); takes about 10 seconds
% 
%     nrows = size(A,1);
%     
%     visited = zeros(nrows,1); % visited
%     noise = zeros(nrows,1); % noise
%     clusters = zeros(nrows,1); % the clusters
% 
%     clusterIndex = 1;
% 
%     for i = 1:nrows
%         if visited(i) == 1
%             continue
%         end
%         % get the unvisited items with a valid similarity
%         region = A(:,i) > Eps & ~visited;
%         % make sure we have enough neighbours
%         % we leave the item "unvisited", it could become part of another
%         % cluster
%         if nnz(region) < minPts
%             noise(i) = 1 ;% add to noise
%         else
%             % unassigned, assign to the current cluster index
%             clusters(i) = clusterIndex;
%             
%             % while we have non-zero elements
%             while nnz(region) > 0
%                 % get the index of the first non-zero element
%                 ip = find(region,1);
%                 % remove it from the region, declare assignment
%                 region(ip) = 0;
%                 visited(ip) = 1;
%                 % get unvisited neighbours with a valid similarity
%                 regprime = A(:,ip) > Eps & ~visited ;
%                 % add those items to the region if there are enough
%                 if nnz(regprime) >= minPts
%                     region = regprime | region;
%                 end
%                 % assign to the current cluster, remove from noise
%                 noise(ip) = 0;
%                 clusters(ip) = clusterIndex;
%             end
%             clusterIndex = clusterIndex + 1;
%         end
%     end
% end

