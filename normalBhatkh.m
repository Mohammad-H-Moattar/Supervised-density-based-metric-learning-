function  E=normalBhatkh(disBhattacharyyaProjected,n_class ,n_components,Xprojectpercomponent)

for i=1:n_class
  for j=1:n_components{i}
  numberpercomponent{i,j}=size(Xprojectpercomponent{i,j},1);
  end
end

makhrag=0;
 for x=1:n_class-1
    for y=x+1:n_class
      for i=1:n_components{x}
        for j=1:n_components{y} 
        makhrag=makhrag+cell2mat(numberpercomponent(x,i))*cell2mat(numberpercomponent(y,j))*disBhattacharyyaProjected(i,j,x);
    end
  end 
  end
  end

 for x=1:n_class-1
    for y=x+1:n_class
      for i=1:n_components{x}
        for j=1:n_components {y}     
     E(i,j)=(cell2mat(numberpercomponent(x,i))*cell2mat(numberpercomponent(y,j))*disBhattacharyyaProjected(i,j,x))/makhrag;
    end
  end
    end
 end

 end
