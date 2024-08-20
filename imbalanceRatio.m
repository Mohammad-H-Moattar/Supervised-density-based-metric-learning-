function IR=imbalanceRatio(label,n_data)
countpositive=0;
countnegative=0;
 for i=1:n_data
  if label(i,1)==1
     countnegative=countnegative+1;
 end
 if label(i,1)==2
     countpositive=countpositive+1;
 end
 end
 if countnegative>countpositive
 IR=countnegative/countpositive;
 else
 IR=countpositive/countnegative;
 end
 

end