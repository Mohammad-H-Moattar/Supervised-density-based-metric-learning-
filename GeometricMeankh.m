function  logarithmEA=GeometricMeankh(E,n_class,n_components)
Eproduct=1;
for x=1:n_class-1
   for y=x+1:n_class
     for i=1:n_components{x}
       for j=1:n_components{y}
        Eproduct=E(i,j)*Eproduct;    
    end
  end
 end
end

EA=nthroot(Eproduct,i*j);

logarithmEA=log10(EA);
end

