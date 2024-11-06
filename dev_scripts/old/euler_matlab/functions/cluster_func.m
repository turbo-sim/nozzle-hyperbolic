function f = cluster_func(x,a)
% Muller introduced this function in his notes (I should find a reference)
% 0<x<1 is an uniformly spaced vector
% a>1 is the clustering parameter

f = 1 + a*(1-((a+1)/(a-1)).^(1-x))./(1+((a+1)/(a-1)).^(1-x));

% The cluster function puts more points in the region where z is close to 0
% If you want more points in the region where z is close to 1 define the
% problem in the opposite manner and then reverse the resulting vector

end