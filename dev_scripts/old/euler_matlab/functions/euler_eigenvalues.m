function  a = euler_eigenvalues(Q,gamma)

% Eigenvalues of the compressible Euler equations
rho = Q(1,:); rhou = Q(2,:); rhoE = Q(3,:);
u = rhou./rho;
p = (gamma-1)*(rhoE-1/2*rhou.^2./rho);
c = sqrt(gamma*p/rho);
a = [u-c; u; u+c];

end
