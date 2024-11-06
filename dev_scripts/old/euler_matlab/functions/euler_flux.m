function  f = euler_flux(Q,gamma)

% Flux function of the compressible Euler equations
rho = Q(1,:); rhou = Q(2,:); rhoE = Q(3,:);
u = rhou./rho;
p = (gamma-1)*(rhoE-1/2*rhou.^2./rho);
f = [rhou; rhou.*u+p; (rhoE+p).*u];

end