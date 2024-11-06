function A = euler_roe_matrix(Q_L,Q_R,gamma)

% Compute the left state
rho_L  = Q_L(1);
rhou_L = Q_L(2);
rhoE_L = Q_L(3);
u_L    = rhou_L/rho_L;
p_L    = (gamma-1)*(rhoE_L-1/2*rhou_L^2/rho_L);
H_L    = (rhoE_L+p_L)/rho_L;
c_L    = sqrt(gamma*p_L/rho_L);

% Compute the right state
rho_R  = Q_R(1);
rhou_R = Q_R(2);
rhoE_R = Q_R(3);
u_R    = rhou_R/rho_R;
p_R    = (gamma-1)*(rhoE_R-1/2*rhou_R^2/rho_R);
H_R    = (rhoE_R+p_R)/rho_R;
c_R    = sqrt(gamma*p_R/rho_R);

% Compute the Roe's averages
u = (sqrt(rho_L)*u_L+sqrt(rho_R)*u_R)/(sqrt(rho_L)+sqrt(rho_R));
H = (sqrt(rho_L)*H_L+sqrt(rho_R)*H_R)/(sqrt(rho_L)+sqrt(rho_R));
c = sqrt((gamma-1)*(H-u^2/2));

% Compute the eigenvector matrix
R = [    1,     1,     1;
       u-c,     u,   u+c;
     H-c*u, u^2/2, H+u*c];
 
% % Compute the eigenvalue matrix without entropy fix
% L = diag([abs(u-c),abs(u),abs(u+c)]);

% % Compute the eigenvalue matrix with entropy fix
eig1_L = u_L-c_L;
eig1_R = u_R-c_R;
eig3_L = u_L+c_L;
eig3_R = u_R+c_R;
L = diag([H_fix(eig1_L,eig1_R,(u-c)), u, H_fix(eig3_L,eig3_R,(u+c))]);

% Compute Roe's matrix
A = R*L*inv(R);

end

function H = H_fix(a_L,a_R,a)
a = abs(a);
delta = max(a_R-a_L,2*eps);
H = a.*(a >= delta) + (a^2+delta^2)/(2*delta)*(a < delta);
end



