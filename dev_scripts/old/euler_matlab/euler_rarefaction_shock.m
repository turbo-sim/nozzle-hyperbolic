%% Exact solution of the Euler equations for the shock tube problem
% This case study displays a 1-rarefaction wave and a 3-shock wave
% Author: Roberto Agromayor
% Date: 29/11/2017
clear all
close all
clc
addpath(fullfile(pwd,'/functions'))
plot_settings


%% Define the computational problem
% Define the grid
Nx = 500;                                                                   % Number of cells
x1 = -1.00;                                                                % Left boundary
x2 = +1.00;                                                                % Right boundary
a = 100.0;                                                                  % Clustering parameter
x = linspace(0,1,Nx);                                                      % x grid vector
x = cluster_func(x,a);
x = x1 + (x2-x1)*x;

% Define the simulation time
t1 = 0.00;                                                                 % Initial time
t2 = 0.25;                                                                 % Final time
t = [t1,t2];                                                               % Time vector

% Define the CFL condition
CFL = 0.50;                                                                % Maximum allowed local Courant number

% Define the properties of the gas
R = 287.00;                                                                % Gas constant of air
gamma = 1.40;                                                              % Heat capacity ratio of air



%% Define the initial conditions
% Left side of the shock tube
d_L = 8.00;
u_L = 0.00;
p_L = 10.00;
V_L = [d_L; u_L; p_L];
q_L = [d_L; d_L*u_L; p_L/(gamma-1)+d_L*u_L^2/2];                           % Left state in conservative variables

% Right side of the shock tube
d_R = 1.00;
u_R = 0.50;
p_R = 1.00;
V_R = [d_R; u_R; p_R];
q_R = [d_R; d_R*u_R; p_R/(gamma-1)+d_R*u_R^2/2];                           % Right state in conservative variables

% Defining the initial conditions using the properties of matrix product
Q_o = @(x) q_L*(x <= 0)+q_R*(x > 0);


% Group up the problem parameters
parameters.N_eq = 3;                                                       % Number of equations of the hyperbolic system
parameters.CFL = CFL;                                                      % Maximum allowed Courant number
parameters.Q_o = Q_o;                                                      % Initial condition function
parameters.time_method = 'euler';                                            % Choose: 'euler', 'RK3', or 'RK4'
parameters.space_method = 'rusanov';                                       % Choose: 'godunov', 'roe', 'rusanov', or 'lax-friedrichs'
parameters.reconstruction = 'constant';                                    % Choose: 'constant' or 'linear'
parameters.flux_function = @(q) euler_flux(q,gamma);                       % Flux function of the Euler equations
parameters.eigenvalues_function = @(q) euler_eigenvalues(q,gamma);         % Eigenvalues of the Euler equations
parameters.roe_function = @(q_L,q_R) euler_roe_matrix(q_L,q_R,gamma);      % Eigenvalues of the Euler equations
parameters.exact_solution = @(q_L,q_R) euler_exact(0,1,q_L,q_R,gamma);     % Exact solution of the of the Euler equations for the Riemann problem


%% Compute the exact solution of the problem
% Define the grid
N_exact = 500; x_exact = linspace(x1,x2,N_exact);
q_exact = zeros(3,N_exact);
for i = 1:N_exact
    q_exact(:,i) = euler_exact(x_exact(i),t2,q_L,q_R,gamma);
end
d_exact = q_exact(1,:);
u_exact = q_exact(2,:)./q_exact(1,:);
p_exact = (gamma-1)*(q_exact(3,:)-1/2*q_exact(2,:).^2./q_exact(1,:));
sonic_exact = u_exact-sqrt(gamma*p_exact./d_exact);

%% Compute the numerical solution of the problem
parameters.space_method = 'roe';
Q = fvm_euler_1d(x,t,parameters);
d_godunov(:,:) = Q(1,:,:);
u_godunov(:,:) = Q(2,:,:)./Q(1,:,:);
p_godunov(:,:) = (gamma-1)*(Q(3,:,:)-1/2*Q(2,:,:).^2./Q(1,:,:));
% 
% parameters.space_method = 'rusanov';
% Q = fvm_euler_1d(x,t,parameters);
% d_rusanov(:,:) = Q(1,:,:);
% u_rusanov(:,:) = Q(2,:,:)./Q(1,:,:);
% p_rusanov(:,:) = (gamma-1)*(Q(3,:,:)-1/2*Q(2,:,:).^2./Q(1,:,:));
% 
% parameters.space_method = 'roe';
% Q = fvm_euler_1d(x,t,parameters);
% d_roe(:,:) = Q(1,:,:);
% u_roe(:,:) = Q(2,:,:)./Q(1,:,:);
% p_roe(:,:) = (gamma-1)*(Q(3,:,:)-1/2*Q(2,:,:).^2./Q(1,:,:));
% 
% 
%% Plot the exact and numerical solutions
fig1 = figure(1);
set(fig1,'Position',get(0,'ScreenSize'));

% Subplot for density
subplot(1,3,1); hold on; box on; axis square
xlabel('Position (x)'); ylabel('Density');
plot(x,d_godunov(:,end),'ko','linewidth',0.75,'MarkerSize',2,'MarkerFaceColor','w');
plot(x_exact,d_exact,'k')
legend('Numerical solution Godunov','Exact solution','Location','SouthOutside')

% Subplot for velocity
subplot(1,3,2); hold on; box on; axis square
xlabel('Position (x)'); ylabel('Velocity');
plot(x,u_godunov(:,end),'bo','linewidth',0.75,'MarkerSize',2,'MarkerFaceColor','w');
plot(x_exact,u_exact,'b')
legend('Numerical solution Godunov','Exact solution','Location','SouthOutside')

% Subplot for pressure
subplot(1,3,3); hold on; box on; axis square
xlabel('Position (x)'); ylabel('Pressure');
plot(x,p_godunov(:,end),'ro','linewidth',0.75,'MarkerSize',2,'MarkerFaceColor','w');
plot(x_exact,p_exact,'r')
legend('Numerical solution Godunov','Exact solution','Location','SouthOutside')


% 
% fig2 = figure(2);
% set(fig2,'Position',get(0,'ScreenSize'));
% 
% % Subplot for density
% subplot(1,4,1); hold on; box on; axis square
% xlabel('Position (x)'); ylabel('Density');
% plot(x,d_rusanov(:,end),'bo','linewidth',0.75,'MarkerSize',2,'MarkerFaceColor','w');
% plot(x,d_roe(:,end),'ro','linewidth',0.75,'MarkerSize',2,'MarkerFaceColor','w');
% plot(x_exact,d_exact,'k')
% legend('Rusanov','Roe','Exact solution','Location','SouthOutside')
% 
% % Subplot for velocity
% subplot(1,4,2); hold on; box on; axis square
% xlabel('Position (x)'); ylabel('Velocity');
% plot(x,u_rusanov(:,end),'bo','linewidth',0.75,'MarkerSize',2,'MarkerFaceColor','w');
% plot(x,u_roe(:,end),'ro','linewidth',0.75,'MarkerSize',2,'MarkerFaceColor','w');
% plot(x_exact,u_exact,'k')
% legend('Rusanov','Roe','Exact solution','Location','SouthOutside')
% 
% % Subplot for pressure
% subplot(1,4,3); hold on; box on; axis square
% xlabel('Position (x)'); ylabel('Pressure');
% plot(x,p_rusanov(:,end),'bo','linewidth',0.75,'MarkerSize',2,'MarkerFaceColor','w');
% plot(x,p_roe(:,end),'ro','linewidth',0.75,'MarkerSize',2,'MarkerFaceColor','w');
% plot(x_exact,p_exact,'k')
% legend('Rusanov','Roe','Exact solution','Location','SouthOutside')
% 
% 

