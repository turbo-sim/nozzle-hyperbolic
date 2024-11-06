function [Q,t] = fvm_euler_1d(x,t,parameters)

% Extend the grid to ghost cells
Nj = length(x);
a = x(1)-2*(x(2)-x(1));
b = x(1)-(x(2)-x(1));           
c = x(Nj)+(x(Nj)-x(Nj-1));
d = x(Nj)+2*(x(Nj)-x(Nj-1));
x = [a b x c d];
Nj = length(x);

% Initialize the time vector
t_end = t(end);
t = zeros(1000,1);
dt = sqrt(eps);
n = 1;

% Initialize the solution
N = parameters.N_eq;            % Number of equations of the system
Q = zeros(N,Nj,1000);           % Preallocate space for the solution
Q_o = parameters.Q_o;           % Load the initial condition
Q(:,:,1) = Q_o(x);              % Set the initial condition


while t(n) <= t_end

    % Left outflow boundary condition
    Q(:,3,n) = Q(:,1,n);
    Q(:,2,n) = Q(:,1,n);
    
    % Right outflow boundary condition
    Q(:,Nj,n)   = Q(:,Nj-2,n);
    Q(:,Nj-1,n) = Q(:,Nj-2,n);
    
    
    % Use the desired time integration method
    if strcmp(parameters.time_method,'euler') == 1
        % Update the solution using the explicit Euler method
        Q(:,:,n+1) = Q(:,:,n)+dt*R(Q(:,:,n),t(n),x,parameters);
        
    elseif strcmp(parameters.time_method,'RK3') == 1
        % Update the solution using Gottlieb's TVD RK3 method
        Q1 = Q(:,:,n)+dt*R(Q(:,:,n),t(n),x,parameters);
        Q2 = 3/4*Q(:,:,n)+1/4*Q1+dt/4*R(Q1,t(n)+dt/4,x,parameters);
        Q(:,:,n+1) = 1/3*Q(:,:,n)+2/3*Q2+2*dt/3*R(Q2,t(n)+dt/3,x,parameters);
        
    elseif strcmp(parameters.time_method,'RK4') == 1
        % Update the solution using the classical RK4 method
        k1 = R(Q(:,:,n),t(n),x,parameters);
        k2 = R(Q(:,:,n)+k1*dt/2,t(n)+dt/2,x,parameters);
        k3 = R(Q(:,:,n)+k2*dt/2,t(n)+dt/2,x,parameters);
        k4 = R(Q(:,:,n)+k3*dt,t(n)+dt,x,parameters);
        Q(:,:,n+1) = Q(:,:,n)+1/6*(k1+2*k2+2*k3+k4)*dt;
        
    else
        error('Choose a valid time method: ''euler'', ''RK3'', or ''RK4''')
        
    end
    
    % Adjust the time step based on the CFL condition for interior cells
    lambda = max(abs(parameters.eigenvalues_function(Q(:,3:Nj-2,n))),[],1);
    x_face(1:Nj-3) = x(2:Nj-2)+(x(3:Nj-1)-x(2:Nj-2))/2;
    dx(1:Nj-4) = x_face(:,2:Nj-3)-x_face(:,1:Nj-4);
    dt = (parameters.CFL)./max(lambda./dx);
    
    % Update the time
    t(n+1) = t(n)+dt;
    n = n+1;
    

end

% Discard the solution in the ghost cells
Q = Q(:,3:end-2,1:n);

end



function RHS = R(Q,~,x,parameters)

% Load relevant variables
[N,Nj] = size(Q);

% Compute the right hand side using a constant cell reconstruction
if strcmp(parameters.reconstruction,'constant') == 1
    
    % Define the vector of conservative variables at the faces  using a 
    % constant reconstruction
    Q_L(:,1:Nj-1) = Q(:,1:Nj-1);          % Left face
    Q_R(:,1:Nj-1) = Q(:,2:Nj);            % Right face

    % Compute the numerical fluxes
    F(:,1:Nj-1) = F_numeric(Q_L,Q_R,parameters);
    
    % Compute the cell sizes
    x_face(1:Nj-1) = x(1:Nj-1)+(x(2:Nj)-x(1:Nj-1))/2;
    dx(1:N,1:Nj-2) = ones(N,1)*(x_face(:,2:Nj-1)-x_face(:,1:Nj-2));
    
    % Compute the right hand side for the current time step of the ODE
    % RHS is computed for the interior cells
    % RHS is set to zero for ghost cells
    RHS = zeros(N,Nj);
    RHS(:,2:Nj-1) = -(F(:,2:Nj-1)-F(:,1:Nj-2))./dx;
    
end


% Compute the right hand side using a linear cell reconstruction
if strcmp(parameters.reconstruction,'linear') == 1
    
    % Define the vector of conservative variables at the faces using a 
    % linear reconstruction
    Q_L(:,1:Nj-3) = Q(:,2:Nj-2)+1/2*minmod(Q(:,3:Nj-1)-Q(:,2:Nj-2),Q(:,2:Nj-2)-Q(:,1:Nj-3));
    Q_R(:,1:Nj-3) = Q(:,3:Nj-1)-1/2*minmod(Q(:,3:Nj-1)-Q(:,2:Nj-2),Q(:,4:Nj)-Q(:,3:Nj-1));

    % Compute the numerical fluxes
    F(:,1:Nj-3) = F_numeric(Q_L,Q_R,parameters);
    
    % Compute the cell sizes
    x_face(1:Nj-1) = x(1:Nj-1)+(x(2:Nj)-x(1:Nj-1))/2;
    dx(1:N,1:Nj-2) = ones(N,1)*(x_face(:,2:Nj-1)-x_face(:,1:Nj-2));
    
    % Compute the right hand side for the current time step of the ODE
    % RHS is computed for the interior cells
    % RHS is set to zero for ghost cells
    RHS = zeros(N,Nj);
    RHS(:,3:Nj-2) = -1/dx*(F(:,2:Nj-3)-F(:,1:Nj-4));

end


end


function F = F_numeric(Q_L,Q_R,parameters)

    % Use the desired space discretization method
    if strcmp(parameters.space_method,'godunov') == 1
        % Compute the numerical fluxes with the Godunov method
        F = F_Godunov(Q_L,Q_R,parameters);
        
    elseif strcmp(parameters.space_method,'roe') == 1
        % Compute the numerical fluxes with the Roe method
        F = F_Roe(Q_L,Q_R,parameters);
        
    elseif strcmp(parameters.space_method,'rusanov') == 1
        % Compute the numerical fluxes with the Rusanov method
        F = F_Rusanov(Q_L,Q_R,parameters);

    elseif strcmp(parameters.space_method,'lax-friedrichs') == 1
        % Compute the numerical fluxes with the Lax-Friedrichs method
        F = F_LaxFriedrichs(Q_L,Q_R,parameters);

    else
        error('Choose a valid space method')
        
    end

end



function F = F_Godunov(Q_L,Q_R,parameters)

% Compute the exact solution of the face Riemann problem
[N,Nx] = size(Q_L); q_Godunov = zeros(N,Nx);
for i = 1:Nx
    q_Godunov(:,i) = parameters.exact_solution(Q_L(:,i),Q_R(:,i));
end

% Compute the Godunov numerical flux
F = parameters.flux_function(q_Godunov);

end

function F = F_Roe(Q_L,Q_R,parameters)

% Compute the left and right fluxes
F_L = parameters.flux_function(Q_L);
F_R = parameters.flux_function(Q_R);
F   = 0*F_L;

% Compute the Roe numerical flux
for i = 1:length(Q_L)
    A = parameters.roe_function(Q_L(:,i),Q_R(:,i));
    F(:,i) = 1/2*(F_L(:,i)+F_R(:,i)-A*(Q_R(:,i)-Q_L(:,i)));
end


end


function F = F_Rusanov(Q_L,Q_R,parameters)

% Compute the left and right fluxes
F_L = parameters.flux_function(Q_L);
F_R = parameters.flux_function(Q_R);

% Compute the characteristic speed
a1 = max(abs(parameters.eigenvalues_function(Q_L)));
a2 = max(abs(parameters.eigenvalues_function(Q_R)));
a = max(a1,a2);
a = [a; a; a];

% Compute the Rusanov numerical flux
F  = 1/2*(F_L+F_R-a.*(Q_R-Q_L));

end

function F = F_LaxFriedrichs(Q_L,Q_R,parameters)

% Load parameters
dx = parameters.dx;
dt = parameters.dt;

% Compute the left and right fluxes
F_L = parameters.flux_function(Q_L);
F_R = parameters.flux_function(Q_R);

% Compute the Lax-Friedrichs numerical flux
F  = 1/2*(F_L+F_R-dx/dt*(Q_R-Q_L));

end




function MinMod_Limiter = minmod(a,b)
% MinMod limiter function for the MUSCL approach
MinMod_Limiter = sign(a).*max(0,min(abs(a),sign(a).*b));
end








