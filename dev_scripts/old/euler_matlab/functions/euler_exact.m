function q = euler_exact(x,t,q_L,q_R,g)
%% Description of the code
% Author: Roberto Agromayor
% Date: 30/11/2017

% The parameters required to build the solution are first computed using
% the general formulas given in LeVeque Chapter 14
% Then the solutions are constructed for the four possible configurations:
%
% Case A: The 1-wave is a expansion wave and the 3-wave is a shock wave
% Case B: The 1-wave is a shock wave and the 3-wave is a expansion wave
% Case C: The 1-wave is a shock wave and the 3-wave is a shock wave
% Case D: The 1-wave is a expansion wave and the 3-wave is a expansion wave
%
% This may not be the most elegant to program it but I think that it makes
% it easy to undestand the code


%% Obtain all the parameters of the solution
% State to the left in primitive variables
d_L = q_L(1);
u_L = q_L(2)/q_L(1);
p_L = (g-1)*(q_L(3)-1/2*q_L(2)^2/q_L(1));
c_L = sqrt(g*p_L/d_L);
V_L = [d_L; u_L; p_L];

% State to the right in primitive variables
d_R = q_R(1);
u_R = q_R(2)/q_R(1);
p_R = (g-1)*(q_R(3)-1/2*q_R(2)^2/q_R(1));
c_R = sqrt(g*p_R/d_R);
V_R = [d_R; u_R; p_R];


% Solve an non-linear scalar equation to find the pressure and velocity
% between the shock and the rarefaction wave
b = (g+1)/(g-1);
phi_error = @(p) phi_L(p,V_L,g)-phi_R(p,V_R,g);
% p_star = fzero(phi_error,[0 5*max(p_R,p_L)]);
p_star = fzero(phi_error,sqrt(p_R*p_L));
u_star = phi_R(p_star,V_R,g);


% Determine the density after the 1-wave
if p_star < p_L
    % Rerefaction wave relation
    d_star_L = d_L*(p_star/p_L)^(1/g);
elseif p_star >= p_L
    % Shock wave relation
    d_star_L = d_L*(1+b*p_star/p_L)/(b+p_star/p_L);
    % Shock speed of the 1-shock
    s_L = (d_star_L*u_star-d_L*u_L)/(d_star_L-d_L+eps);
end


% Determine the density before the 3-wave
if p_star < p_R
    % Rerefaction wave relation
    d_star_R = d_R*(p_star/p_R)^(1/g);
elseif p_star >= p_R
    % Shock wave relation
    d_star_R = d_R*(1+b*p_star/p_R)/(b+p_star/p_R);
    % Shock speed of the 1-shock
    s_R = (d_star_R*u_star-d_R*u_R)/(d_star_R-d_R+eps);
end


% Determine the speed of sound before and after the contact discontinuity
c_star_R = sqrt(g*p_star/d_star_R);
c_star_L = sqrt(g*p_star/d_star_L);


%% Case A - 1-rarefaction and 3-shock
if p_star <= p_L && p_star >= p_R

    if x/t <= u_L-c_L
        d = d_L;
        u = u_L;
        p = p_L;
        
    elseif x/t > u_L-c_L && x/t <= u_star-c_star_L
        u = ((g-1)*u_L+2*(c_L+x/t))/(g+1);
        d = (d_L^g*(u-x/t)^2/(g*p_L))^(1/(g-1));
        p = p_L/d_L^g*d^g;
        
    elseif x/t > u_star-c_star_L && x/t <= u_star
        d = d_star_L;
        u = u_star;
        p = p_star;
        
    elseif x/t > u_star && x/t <= s_R
        d = d_star_R;
        u = u_star;
        p = p_star;
        
    elseif x/t > s_R
        d = d_R;
        u = u_R;
        p = p_R;
        
    end
    
end


%% Case B - 1-shock and 3-rarefaction
if p_star >= p_L && p_star <= p_R
    
    if x/t <= s_L
        d = d_L;
        u = u_L;
        p = p_L;
        
    elseif x/t > s_L && x/t <= u_star
        d = d_star_L;
        u = u_star;
        p = p_star;
        
    elseif x/t > u_star && x/t <= u_star+c_star_R
        d = d_star_R;
        u = u_star;
        p = p_star;        
        
    elseif x/t > u_star+c_star_R && x/t <= u_R+c_R
        u = ((g-1)*u_R-2*(c_R-x/t))/(g+1);
        d = (d_R^g*(u-x/t)^2/(g*p_R))^(1/(g-1));
        p = p_R/d_R^g*d^g;
        
    elseif x/t > u_R+c_R 
        d = d_R;
        u = u_R;
        p = p_R;
        
    end
    
end


%% Case C - 1-shock and 3-shock
if p_star >= p_L && p_star >= p_R

    if x/t <= s_L
        d = d_L;
        u = u_L;
        p = p_L;
        
    elseif x/t > s_L && x/t <= u_star
        d = d_star_L;
        u = u_star;
        p = p_star;
        
    elseif x/t > u_star && x/t <= s_R
        d = d_star_R;
        u = u_star;
        p = p_star;
        
    elseif x/t > s_R
        d = d_R;
        u = u_R;
        p = p_R;
        
    end
    
end


%% Case D - 1-rearefaction and 3-rarefaction
if p_star <= p_L && p_star <= p_R

    if x/t <= u_L-c_L
        d = d_L;
        u = u_L;
        p = p_L;
        
    elseif x/t > u_L-c_L && x/t <= u_star-c_star_L
        u = ((g-1)*u_L+2*(c_L+x/t))/(g+1);
        d = (d_L^g*(u-x/t)^2/(g*p_L))^(1/(g-1));
        p = p_L/d_L^g*d^g;
        
    elseif x/t > u_star-c_star_L && x/t <= u_star
        d = d_star_L;
        u = u_star;
        p = p_star;

    elseif x/t > u_star && x/t <= u_star+c_star_R
        d = d_star_R;
        u = u_star;
        p = p_star;        
        
    elseif x/t > u_star+c_star_R && x/t <= u_R+c_R
        u = ((g-1)*u_R-2*(c_R-x/t))/(g+1);
        d = (d_R^g*(u-x/t)^2/(g*p_R))^(1/(g-1));
        p = p_R/d_R^g*d^g;
        
    elseif x/t > u_R+c_R 
        d = d_R;
        u = u_R;
        p = p_R;
        
    end
    
end


%% Convert the solution to conservative variables
q = [d; d*u; p/(g-1)+d*u^2/2];
    
end


function phi = phi_L(p,V_L,g)
% Load variables
d_L = V_L(1);
u_L = V_L(2);
p_L = V_L(3);
c_L = sqrt(g*p_L/d_L);
b = (g+1)/(g-1);

if p <= p_L
    % Rerefaction wave relation
    phi = u_L + 2*c_L/(g-1)*(1-(p/p_L).^((g-1)/(2*g)));
elseif p > p_L
    % Shock wave relation
    phi = u_L + 2*c_L*((1-(p/p_L))/sqrt(1+b*(p/p_L)))/sqrt(2*g*(g-1));
end

end

function phi = phi_R(p,V_R,g)

% Load variables
d_R = V_R(1);
u_R = V_R(2);
p_R = V_R(3);
c_R = sqrt(g*p_R/d_R);
b = (g+1)/(g-1);

if p <= p_R
    % Rerefaction wave relation
    phi = u_R - 2*c_R/(g-1)*(1-(p/p_R).^((g-1)/(2*g)));
elseif p > p_R
    % Shock wave relation
    phi = u_R - 2*c_R*((1-(p/p_R))/sqrt(1+b*(p/p_R)))/sqrt(2*g*(g-1));
end

end