
pot = @(x,y) (1./(x-y).^(12)) - (2./( x-y).^6); % Lennard Jones Potential
dpot = @(x,y) (12 ./ (x-y).^7 - 12 ./ ( x-y).^13); % first derivative
d2pot = @(x,y) (-84 ./ (x-y).^8 + 156 ./ ( x-y).^14); % 2nd derivative

%dpot = @(x,y) 2*(x-y);     % simple potentials for debugging
%d2pot  = @(x,y) repmat(2,length(x),1);
%d2_pot = 2;

% Make a model via randomly choosing starting  positions for 5 atomistic
% and 12 continuum repatoms. No Coarseness in contin. region
% Indexing: Continuum = \{1,2,3,4,5,6\} \cup \{ 12,13,14,15,16,17\} and
% Atomistic = \{ 7,8,9,10,11\}
%  orginal positions given by the zero vector. 

%u = -0.005 + (0.01).*rand(17,1);
% positions = VECTORofPOSITIONS
u = zeros(17,1);
%u(6) = 0.001;
%u(10)= -0.01;


f=zeros(17,1);
% add forcing on atoms to see deformation over time.
f(1) = 0.1 ; f(2) = 0.02; f(6) = -0.01; f(11) = -0.01 ; f(12) = -0.1;
ind = (1:17)';
colors = [repmat([0,0,1],6,1) ; repmat([1,0,0],5,1) ; repmat([0,0,1],6,1)];
subplot(2,1,1)
scatter(ind,u,[], colors,'filled')
title('Initial')
ylabel('displacement')
xlabel('Position, red = atomic, blue = continuum')
hold on
%Solve for zero external forcing, i.e., d/dx_i ( E) = 0 for each ith
%position, and energy at that position. 


%solve for the positions such that forces =0 using Newtons's Method
indp = circshift(ind, -1);
indpp = circshift(ind, -2);

indm = circshift(ind, 1);
indmm = circshift(ind, 2);

% Indicator functions for atomistic and continuum regions
Xa = diag([zeros(6,1);ones(5,1);zeros(6,1)]);

Xc = eye(17,17) - Xa;
P = eye(17,17) - 1/17 * ones(17,17);

% Gradients and Hessians for the system
Grad = @(u) P*Xa * ( - dpot(1+u(indp,:), u(ind,:)) + dpot(1+u(ind,:), u(indm,:)) ...
     - dpot(2+u(indpp,:), u(ind,:)) + dpot(2+u(ind,:), u(indmm,:)) ) ...
     + P*Xc * ( - dpot(1+u(indp,:), u(ind,:)) + dpot(1+u(ind,:), u(indm,:)) ...
     - 2*dpot(2+2*u(indp,:), 2*u(ind,:)) + 2*dpot(2+2*u(ind,:), 2*u(indm,:)) );
  
atmH = @(u) circshift(diag(-d2pot(2+u(ind,:), u(indmm,:))),-2,2) + ...
            circshift(diag(-d2pot(1+u(ind,:), u(indm,:))),-1,2) + ...
            diag(d2pot(1+u(indp,:), u(ind,:)) + d2pot(2+u(indpp,:), u(ind,:)) ...
               + d2pot(1+u(ind,:), u(indm,:)) + d2pot(2+u(ind,:), u(indmm,:))) ...
          + circshift(diag(-d2pot(1+u(indp,:), u(ind,:))),1,2) ...
          + circshift(diag(-d2pot(2+u(indpp,:), u(ind,:))),2,2);
contH = @(u)  circshift(diag(-d2pot(1+u(indp,:), u(ind,:)) ...
                           -4*d2pot(2+2*u(indp,:), 2*u(ind,:))),1,2) ...
               + diag(d2pot(1+u(indp,:), u(ind,:)) + 4*d2pot(2+2*u(indp,:), 2*u(ind,:)) ...
                    + d2pot(1+u(ind,:), u(indm,:)) + 4*d2pot(2+2*u(ind,:), 2*u(indm,:))) ...
               + circshift(diag(-d2pot(1+u(ind,:), u(indm,:)) ...
                           -4*d2pot(2+2*u(ind,:), 2*u(indm,:))),-1,2);     
H = @(u) P*(Xa * atmH(u) + Xc * contH(u));

delta = 1e-6;
fdH = @(u) (Grad(u*ones(1,17) + delta*eye(17,17)) ...
    - Grad(u*ones(1,17)-  delta*eye(17,17)) )/ (2*delta);

% Newton's Solver
 N = 100; 
for i = 1:N
    % Iteration stepL xn+1 = xn -f(xn)/f'(xn)
   u(:,i+1) = u(:,i) - (H(u(:,i)) + ones(17,17))\(Grad(u(:,i))-f);
    
      
end

% Plotter
subplot(2,1,2)
scatter(ind,u(:,end),[], colors,'filled')
title('Final')
ylabel('displacement')
xlabel('Position, red = atomic, blue = continuum')
suptitle({'Deformation from u(6)  = 0.001, u(10) = -0.01 with forcing'; 'f(1) =0.1, f(2) = 0.02, f(6) = -0.1, f(11) = -0.01, f(12) = -0.01'})
hold off



# Atomic_Continuum_modellling
