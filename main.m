clear all; close all; 

%%%%%%%%%%%%%%%%%%%%%%%%%%
% Make system
%%%%%%%%%%%%%%%%%%%%%%%%%%
n = 5; m = 2; 
  
% *** system ***
while 1
  sys = drss(n,1,m); if max(eig(sys.A)) < 1; break; end
end
A = sys.A; B = sys.B; 

% *** initial state***
x0 = rand(n,1);

% *** Exp noise ***
Nw = 100; 
noise_f = (rand(m,Nw) - 0.5)*50; 
noise_b = -pi + rand(m,Nw) * 2*pi; 

%%%%%%%%%%%%%%%%%%%%%%%%%%
% Get data
%%%%%%%%%%%%%%%%%%%%%%%%%%
N = 2000; 
t = [0:N]'; 

% *** input signal ***
u = zeros(m,N+1); 
for i=1:N+1
  u(:,i) = 1.0*sum(sin(noise_f * t(i) + noise_b), 2); 
end
u = u'; 

% *** simulation ***
[y, t, x] = lsim(sys,u',t,x0); 

fprintf('Data Collection is Done. \n'); 

%%%%%%%%%%%%%%%%%%%%%%%%%%
% Process Data
%%%%%%%%%%%%%%%%%%%%%%%%%%
Ixx1 = zeros(N, n^2);  
Ixx2 = zeros(N, n^2);  
Ixw = zeros(N, n*m); 
Iww = zeros(N, m^2);  

for i=1:N
  Ixx1(i,:) = kron(x(i,:), x(i,:)); 
  Ixx2(i,:) = kron(x(i+1,:), x(i+1,:)); 
  Ixw(i,:) = kron(x(i,:), u(i,:)); 
  Iww(i,:) = kron(u(i,:), u(i,:)); 
end

fprintf('Data Computation is Done. \n'); 

%%%%%%%%%%%%%%%%%%%%%%%%%%
% Model-based Optimal Control
%%%%%%%%%%%%%%%%%%%%%%%%%%
Q = 1000*eye(n); R = eye(m);

[Kopt, Popt] = dlqr(A,B,Q,R); 
Kopt = -Kopt; 
Sopt = B'*Popt*B; 
Mopt = (R+Sopt)*Kopt;

%%%%%%%%%%%%%%%%%%%%%%%%%%
% Check equations
%%%%%%%%%%%%%%%%%%%%%%%%%%
Theta_opt = [Ixx1-Ixx2, Iww-Ixx1*kron(Kopt',Kopt'), 2*(Ixx1*kron(eye(n),Kopt')-Ixw)]; 
QKopt = Q + Kopt'*R*Kopt; 
Xi_opt = Ixx1*QKopt(:); 

e1 = Xi_opt - Theta_opt*[Popt(:); Sopt(:); Mopt(:)];
if norm(e1) > 1e-4; disp(norm(e1)); end;

%%%%%%%%%%%%%%%%%%%%%%%%%%
% Learning
%%%%%%%%%%%%%%%%%%%%%%%%%%
[Pc, Kc] = Off_PI(Ixx1, Ixx2, Ixw, Iww, Q, R); 

fprintf('\n === End === \n\n'); 

K = Kc{end};

% Show results
disp(['K', '=']); disp(K);
disp(['Kopt', '=']); disp(Kopt);

