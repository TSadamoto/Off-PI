function [P_cell, K_cell, varargout]=Off_PI(Ixx1, Ixx2, Ixw, Iww, Q, R)

% **** Preliminary ****
n = round(sqrt(size(Ixx1, 2))); %dim of state
m = round(sqrt(size(Iww, 2))); %dim of input

MaxIteration = 20; 
P_old = zeros(n);
P = eye(n)*0.1; %Anything can be ok
K_old = ones(m, n);
K = zeros(m,n); 
it = 0; 

P_cell = cell(1,MaxIteration); 
K_cell = cell(1,MaxIteration+1); 
K_cell{1} = K; 

% **** learning ****
while norm(K-K_old)>1e-4 & it<MaxIteration
  % *** preliminary for iterations *** 
  it = it+1;                        
  P_old = P;             
  K_old = K;             
  
  % *** Define equation *** 
  Theta = [Ixx1-Ixx2, Iww-Ixx1*kron(K',K'), 2*(Ixx1*kron(eye(n),K')-Ixw)]; 
  QK = Q + K'*R*K; 
  Xi = Ixx1*QK(:); 

  % *** Solve the equation *** 
  pp = Theta\Xi;

  % *** Generate matrices from the computed results  *** 
  P = reshape(pp(1:n*n), [n, n]);  
  P = (P + P')/2;
  S = reshape(pp(n^2+1:n^2+m^2), [m, m]);  
  S = (S + S')/2;
  M = reshape(pp(n^2+m^2+1:end), [m, n]);
  
  K = inv(R + S)*M; 
  
  % *** Show progress *** 
  disp(['K', num2str(it), '=']); disp(K);
  
  P_cell{it} = P; 
  K_cell{it+1} = K;   
end

P_cell(it+1:end) = []; 

if it < MaxIteration; K_cell(it+2:end) = []; end
