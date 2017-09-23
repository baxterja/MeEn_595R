clear all
close all

load('hw1_soln_data.mat');

m = 100;
b = 20;
Ts = .05;

A_C = [0 1;0  -b/m];
B_C = [0; 1/m];
C_C = [1 0];
D_C = 0;

SYSC = ss(A_C,B_C,C_C,D_C);
SYSD = c2d(SYSC,Ts);

A = SYSD.A;
B = SYSD.B;
C = SYSD.C;
D = SYSD.D;

Time = t;%0:Ts:40;

states_actual = [xtr;vtr]%zeros(2,length(Time)+1);
states_predicted = zeros(2,length(Time)+1);
kalman_gains = zeros(2,length(Time));
covariances = zeros(2,length(Time)+1);

index_counter = 1

R = flipud(fliplr(R));%diag([.0001,.01]);
%Q = .001;

Sigma = .000001*eye(2);
covariances(1,1) = Sigma(1,1);
covariances(2,1) = Sigma(2,2);

estimated_state =flipud(mu0);%[0;0];
states_predicted(:,1) = estimated_state;

u_overtime = u;
for t = Time
%     if t<5
%         u = 50;
%     elseif t>=25 && t<30
%         u = -50;
%     else
%         u = 0;
%     end
    u = u_overtime(index_counter);
%     current_state = states_actual(:,index_counter);
%     next_state = A*current_state+B*u+mvnrnd([0;0],R)';
%     states_actual(:,index_counter+1) = next_state;
    
    estimated_state  = A*estimated_state+B*u;
    Sigma = A*Sigma*A'+R;
    K = Sigma*C'/(C*Sigma*C'+Q);
    estimated_state = estimated_state+...
        K*(z(index_counter)-C*estimated_state);
    Sigma = (eye(2)-K*C)*Sigma;
    kalman_gains(:,index_counter) = K;
    covariances(:,index_counter+1) = diag(Sigma);
    
    states_predicted(:,index_counter+1)=estimated_state;
    
    index_counter = index_counter + 1;
end


subplot(3,2,5);
plot(Time,kalman_gains(1,:));

subplot(3,2,6);
plot(Time,kalman_gains(2,:));

%

subplot(3,2,1);
plot(Time,states_actual(1,:));
hold on
plot(Time,states_predicted(1,1:end-1));

subplot(3,2,2);
plot(Time,states_actual(2,:));
hold on
plot(Time,states_predicted(2,1:end-1));

%Time = [Time max(Time)+Ts];
subplot(3,2,3);
plot(Time,covariances(1,1:end-1));
hold on
plot(Time,states_actual(1,:)-states_predicted(1,2:end));
plot(Time,2*sqrt(covariances(1,1:end-1)));
plot(Time,-2*sqrt(covariances(1,1:end-1)));
ylim([-1 1]);

subplot(3,2,4);
plot(Time,covariances(2,1:end-1));
hold on
plot(Time,states_actual(2,:)-states_predicted(2,2:end));
plot(Time,2*sqrt(covariances(2,1:end-1)));
plot(Time,-2*sqrt(covariances(2,1:end-1)));
ylim([-1 1]);