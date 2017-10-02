clear all
close all

num_landmarks = 6;
landmarks =[[6 -7 6 1 0 10];...%x
            [4  8 -4 -8 0 -19];...%y
            [1  2 3 4 5 6]];%index
        
Ts = .1;
time = 0:Ts:20;
x = zeros(3,length(time)+1);
u = zeros(2,length(time));
x_estimate = zeros(3,length(time)+1);
sigma_estimate = zeros(3,length(time)+1);
z = zeros(3,num_landmarks*length(time));
x_0 = [-5;-3;pi/2];
x(:,1) = x_0;
% alpha1 = .1^2;
% alpha2 = .01^2;
% alpha3 = .01^2;
% alpha4 = .1^2;
alpha1 = .1;
alpha2 = .01;
alpha3 = .01;
alpha4 = .1;
sig_r = .1^2;
sig_phi = .05^2;

x_curr = x_0;
z_temp = zeros(3,1);
for i = 1:length(time)
    t = time(i);
    theta = x_curr(3);
    v = 1+.5*cos(2*pi*.2*t);
    w = -.2+2*cos(2*pi*.6*t);
    v = v + mvnrnd(0,alpha1*v.^2+alpha2*w.^2);
    u(:,i)= [v;w];
    w = w + mvnrnd(0,alpha3*v.^2+alpha4*w.^2);
%     if(abs(w)<.3)
%         w = sign(w)*.1;
%     end
    
    x_curr = x_curr + [-v/w*sin(theta)+v/w*sin(theta+w*Ts);...
                       v/w*cos(theta)-v/w*cos(theta+w*Ts);...
                       w*Ts];
    theta = x_curr(3);
    for k = 1:num_landmarks
        mx = landmarks(1,k);
        my = landmarks(2,k);
        z_temp(1) = sqrt((x_curr(1)-mx)^2+(x_curr(2)-my)^2)+mvnrnd(0,sig_r);
        z_temp(2) = atan2(my-x_curr(2),mx- x_curr(1))-x_curr(3)+mvnrnd(0,sig_phi);
        z_temp(3) = k;
        z(:,num_landmarks*(i-1)+k)= z_temp;
    end
    x(:,i+1) = x_curr;
end

% angle = x(3,:);
% while((angle<0)>0)
%    angle(angle<0) = angle(angle<0)+2*pi; 
% end
% while((angle>2*pi)>0)
%    angle(angle>2*pi) = angle(angle>2*pi)-2*pi; 
% end
% x(3,:) = angle;
% for i = 1:length(time)
%     plot(x(1,:),x(2,:));
%     hold on
%     scatter(landmarks(1,:),landmarks(2,:));
%     z_temp = z(:,num_landmarks*(i-1)+1:num_landmarks*(i));
%     x_c = x(1,i+1);
%     y_c = x(2,i+1);
%     theta = x(3,i+1);
%     x_temp = [x_c x_c+.5*cos(theta:pi/8:theta+2*pi)];
%     y_temp = [y_c y_c+.5*sin(theta:pi/8:theta+2*pi)];
%     plot(x_temp,y_temp);
%     for k = 1:6
%         plot([x_c x_c-z_temp(1,k)*cos(z_temp(2,k)+theta+pi)],...
%              [y_c y_c-z_temp(1,k)*sin(z_temp(2,k)+theta+pi)]);
%     end
%     hold off
% end

x_est_0 = x_0;
x_estimate(:,1) = x_est_0;
Sigma = .01*eye(3);
sigma_estimate(:,1) = diag(Sigma);
mu = x_estimate(:,1);
for i = 1:length(time)
    
    [mu, Sigma, p] = EKF_update(mu, Sigma, u(:,i), z(:,num_landmarks*(i-1)+1:num_landmarks*i), landmarks);
    x_estimate(:,i+1) = mu;
    sigma_estimate(:,i+1) = diag(Sigma);
end
plot(x(1,:),x(2,:));
hold on
scatter(landmarks(1,:),landmarks(2,:));
plot(x_estimate(1,:),x_estimate(2,:));
figure
plot(x_estimate(1,:)-x(1,:));
hold on
plot(2*sqrt(sigma_estimate(1,:)))
plot(-2*sqrt(sigma_estimate(1,:)))

figure
plot(x_estimate(2,:)-x(2,:));
hold on
plot(2*sqrt(sigma_estimate(2,:)))
plot(-2*sqrt(sigma_estimate(2,:)))

figure
plot(x_estimate(3,:)-x(3,:));
hold on
plot(2*sqrt(sigma_estimate(3,:)))
plot(-2*sqrt(sigma_estimate(3,:)))

function [mu, Sigma, p] = EKF_update(mu, Sigma, u, z, m)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
    Ts = .1;
%     alpha1 = .1^2;
%     alpha2 = .01^2;
%     alpha3 = .01^2;
%     alpha4 = .1^2;
    
    alpha1 = .1;
    alpha2 = .01;
    alpha3 = .01;
    alpha4 = .1;
    sig_r = .1^2;
    sig_phi = .05^2;
    
    theta = mu(3);
    v = u(1);
    w = u(2);
    G = [[1 0 -v/w*cos(theta)+v/w*cos(theta+w*Ts)];...
         [0 1 -v/w*sin(theta)+v/w*sin(theta+w*Ts)];...
         [0 0 1]];
    V = [[-sin(theta)+sin(theta+w*Ts)/w v*(sin(theta)-sin(theta+w*Ts))/w.^2+v*cos(theta+w*Ts)*Ts/w];...
         [cos(theta)-cos(theta+w*Ts)/w -v*(cos(theta)-cos(theta+w*Ts))/w.^2+v*sin(theta+w*Ts)*Ts/w];...
         [0 Ts]];
    M = diag([alpha1*v.^2+alpha2*w.^2, alpha3*v.^2+alpha4*w.^2]);
    mu = mu + [-v/w*sin(theta)+v/w*sin(theta+w*Ts);...
               v/w*cos(theta)- v/w*cos(theta+w*Ts);...
               w*Ts];
           

%     while((mu(3)<0))
%        mu(3) = mu(3)+2*pi; 
%     end
%     while(mu(3)>2*pi)
%        mu(3) = mu(3)-2*pi; 
%     end
    
           
    Sigma = G*Sigma*G.'+V*M*V.';
    
    
    Q = diag([sig_r sig_phi]);% 1e-8]);
    
    [three, num_observed_featurs] = size(z);
    p_z = zeros(1,num_observed_featurs);
    for i = 1:num_observed_featurs
        mx = m(1,i);
        my = m(2,i);
        pillar = m(3,i);
%         %j = c(:,i);
        q = (mx-mu(1)).^2+(my-mu(2)).^2;
        z_hat = [sqrt(q);atan2(my-mu(2),mx-mu(1))-mu(3)];%;m(3,i)];
        H = [[-(mx-mu(1))/sqrt(q) -(my-mu(2))/sqrt(q) 0];...
             [(my-mu(2))/q -(mx-mu(1))/q -1]];%...
             %[0 0 0]];
        S = H*Sigma*H.'+Q;
        K = Sigma*H.'/S;
        error = z(1:2,i)-z_hat;
        mu = mu + K*(z(1:2,i)-z_hat);
%         while((mu(3)<0))
%            mu(3) = mu(3)+2*pi; 
%         end
%         while(mu(3)>2*pi)
%            mu(3) = mu(3)-2*pi; 
%         end
        Sigma = (eye(length(Sigma))-K*H)*Sigma;
        %p_z(i) = 1/sqrt(2*pi*S)*exp(-1/2*(z(:,i)-z_hat).'/S*(z(:,i)-z_hat));
        %p = 1;
    end
    p = 1;%prod(p_z(i));
end