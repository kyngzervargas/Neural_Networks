clear all;
% close all;

% load 'apple(hue,ecc).mat'   
% load 'banana(hue,ecc).mat'
% load 'orange(hue,ecc).mat'

%%
% Data = [apple(1:10,:);banana(1:5,:);orange(1:10,:)];
% x1 = apple(1:20,1);
% x2 = apple(1:20,2);
%% DATA POINTS
x1 = linspace(0,2*pi,500);
x2 = f(x1,2);
x3 = f(x1,3);
x4 = f(x1,4);
x5 = f(x1,5);
x6 = f(x1,6);
x7 = f(x1,7);
x8 = f(x1,8);
x9 = f(x1,9);
x10 = f(x1,10);
x11 = f(x1,11);

expected = sin(x1);
% expected = (x1);

%Normalize Values
x1 = n(x1);
x2 = n(x2);
x3 = n(x3);
x4 = n(x4);
x5 = n(x5);
x6 = n(x6);
x7 = n(x7);
x8 = n(x8);
x9 = n(x9);
x10 = n(x10);
x11 = n(x11);

expected = n(expected);

%% TRAINING

%Data and weights
training = [x1;x2;x3;x4;x5;x6;x7;x8;x9;x10;x11]';
% training = [x1,x2];
[N F] = size(training);
output = 1;
wji = rand(output,F+1);

%List
E_list = [];
Erms_list = [];

%Variables
epoch = 0;
eta = 0.1;
min_error = 0.03;
iterations = 0;

%% NN part
while iterations <= 1e5
    dE = zeros(size(wji));
    Eq = 0;
    for pt = 1:N
        xi = [1,training(pt,:)];
        aj = dot(wji,xi);
        zj = g(aj);
        dj = gprime(aj)*(zj-expected(pt));
        dEq = dj*xi;
        dE = dE + dEq;
        Eq = Eq + (zj-expected(pt))*(zj-expected(pt));
    end
    E_list = [E_list,Eq/2];
    Erms_list = [Erms_list,sqrt(Eq/N)];
    wji = wji - eta*dE;
    epoch = epoch + 1;
    iterations = iterations + 1;
%     if Erms_list(end) <= min_error
%         break
%     end
    
end

%% TESTING THE NN
% x1t = apple(11:20,1);
% x2t = apple(11:20,2);
% x1t = apple(1:20,1);
% x2t = apple(1:20,2);

x1t = linspace(0,2*pi,500);
x2t = f(x1t,2);
x3t = f(x1t,3);
x4t = f(x1t,4);
x5t = f(x1t,5);
x6t = f(x1t,6);
x7t = f(x1t,7);
x8t = f(x1t,8);
x9t = f(x1t,9);
x10t = f(x1t,10);
x11t = f(x1t,11);
% 
expectedt = sin(x1t);
% expectedt = (x1t);
% 
x1t = n(x1t);
x2t = n(x2t);
x3t = n(x3t);
x4t = n(x4t);
x5t = n(x5t);
x6t = n(x6t);
x7t = n(x7t);
x8t = n(x8t);
x9t = n(x9t);
x10t = n(x10t);
x11t = n(x11t);
% 
expectedt = n(expectedt);
%% TEST SET

test = [x1;x2;x3;x4;x5;x6;x7;x8;x9;x10;x11]';
% test = [x1t,x2t];
[Nt Ft] = size(test);

test_out = [];
for pt = 1:Nt
    xi = [1,test(pt,:)];
    aj = dot(wji,xi);
    zj = g(aj);
    test_out = [test_out,zj(1)];
end

%% FIGURES

figure;
plot(x1t,expectedt,'k');
% scatter(x1t,expectedt,'ksq');
hold on
plot(x1t,test_out,'b');
% scatter(x1t,test_out,'rd');
title('Neutral network for classification');
% xlabel('Hue');
% ylabel('Eccentricity');
xlabel('x-axis');
ylabel('y-axis');
legend('Theoretical','Predicted');

%% FUNCTIONS

%Sigmoid
function output = g(a)
output = 1/(1+exp(-a));
end

function output = gprime(a)
output = g(a)*(1-g(a));
end

%For normalization
function output = n(x)
output = (x-min(x))/(max(x)-min(x));
end

%For polynomial
function output = f(x,n)
   output = x.^n;
end