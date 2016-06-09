xTest = csvread('testDataX.txt');
train = csvread('trainDataXY.txt'); 
y = train(1,:);             % yTrain data
x = train(2:end,:);         % xTrain data


matrix1 = zeros(5, 45);
len = size(y, 1);

arr_start = 1;
arr_end = 9;

for i=1:5
    matrix1(i, arr_start:arr_end)=1;
    arr_start = arr_start+9;
    arr_end = arr_end+9;
end

% x = double(x);
% xTrans = transpose(x);
% yTrans = transpose(y);
% 
 B = pinv(x') * double(matrix1');
%BTrans = transpose(B);
% 
% y1 = BTrans * x;
ytest1 = B' * xTest;
% 
[Ytest2value Ytest2] = max(ytest1,[],1);
% [Ytrain2value Ytrain2] = max(y1,[],1);

%plot(Ytest2value, ytest1);
disp(Ytest2);    

