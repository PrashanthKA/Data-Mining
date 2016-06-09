% Write simple classifiers with KNN, LDA, Centroid, SVM 
% through 5 fold cross validation

clc
clear all
close all

%% Loading Data File
% loading Face_Data file
fid = fopen('ATNTFaceImage400.txt');
ATNTFaceImage400 = [];
while ~feof(fid)
    one_row = textscan(fid, '%d', 400, 'delimiter', ',');
    ATNTFaceImage400 = [ATNTFaceImage400; one_row{1}'];
end
fclose(fid);
ATNTFaceImage400 = ATNTFaceImage400';
Face_Group = ATNTFaceImage400(:, 1);
Face_Data = double(ATNTFaceImage400(:, 2 : end));

trIdx = 0;
teIdx = 0;
% train = zeros(size(Face_Data));
% test = train;
% trLabel = zeros(size(Face_Group));
% teLabel = trLabel;
for i = 1:size(Face_Data,1)
    if rand<0.65
        trIdx = trIdx + 1;
        train(trIdx,:) = Face_Data(i,:);
        trLabel(trIdx,1) = Face_Group(i,1);
    else
        teIdx = teIdx + 1;
        test(teIdx,:) = Face_Data(i,:);
        teLabel(teIdx,1) = Face_Group(i,1);
    end
end

u = unique(trLabel);
% KNN Classification
knnModel = fitcknn(train, trLabel);
knnLabel = predict(knnModel, test);
accuracy_KNN= sum(knnLabel==teLabel)/numel(teLabel);

% SVM Classification
Scores = zeros(size(test,1), numel(u));
    for k = 1:numel(u)
        trGroup = (trLabel==u(k));
        svmModel{k} = fitcsvm(train, trGroup, 'Standardize',true,...
                            'KernelFunction','linear','KernelScale','auto');
    end
    for k = 1:numel(u)
        [~, score] = predict(svmModel{k},test);
        Scores(:,k) = score(:,2);
    end
    for k = 1:size(test,1)
        [~, id] = sort(Scores(k,:), 'descend');
         svmLabel(k,1) = id(1);
    end
    accuracy_SVM = sum(svmLabel==teLabel)/teIdx;
       
 % Centroid Classification
 center = zeros(numel(u), size(train,2));
     for k = 1:numel(u)
         tGroup = (trLabel==u(k));
         center(k,:) = mean(train(tGroup,:));
     end
     for t = 1:size(test,1)
         for k = 1:numel(u)
             distance(k) = sum((test(t,:)-center(k,:)).^2).^0.5;
         end
         [~, id] = sort(distance);
         cenLabel(t,1) = id(1);
     end
     accuracy_CENTROID = sum(cenLabel==teLabel)/numel(teLabel);
 
                                          
% Linear Regression Classification
%    Scores = zeros(size(test,1), numel(u));
%    for k = 1:numel(u)
%        trGroup = (trLabel==u(k));
%        lrModel = fitlm(train, trGroup);
%        Scores(:,k) = predict(lrModel, test);
%    end
%    for k = 1:size(test,1)
%        [~, id] = sort(Scores(k,:));
%        lrLabel(k,1) = id(1);
%    end
%    accuracy_LR(i,j) = sum(lrLabel==teLabel)/numel(teLabel);
     obj = fitcdiscr(train, trLabel);
     lrLabel = predict(obj, test);
     accuracy_LR = sum(lrLabel==teLabel)/numel(teLabel);

figure(1)
plot([1 10], [accuracy_SVM*100 accuracy_SVM*100],'-rs');
grid on
hold on
plot([1 10], [accuracy_KNN*100 accuracy_KNN*100],'-bs');
hold on
plot([1 10], [accuracy_LR*100 accuracy_LR*100],'-ms');
hold on
plot([1 10], [accuracy_CENTROID*100 accuracy_CENTROID*100],'-gs');
hold on
title('Accuracy of KNN, SVM, LR and Centroid methods on Original data')
legend('SVM classifier', 'KNN classifier','LR Classifier','Centroid classifier');

