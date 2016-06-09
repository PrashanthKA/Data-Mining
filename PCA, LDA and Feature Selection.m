% Write simple classifiers with KNN, LDA, Centroid, SVM 
% through 5 fold cross validation

clc
clear all
close all


fprintf('        Please select the Reduction Method.\n');
fprintf(' 1: PCA method  2: LDA method  3: F_Statistics method\n');

method = input('Input method :  ');

switch method
    case 1
        mstr = ' on transformed data using PCA';
    case 2
        mstr = ' on transformed data using LDA';
    case 3
        mstr = ' on transformed data usign F-Statistics';
end

%% Loading Data File
% loading Face_Data file
fid = fopen('Gene.txt');
GeneDtaset = [];
while ~feof(fid)
    one_row = textscan(fid, '%f', 50, 'delimiter', ',');
    GeneDtaset = [GeneDtaset; one_row{1}'];
end
fclose(fid);
GeneDtaset = GeneDtaset';
Face_Group = GeneDtaset(:, 1);
Face_Data = double(GeneDtaset(:, 2 : end));

% % loading Hand_Data file
% fid = fopen('HandWrittenLetters.txt');
% HandWrittenLetters = [];
% while ~feof(fid)
%     one_row = textscan(fid, '%d', 1014,  'delimiter', ',');
%     HandWrittenLetters = [HandWrittenLetters; one_row{1}'];
% end
% fclose(fid);
% HandWrittenLetters = HandWrittenLetters';
% Hand_Group = HandWrittenLetters(:, 1);
% Hand_Data = HandWrittenLetters(:, 2 : end);

%% Create Cross-Validation Partition for Data (5-fold)
CVO = cvpartition(Face_Group, 'k', 5);
accuracy_KNN = zeros(CVO.NumTestSets, 1);
accuracy_SVM = zeros(CVO.NumTestSets, 1);
accuracy_CENTROID = zeros(CVO.NumTestSets, 1);
accuracy_LR = zeros(CVO.NumTestSets, 1);

dimArray = [5 10 15 20 25 30 40 50 100 200]; % reducing dimensionality array
if isempty(method) || ~isnumeric(method)
    method = 1;
end

%% Classify the transformed data using KNN, SVM, Centroid, LR Classifiers
for i = 1:CVO.NumTestSets
    trIdx = CVO.training(i);
    teIdx = CVO.test(i);
    trLabel = Face_Group(trIdx);
    teLabel = Face_Group(teIdx);
    u = unique(trLabel);
         
    for j = 1:numel(dimArray)
        dim = dimArray(j);
%% Reduce dimensionality of data using PCA, LDA, and F-Statistics
        switch method
            case 1
                % PCA method
                [coeff, ~, ~] = princomp(zscore(Face_Data(trIdx,:)));
                coeff = zscore(coeff);
                transMatrix = coeff(:,1:dim);
            case 2
                % LDA method
                transMatrix = reduction_LDA(Face_Data(trIdx,:), trLabel, dim);
            case 3
                % F_Statistics method
                transMatrix = FStatistics(Face_Data(trIdx,:), trLabel, dim);
        end

        train = Face_Data(trIdx, :)*transMatrix;
        test = Face_Data(teIdx, :)*transMatrix;
        
%% Classify transformed data using KNN, SVM, Centroid, LR Methods
        % KNN Classification
        knnModel = fitcknn(train, trLabel);
        knnLabel = predict(knnModel, test);
        accuracy_KNN(i,j)= sum(knnLabel==teLabel)/numel(teLabel);
        
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
        accuracy_SVM(i,j) = sum(svmLabel==teLabel)/numel(teLabel);
       
        % Centroid Classification
        center = zeros(numel(u), dim);
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
        accuracy_CENTROID(i,j) = sum(cenLabel==teLabel)/numel(teLabel);
 
                                          
        % Linear Regression Classification
%         Scores = zeros(size(test,1), numel(u));
%         for k = 1:numel(u)
%             trGroup = (trLabel==u(k));
%             lrModel = fitlm(train, trGroup);
%             Scores(:,k) = predict(lrModel, test);
%         end
%         for k = 1:size(test,1)
%             [~, id] = sort(Scores(k,:));
%             lrLabel(k,1) = id(1);
%         end
%         accuracy_LR(i,j) = sum(lrLabel==teLabel)/numel(teLabel);
        obj = fitcdiscr(train, trLabel);
        lrLabel = predict(obj, test);
        accuracy_LR(i,j) = sum(lrLabel==teLabel)/numel(teLabel);
        
    end       
end

mAccSvm = mean(accuracy_SVM);
mAccKnn = mean(accuracy_KNN);
mAccCentroid = mean(accuracy_CENTROID);
mAccLR = mean(accuracy_LR);

figure(1)
x = [5 10 15 20 25 30 40 50 100 200];
plot(x,mAccSvm*100,'-rs');
grid on
hold on
plot(x,mAccKnn*100,'-bs');
hold on
plot(x,mAccLR*100,'-ms');
hold on
plot(x,mAccCentroid*100,'-gs');
title_name = strcat('Prashanth Kolandaiwsami Arjunan (1001110082) - Problem B (Gene Dataset)');
title(title_name);
%legend('SVM classifier', 'KNN classifier','LR Classifier','Centroid classifier');

