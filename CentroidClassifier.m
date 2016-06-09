function Centroid_Classifier_
Result = CentroidClassifier(trainData, testData)
xtrain = csvread(trainData);
xtest = csvread(testData);

% pick class label
classLabel = xtrain(1,:);

%delete first row in xtrain beacause it contains class labels
xtrain(1,:) = [];
[xtrain_row, ~] = size(xtrain);

[number_for_class, unique_classLabel] = hist(classLabel, unique(classLabel));
count_label = length(unique_classLabel);

class_mean_vector = zeros(xtrain_row, count_label);

start_pos = 1;
end_pos = 0;

% calculating centroid
for k = 1:count_label
    class = unique_classLabel(k);
    number = number_for_class(k);
    end_pos = class * number;
    single_type_class = xtrain(:,start_pos:end_pos);
    class_mean_vector(:,k) = mean(single_type_class,2);
    start_pos = end_pos + 1;
end

% size of test data
[~, test_col] = size(xtest);

Centroid_Classifier_Result = zeros(1,test_col);

for i = 1:test_col
    test_vector = xtest(:,i);
    test_vector = repmat(test_vector, 1, size(class_mean_vector, 2));
    dist = sqrt(sum((test_vector - class_mean_vector).^2));
    %dist = pdist2(test_vector', class_mean_vector');
    [~, pos] = min(dist);
    Centroid_Classifier_Result(:,i) = pos;
end
%csvwrite('results.txt',Centroid_Classifier_Result)
tmp = csvread('testDataXY.txt');
actual_class_label = tmp(1,:)
fileID = fopen('results.txt', 'a');
fprintf(fileID, '\n\nResult of Centroid Classifier for ATNT50\n\n');
fprintf(fileID, '%d,', Centroid_Classifier_Result);
fclose(fileID);



    
