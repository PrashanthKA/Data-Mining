
K = 1;
 trainData = load('trainDataXY.txt');
 trainData = trainData';
    train = trainData(:,2:end); 
    labels = trainData(:, 1);
    M = size(train, 1);

    N = size(train, 2);
    testData = load('testDataX.txt');
    testData = testData';

    T = size(testData,1);
    labelAssignment = zeros(T, 1);
    for i = 1:T
            distanceToInstances = zeros(M, 1);
            for instance = 1:M
                      
                dist = norm(testData(i,:) - train(instance,:));
                distanceToInstances(instance) = dist;
            end;

          
            [Value Index] = sort(distanceToInstances);
            NeighBorLabels = labels(Index(1:K));
            labelAssignment(i) = mode(NeighBorLabels)'
    end;
