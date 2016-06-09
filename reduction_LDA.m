function matrix = reduction_LDA(data, label, num)
    u = unique(label);
    data_mean = mean(data);
    SB = zeros(size(data,2), size(data,2));
    SW = zeros(size(data,2), size(data,2));
    for i = 1:numel(u)
        tLabel = (label==u(i));
        tData = data(tLabel,:);
        class_mean = mean(tData);
        for j = 1:size(tData,1)
            XC(j,:) = tData(j,:) - class_mean;
        end
        SW = SW + cov(XC)*numel(tLabel)/size(data,1);
    end
    for i = 1:size(data,1)
        X(i,:) = data(i,:) - data_mean;
    end
    SB = cov(X) - SW;
    [vecB, valB] = eig(SB);
    D = diag(valB);
    [~, id] = sort(D, 'descend');
    vecB = vecB(:,id);
    H = vecB*SW;
    [f_vec, f_val] = eig(H*H');
    matrix = f_vec(:, 1:num);
%     u = unique(label);
%     mean_data = mean(data);
%     for i = 1:numel(u)
%         tLabel = (label==u(i));
%         classmean_data(i,:) = mean(data(tLabel,:));
%     end
%     for i = 1:numel(u)
%         XB(i,:) = classmean_data(i,:) - mean_data;
%     end
%     for i = 1:size(data,2)
%         XW(:,i) = data(:,i)-classmean_data(:, floor((i-1)/10+1));
%     end
%     
%     [VecB, valB] = eig(XB'*XB);
%     D = diag(valB);
%     [temp, sorted_ind] = sort(D,'descend');
%     VecB = VecB(:, sorted_ind);
%     D = D(sorted_ind);
%     V = XB*VecB;
%     
%     Y = V(:, num);
%     DB = Y'*XB*XB'*Y;
%     Z = Y*DB^(-0.5);
%     H = Z'*XW;
%     [f_vec, f_values] = eig(H*H');
%     vec = f_vec(1:num);
%     matrix = Z*vec;
    
    
   
end