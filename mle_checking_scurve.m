clc;
clear all;
close all;

%% Swiss roll data

n_samples=2000;
K=12;

% GENERATE SAMPLED DATA
angle = pi*(1.5*rand(1,n_samples/2)-1); height = 5*rand(1,n_samples);
MixedData = [[cos(angle), -cos(angle)]; height;[ sin(angle), 2-sin(angle)]];
MixedData = MixedData';

figure
scatter3(MixedData(:,1),MixedData(:,2),MixedData(:,3));
%% standardized data (between 0 and 1)

MixedData = zscore(MixedData);
MixedData = MixedData(randperm(size(MixedData,1)),:);
%MixedData = Xstd(1:n_samples,:);

%% Experiment

k2 = [4;8;10;12;16;20;40;80;100];
result_no_dim1 = zeros(length(k2),1);
result_no_dim2 = zeros(length(k2),1);
experiments = [k2, result_no_dim1, result_no_dim2];


%% MLE dimensionality estimation

% con promedio sobre el estimado y sobre la inversa


% Compute matrix of log nearest neighbor distances
MixedData = MixedData';
[d, n] = size(MixedData);
X2 = sum(MixedData.^2, 1); 
for i=1:numel(k2)
    k1 = k2(i)/2;
    knnmatrix = zeros(k2(i), n);
        if n < 3000
           distance = repmat(X2, n, 1) + repmat(X2', 1, n) - 2 * (MixedData' * MixedData);
           distance = sort(distance);
           knnmatrix= .5 * log(distance(2:k2(i) + 1,:));
        else
            for j=1:n
                distance = sort(repmat(X2(j), 1, n) + X2 - 2 * MixedData(:,j)' * MixedData);
                distance = sort(distance);
                knnmatrix(:,j) = .5 * log(distance(2:k2(i) + 1))'; 
            end
        end  

        % Compute the ML estimate
        S = cumsum(knnmatrix, 1);
        indexk = repmat((k1:k2(i))', 1, n);
        dhat = -(indexk - 2) ./ (S(k1:k2(i),:) - knnmatrix(k1:k2(i),:) .* indexk);
        % Average over estimates and over values of k
        no_dims1 = mean(mean(dhat));
        experiments(find(experiments(:,1)==k2(i)), 2) = no_dims1;
        % Average over the inverses
        no_dims2 = mean(dhat.^(-1));
        no_dims2 = mean(no_dims2.^(-1));
        experiments(find(experiments(:,1)==k2(i)), 3) = no_dims2;
             
end

disp(experiments)
% Plot histogram of estimates for all datapoints
%{
figure
hist(mean(dhat), 160)
xlim([0 20])
title({'MLE estimated dimension, 12000 samples',sprintf('k1=%g, k2=%g',k1,k2)})
l1 = legend(sprintf('mean dim %f',no_dims1));       
%}
