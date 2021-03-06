%% Hyper-parameters
% Number of randomized Haar-features
nbrHaarFeatures = 250;
% Number of training images, will be evenly split between faces and
% non-faces. (Should be even.)
nbrTrainImages = 500;
% Number of weak classifiers
nbrWeakClassifiers = 70;

%% Load face and non-face data and plot a few examples
load faces;
load nonfaces;
faces = double(faces(:,:,randperm(size(faces,3))));
nonfaces = double(nonfaces(:,:,randperm(size(nonfaces,3))));

% figure(1);
% colormap gray;
% for k=1:25
%     subplot(5,5,k), imagesc(faces(:,:,10*k));
%     axis image;
%     axis off;
% end

% figure(2);
% colormap gray;
% for k=1:25
%     subplot(5,5,k), imagesc(nonfaces(:,:,10*k));
%     axis image;
%     axis off;
% end

%% Generate Haar feature masks
haarFeatureMasks = GenerateHaarFeatureMasks(nbrHaarFeatures);

% figure(3);
% colormap gray;
% for k = 1:25
%     subplot(5,5,k),imagesc(haarFeatureMasks(:,:,k),[-1 2]);
%     axis image;
%     axis off;
% end

%% Create image sets (do not modify!)

% Create a training data set with examples from both classes.
% Non-faces = class label y=-1, faces = class label y=1
trainImages = cat(3,faces(:,:,1:nbrTrainImages/2),nonfaces(:,:,1:nbrTrainImages/2));
xTrain = ExtractHaarFeatures(trainImages,haarFeatureMasks);
yTrain = [ones(1,nbrTrainImages/2), -ones(1,nbrTrainImages/2)];

% Create a test data set, using the rest of the faces and non-faces.
testImages  = cat(3,faces(:,:,(nbrTrainImages/2+1):end),...
                    nonfaces(:,:,(nbrTrainImages/2+1):end));
xTest = ExtractHaarFeatures(testImages,haarFeatureMasks);
yTest = [ones(1,size(faces,3)-nbrTrainImages/2), -ones(1,size(nonfaces,3)-nbrTrainImages/2)];

% Variable for the number of test-data.
nbrTestImages = length(yTest);

%% Implement the AdaBoost training here
%  Use your implementation of WeakClassifier and WeakClassifierError
tic

d = ones(nbrTrainImages,1)/nbrTrainImages; %initialization of weigths
parameters = zeros(nbrWeakClassifiers,4);

for i = 1:nbrWeakClassifiers                    %Loop through all classifiers
    min_error = inf;
    
    C_opti = zeros(nbrTrainImages,1);           %The best classification 
    polarity_opti = 0;                          %The best polarity value
    Haar_opti = 0;                              %The best Haar feature
    threshold_opti = 0;

    for haar = 1:nbrHaarFeatures                   %Loop through all Haarfeatures
        for train_img = 1:nbrTrainImages                %Loop through all traingin data
            
            polarity = 1;
            threshold = xTrain(haar,train_img);
            C = WeakClassifier(threshold,polarity,xTrain(haar,:));
            E = WeakClassifierError(C,d,yTrain);
            
            %Reverse the polarity if error greater than .5
            if E > 0.5
                polarity = -1;
                E = 1-E;    
            end
            
            %Check if minimum error
            if E < min_error
                min_error = E;
                
                %Save the optimal values
                C_opti = polarity*C;
                polarity_opti = polarity;
                Haar_opti = haar;
                threshold_opti = threshold;
            end
        end
    end
    
    alpha = 0.5*log((1-min_error)/(min_error));
    d = d.*exp(-alpha*yTrain'.*C_opti); %Update weights
    d = d/sum(d);                   %Normalize

    %Save parameters:
    parameters(i,1) = alpha;
    parameters(i,2) = polarity_opti;
    parameters(i,3) = Haar_opti;
    parameters(i,4) = threshold_opti;
end

toc
%% Evaluate your strong classifier here
%  Evaluate on both the training data and test data, but only the test
%  accuracy can be used as a performance metric since the training accuracy
%  is biased.

test_classification_val = zeros(nbrWeakClassifiers,nbrTestImages);
acc_test = zeros(nbrWeakClassifiers,1);
for i = 1:nbrWeakClassifiers
    test_classification_val(i,:) = parameters(i,1).*WeakClassifier(parameters(i,4),parameters(i,2),xTest(parameters(i,3),:));
    test_classification = sign(sum(test_classification_val(1:i,:)));
    acc_test(i) = 1-sum(test_classification ~= yTest)/length(yTest);
end

strong_classifier = sign(sum(test_classification_val(1:66,:)));
res = strong_classifier ~= yTest;
acc = 1-sum(strong_classifier ~= yTest)/length(yTest)

train_classification_val = zeros(nbrWeakClassifiers,nbrTrainImages);
acc_train = zeros(nbrWeakClassifiers,1);
for i = 1:nbrWeakClassifiers
    train_classification_val(i,:) = parameters(i,1).*WeakClassifier(parameters(i,4),parameters(i,2),xTrain(parameters(i,3),:));
    train_classification = sign(sum(train_classification_val(1:i,:)));
    acc_train(i) = 1-sum(train_classification ~= yTrain)/length(yTrain);
end

%% Plot the error of the strong classifier as a function of the number of weak classifiers.
%  Note: you can find this error without re-training with a different
%  number of weak classifiers.

figure(4);

plot(1:nbrWeakClassifiers,acc_test)
hold on
plot(1:nbrWeakClassifiers,acc_train)
hold on
yline(0.93,'-.b')
title('Combine Plots')
ylabel('accuracy')
xlabel('nbrWeakClassifiers')
legend({'Test','Train', '93%'},'Location','southeast')
%% Plot some of the misclassified faces and non-faces
%  Use the subplot command to make nice figures with multiple images.

missclass = find(res);

figure(5);
colormap gray;
i = 1;
N = length(missclass);
for k = N:-1:N-24
    subplot(5,5,i), imagesc(testImages(:,:,missclass(k)));
    axis image;
    axis off;
    i = i+1;
end


figure(6);
colormap gray;
for k = 1:25
    subplot(5,5,k), imagesc(testImages(:,:,missclass(k)));
    axis image;
    axis off;
end

%% Plot your choosen Haar-features
%  Use the subplot command to make nice figures with multiple images.

figure(7);
colormap gray;
for k = 1:25
    subplot(5,5,k),imagesc(haarFeatureMasks(:,:,parameters(k,3)),[-1 2]);
    axis image;
    axis off;
end
