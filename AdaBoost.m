%% Hyper-parameters

% Number of randomized Haar-features
nbrHaarFeatures = 100;
% Number of training images, will be evenly split between faces and
% non-faces. (Should be even.)
nbrTrainImages = 500;
% Number of weak classifiers
nbrWeakClassifiers = 30;

%% Load face and non-face data and plot a few examples
load faces;
load nonfaces;
faces = double(faces(:,:,randperm(size(faces,3))));
nonfaces = double(nonfaces(:,:,randperm(size(nonfaces,3))));

figure(1);
colormap gray;
for k=1:25
    subplot(5,5,k), imagesc(faces(:,:,10*k));
    axis image;
    axis off;
end

figure(2);
colormap gray;
for k=1:25
    subplot(5,5,k), imagesc(nonfaces(:,:,10*k));
    axis image;
    axis off;
end

%% Generate Haar feature masks
haarFeatureMasks = GenerateHaarFeatureMasks(nbrHaarFeatures);

figure(3);
colormap gray;
for k = 1:25
    subplot(5,5,k),imagesc(haarFeatureMasks(:,:,k),[-1 2]);
    axis image;
    axis off;
end

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

train_samples = length(xTrain(1,:));
d = ones(train_samples,1)/train_samples; %initialization of weigths
parameters = zeros(nbrWeakClassifiers,4);

for i = 1:nbrWeakClassifiers                    %Loop through all classifiers
    min_error = inf;
    C_opti = zeros(nbrTrainImages,1);           %The best classification 
    polarity_opti = 0;                          %The best polarity value
    Haar_opti = 0;                              %The best Haar feature
    threshold_opti = 0;
    
    for j = 1:nbrHaarFeatures                   %Loop through all Haarfeatures
        for k = 1:nbrTrainImages                %Loop through all traingin data
            
            polarity = 1;
            C = WeakClassifier(xTrain(j,k),polarity,xTrain(j,:));
            E = WeakClassifierError(C,d,yTrain);
            
            if E > 1
                    disp(E);
            end
                
            %Reverse the polarity if error greater than .5
            if E > 0.5
                polarity = -polarity;
                E = 1-E;    
            end
            
            %Check if minimum error
            if E < min_error
                min_error = E;
                
                %Save the optimal values
                C_opti = polarity*C;
                polarity_opti = polarity;
                Haar_opti = j;
                threshold_opti = xTrain(j,k);
            end
        end
    end
    
    alpha = 0.5*log((1-min_error)/min_error);
    d = d.*exp(-alpha*yTrain'.*C); %Update weights
    d = d/sum(d);               %Normalize
    
    %Save parameters:
    parameters(i,1) = threshold_opti;
    parameters(i,2) = polarity_opti;
    parameters(i,3) = Haar_opti;
    parameters(i,4) = alpha;
end

%% Evaluate your strong classifier here
%  Evaluate on both the training data and test data, but only the test
%  accuracy can be used as a performance metric since the training accuracy
%  is biased.

acc = zeros(nbrWeakClassifiers,1);
for i = 1:nbrWeakClassifiers
    test_classification = parameters(i,4)*WeakClassifier(parameters(i,1),parameters(i,2),xTest(parameters(i,3),:));
    
    acc(i) = 1-sum(sign(test_classification) ~= yTest')/length(yTest);

end

Classifications = sign(sum(test_classification));


%% Plot the error of the strong classifier as a function of the number of weak classifiers.
%  Note: you can find this error without re-training with a different
%  number of weak classifiers.

figure(4);
plot(1:nbrWeakClassifiers,acc)

%% Plot some of the misclassified faces and non-faces
%  Use the subplot command to make nice figures with multiple images.

figure(5);
colormap gray;
for k=1:25
    subplot(5,5,k), imagesc(faces(:,:,10*k));
    axis image;
    axis off;
end

%% Plot your choosen Haar-features
%  Use the subplot command to make nice figures with multiple images.

figure(6);
