% load the dataset
d = load('data.mat');


trainX  = d.trainX;
testX = d.testX;
trainY = d.trainY;
testY = d.testY;
%plot one training image
imshow(trainX(:,:,:,1));
% define the cnn
layers = [
    imageInputLayer([28 28 1])
	
    convolution2dLayer(3,16,'Padding',1)
    batchNormalizationLayer
    reluLayer
	
    maxPooling2dLayer(2,'Stride',2)
	
    convolution2dLayer(3,32,'Padding',1)
    batchNormalizationLayer
    reluLayer
	
    maxPooling2dLayer(2,'Stride',2)
	
    convolution2dLayer(3,64,'Padding',1)
    batchNormalizationLayer
    reluLayer
	
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];

% define the training miniBatch size (batch gradient descent), and other
% training options
miniBatchSize = 8192;
options = trainingOptions( 'sgdm',...
    'MiniBatchSize', miniBatchSize,...
    'Plots', 'training-progress');


%train the network
net = trainNetwork(trainX, categorical(trainY), layers, options);	

% save all the variables
save net;



