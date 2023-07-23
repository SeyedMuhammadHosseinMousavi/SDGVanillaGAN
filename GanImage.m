%% Synthetic Data Generation (SDG) Using Vanilla GAN

clear;
close all;
clc;
% Loading data
path='o';
imds = imageDatastore(path,IncludeSubfolders=true,LabelSource="foldernames");
% Number of classes
classes = categories(imds.Labels);
numClasses = numel(classes)
% Augment the data to include random horizontal flipping and resize the images to have size 64-by-64.
augmenter = imageDataAugmenter(RandXReflection=true);
augimds = augmentedImageDatastore([64 64],imds,DataAugmentation=augmenter);
% Define the two-input network
% Generating images given random vectors of size 100 and corresponding labels
numLatentInputs = 100;
embeddingDimension = 50;
numFilters = 64;
filterSize = 5;
projectionSize = [4 4 1024];

layersGenerator = [
featureInputLayer(numLatentInputs)
fullyConnectedLayer(prod(projectionSize))
functionLayer(@(X) feature2image(X,projectionSize),Formattable=true)
concatenationLayer(3,2,Name="cat");
transposedConv2dLayer(filterSize,4*numFilters)
batchNormalizationLayer
reluLayer
transposedConv2dLayer(filterSize,2*numFilters,Stride=2,Cropping="same")
batchNormalizationLayer
reluLayer
transposedConv2dLayer(filterSize,numFilters,Stride=2,Cropping="same")
batchNormalizationLayer
reluLayer
transposedConv2dLayer(filterSize,3,Stride=2,Cropping="same")
tanhLayer];
lgraphGenerator = layerGraph(layersGenerator);

layers = [
featureInputLayer(1)
embeddingLayer(embeddingDimension,numClasses)
fullyConnectedLayer(prod(projectionSize(1:2)))
functionLayer(@(X) feature2image(X,[projectionSize(1:2) 1]),Formattable=true,Name="emb_reshape")];

lgraphGenerator = addLayers(lgraphGenerator,layers);
lgraphGenerator = connectLayers(lgraphGenerator,"emb_reshape","cat/in2");
% To train the network with a custom training loop and enable automatic differentiation, convert the layer graph to a dlnetwork object.
netG = dlnetwork(lgraphGenerator)

%%
% Create a network that takes as input 64-by-64-by-1 images and the corresponding labels and 
% outputs a scalar prediction score using a series of convolution layers with batch 
% normalization and leaky ReLU layers. Add noise to the input images using dropout.
dropoutProb = 0.75;
numFilters = 64;
scale = 0.2;

inputSize = [64 64 3];
filterSize = 5;

layersDiscriminator = [
imageInputLayer(inputSize,Normalization="none")
dropoutLayer(dropoutProb)
concatenationLayer(3,2,Name="cat")
convolution2dLayer(filterSize,numFilters,Stride=2,Padding="same")
leakyReluLayer(scale)
convolution2dLayer(filterSize,2*numFilters,Stride=2,Padding="same")
batchNormalizationLayer
leakyReluLayer(scale)
convolution2dLayer(filterSize,4*numFilters,Stride=2,Padding="same")
batchNormalizationLayer
leakyReluLayer(scale)
convolution2dLayer(filterSize,8*numFilters,Stride=2,Padding="same")
batchNormalizationLayer
leakyReluLayer(scale)
convolution2dLayer(4,1)];

lgraphDiscriminator = layerGraph(layersDiscriminator);

layers = [
featureInputLayer(1)
embeddingLayer(embeddingDimension,numClasses)
fullyConnectedLayer(prod(inputSize(1:2)))
functionLayer(@(X) feature2image(X,[inputSize(1:2) 1]),Formattable=true,Name="emb_reshape")];

lgraphDiscriminator = addLayers(lgraphDiscriminator,layers);
lgraphDiscriminator = connectLayers(lgraphDiscriminator,"emb_reshape","cat/in2");
% To train the network with a custom training loop and enable automatic differentiation, convert the layer graph to a dlnetwork object.
netD = dlnetwork(lgraphDiscriminator)

% Define Model Loss Functions
% which takes as input the generator and discriminator networks, a mini-batch of input data, 
% and an array of random values, and returns the gradients of the loss with respect to the 
% learnable parameters in the networks and an array of generated images.
% Train with a mini-batch size of 128 for 500 epochs.
numEpochs = 500;
miniBatchSize = 32;
% Specify the options for Adam optimization. For both networks, use:
learnRate = 0.0002;
gradientDecayFactor = 0.5;
squaredGradientDecayFactor = 0.999;
% Update the training progress plots every 100 iterations.
validationFrequency = 5;
% If the discriminator learns to discriminate between real and generated images too quickly,
% then the generator can fail to train. To better balance the learning of the discriminator and
% the generator, randomly flip the labels of a proportion of the real images. Specify a flip
% factor of 0.5.
flipFactor = 0.5;

%%
% Train the model using a custom training loop. Loop over the training data and update the 
% network parameters at each iteration. To monitor the training progress, display a batch of 
% generated images using a held-out array of random values to input into the generator and the 
% network scores.
augimds.MiniBatchSize = miniBatchSize;
executionEnvironment = "auto";
mbq = minibatchqueue(augimds, ...
MiniBatchSize=miniBatchSize, ...
PartialMiniBatch="discard", ...
MiniBatchFcn=@preprocessData, ...
MiniBatchFormat=["SSCB" "BC"], ...
OutputEnvironment=executionEnvironment);   
% Initialize the parameters for the Adam optimizer.
velocityD = [];
trailingAvgG = [];
trailingAvgSqG = [];
trailingAvgD = [];
trailingAvgSqD = [];
% To monitor training progress, create a held-out batch of 25 random vectors and a corresponding
% set of labels 1 through 5 (corresponding to the classes) repeated five times.
numValidationImagesPerClass = 5;
ZValidation = randn(numLatentInputs,numValidationImagesPerClass*numClasses,"single");
TValidation = single(repmat(1:numClasses,[1 numValidationImagesPerClass]));
% Convert the data to dlarray objects and specify the dimension labels "CB" (channel, batch).
ZValidation = dlarray(ZValidation,"CB");
TValidation = dlarray(TValidation,"CB");
% For GPU training, convert the data to gpuArray objects.
if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
ZValidation = gpuArray(ZValidation);
TValidation = gpuArray(TValidation);
end
% To track the scores for the generator and discriminator, use a trainingProgressMonitor object.
% Calculate the total number of iterations for the monitor.
numObservationsTrain = numel(imds.Files);
numIterationsPerEpoch = floor(numObservationsTrain / miniBatchSize);
numIterations = numEpochs * numIterationsPerEpoch;
% Initialize the TrainingProgressMonitor object. Because the timer starts when you create the
% monitor object, make sure that you create the object close to the training loop.
monitor = trainingProgressMonitor( ...
Metrics=["GeneratorScore","DiscriminatorScore"], ...
Info=["Epoch","Iteration"], ...
XLabel="Iteration");

groupSubPlot(monitor,Score=["GeneratorScore","DiscriminatorScore"])

%% Train the conditional GAN. For each epoch, shuffle the data and loop over mini-batches of data.
epoch = 0;
iteration = 0;
% Loop over epochs.
while epoch < numEpochs && ~monitor.Stop
epoch = epoch + 1;

% Reset and shuffle data.
shuffle(mbq);

% Loop over mini-batches.
while hasdata(mbq) && ~monitor.Stop
iteration = iteration + 1;

% Read mini-batch of data.
[X,T] = next(mbq);

% Generate latent inputs for the generator network. Convert to
% dlarray and specify the dimension labels "CB" (channel, batch).
% If training on a GPU, then convert latent inputs to gpuArray.
Z = randn(numLatentInputs,miniBatchSize,"single");
Z = dlarray(Z,"CB");
if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
Z = gpuArray(Z);
end

% Evaluate the gradients of the loss with respect to the learnable
% parameters, the generator state, and the network scores using
% dlfeval and the modelLoss function.
[~,~,gradientsG,gradientsD,stateG,scoreG,scoreD] = ...
dlfeval(@modelLoss,netG,netD,X,T,Z,flipFactor);
netG.State = stateG;

% Update the discriminator network parameters.
[netD,trailingAvgD,trailingAvgSqD] = adamupdate(netD, gradientsD, ...
trailingAvgD, trailingAvgSqD, iteration, ...
learnRate, gradientDecayFactor, squaredGradientDecayFactor);

% Update the generator network parameters.
[netG,trailingAvgG,trailingAvgSqG] = ...
adamupdate(netG, gradientsG, ...
trailingAvgG, trailingAvgSqG, iteration, ...
learnRate, gradientDecayFactor, squaredGradientDecayFactor);

% Every validationFrequency iterations, display batch of generated images using the
% held-out generator input.
if mod(iteration,validationFrequency) == 0 || iteration == 1

% Generate images using the held-out generator input.
XGeneratedValidation = predict(netG,ZValidation,TValidation);

% Tile and rescale the images in the range [0 1].
I = imtile(extractdata(XGeneratedValidation), ...
GridSize=[numValidationImagesPerClass numClasses]);
I = rescale(I);

% Display the images.
image(I)
xticklabels([]);
yticklabels([]);
title("Generated Images");
end

% Update the training progress monitor.
recordMetrics(monitor,iteration, ...
GeneratorScore=scoreG, ...
DiscriminatorScore=scoreD);

updateInfo(monitor,Epoch=epoch,Iteration=iteration);
monitor.Progress = 100*iteration/numIterations;
end
end

%%
% Generate New Images
% To generate new images of a particular class, use the predict function on the generator with a dlarray object containing a batch of random vectors and an array of labels corresponding to the desired classes. Convert the data to dlarray objects and specify the dimension labels "CB" (channel, batch). For GPU prediction, convert the data to gpuArray objects. To display the images together, use the imtile function and rescale the images using the rescale function.

% Create an array of 36 vectors of random values corresponding to the first class.

numObservationsNew = 36;
idxClass = 3;
ZNew = randn(numLatentInputs,numObservationsNew,"single");
TNew = repmat(single(idxClass),[1 numObservationsNew]);
% Convert the data to dlarray objects with the dimension labels "SSCB" (spatial, spatial, channels, batch).

ZNew = dlarray(ZNew,"CB");
TNew = dlarray(TNew,"CB");
% To generate images using the GPU, also convert the data to gpuArray objects.

if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
    ZNew = gpuArray(ZNew);
    TNew = gpuArray(TNew);
end
% Generate images using the predict function with the generator network.

XGeneratedNew = predict(netG,ZNew,TNew);
% Display the generated images in a plot.

figure
I = imtile(extractdata(XGeneratedNew));
I = rescale(I);
imshow(I)
title("Class: " + classes(idxClass))
