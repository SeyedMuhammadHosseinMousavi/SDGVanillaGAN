function [lossG, lossD] = ganLoss(scoresReal,scoresGenerated)

% Calculate losses for the discriminator network.
lossGenerated = -mean(log(1 - scoresGenerated));
lossReal = -mean(log(scoresReal));

% Combine the losses for the discriminator network.
lossD = lossReal + lossGenerated;

% Calculate the loss for the generator network.
lossG = -mean(log(scoresGenerated));

end