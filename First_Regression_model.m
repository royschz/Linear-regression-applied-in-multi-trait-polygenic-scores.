%% First regression model. 
%% Split each dataset in 80% for the training and 20% for the test. 
% Split the data PGSML
cvp = cvpartition(837, 'HoldOut', 0.2);
rng('default');
idxTrain = training(cvp);
PGSMLtrain = PGSML(idxTrain,:);
idxNew = test(cvp);
PGSMLtest = PGSML(idxNew,:);

% Split the data PhenoZIL 
cvp = cvpartition(837, 'HoldOut', 0.2);
rng('default');
idxTrain = training(cvp);
PhenoZILtrain = PhenoZIPml(idxTrain,:);
idxNew = test(cvp);
PhenoZILtest = PhenoZIPml(idxNew,:);

%Split the data Pheno Heigth
cvp = cvpartition(837, 'HoldOut', 0.2);
rng('default');
idxTrain = training(cvp);
PhenoHeiMLtrain = PhenoHeiML(idxTrain,:);
idxNew = test(cvp);
PhenoHeiMLtest = PhenoHeiML(idxNew,:);

%% The first regression model applied with the PGS and the Phenotype.ZIL data. 
% Training and testing the data with the “fitlm” function.
mdl1 = fitlm(PGSMLtrain,PhenoZILtrain);
Predictions1 = predict(mdl1,PGSMLtest);
RMSE = mean((PhenoZILtest - Predictions1).^2); % Calculate the mean squared error. 
% Plotting the real data with the predictions. 
figure;
plot(PGSMLtrain,PhenoZILtrain,'.')
title('PGS tran vs Phenotypetest','FontSize',16);
xlabel('PGS train data');
ylabel('Phenotype test data');
legend("PSGM Data","Predictions");
plot(PGSMLtest,'.');
hold on
plot(Predictions1,'.');
hold off
title('PGS vs the Predictions', 'FontSize',16);
legend("PSGM Data","Predictions");
% Make the predictions with the model and the PGS trained data.  
PGSMLpredictions = predict(mdl1, PGSMLtrain);
% Calculating the overfitting and measuring the performance of the model. 
Overfitting = mean((PhenoZILtrain - PGSMLpredictions).^2) 
MeanSquaredError = RMSE

%% Fitting Ridge Regression Model.
% For the pre-processing data I have to apply this 
% model in order to find the lowest Mean Squared Error. 
lambda = 0:100;
Ri = ridge(PhenoZILtrain,PGSMLtrain,lambda,0);
plot(lambda,Ri);
RiPre = Ri(1,:) + PGSMLtest*Ri(2:end,:);
RiMSE = mean((PhenoZILtest - RiPre).^2);
plot(lambda,RiMSE);
title('Ridge Regression Model')
[minMSE idx] = min(RiMSE);
RiMin = Ri(:,idx);
plot(PhenoZILtest,"o");
hold on;
plot(RiPre(:,idx),".")
title('Mean square error values of the Ridge Regression')
xlabel('Lambda')
ylabel('Predictions')
legend('PhenoZILtest','Ridge MSE')
hold off
MeanSquaredErrorRidge = RiMin(1,1)

%% Fitting with Lasso.
% Due to the overfitting value, I can apply the
% Lasso model to reduce the value.  
lambda = (0:100)/length(PhenoZILtrain)
[Las,fitinfo] = lasso(PGSMLtrain,PhenoZILtrain,"Lambda",lambda,"Alpha",0.4)
LasPred = fitinfo.Intercept + PGSMLtest*Las
LasMSE = mean((LasPred - PhenoZILtest).^2)
plot(lambda,LasMSE)
title('Mean Lasso Regression')
xlabel('Lambda')
ylabel('LasMSE')
[minLasMSE,idx] = min(LasMSE)

%% Gaussian Process Regression. 
% In order to analyze the data with other types 
% of models, I applied this approach compared 
% to the Linear Regression model. 
Gaumdl = fitrgp(PGSMLtrain,PhenoZILtrain,"KernelFunction","squaredexponential")
Pre = predict(Gaumdl, PGSMLtrain);  
mdlMSE = mean((PhenoHeiMLtrain - Pre).^2)
plot(PGSMLtrain,'.');
hold on
plot(Pre,'.');
hold off
title('Gaussian Process Regression');
legend("PSGM Data","Predictions");