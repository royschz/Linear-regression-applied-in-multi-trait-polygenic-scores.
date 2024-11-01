%% Second regression model. 
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

%% The second regression model applied with the PGS and the Phenotype.Height. 
% First model PGS data with the height data from the 
% phenotype data set.
mdl1 = fitlm(PGSMLtrain,PhenoHeiMLtrain);
Predictions2 = predict(mdl1,PGSMLtest);
RMSE = mean((PhenoHeiMLtest - Predictions2).^2)
figure
plot(PGSMLtrain,PhenoHeiMLtrain,'.');
title('PGS tran vs Height','FontSize',16)
xlabel('PGS train data')
ylabel('Height test data')
legend("PSGM Data","Height Predictions")
hold on;
figure
plot(PGSMLtest,'.');
hold on
plot(Predictions2,'.');
hold off
title('PGS vs the Height Predictions', 'FontSize',16)
legend("PSGM Data","Height Predictions")

% Make the predictions with the model and the PGS trained data.   
PGSMLpredictions = predict(mdl1, PGSMLtrain)
% Calculating the overfitting and measuring the performance of the model. 
Overfitting = mean((PhenoHeiMLtrain - PGSMLpredictions).^2) 
MeanSquaredError = RMSE

%% Fitting Ridge Regression Model.
% Despite the good results, I tried to apply a Ridge Regression. 
lambda = 0.1
Ri = ridge(PhenoHeiMLtrain,PGSMLtrain,lambda,0)
plot(lambda,Ri)
RiPre = Ri(1,:) + PGSMLtest*Ri(2:end,:)
RiMSE = mean((PhenoHeiMLtest - RiPre).^2)
plot(lambda,RiMSE)
title('Ridge Regression Model')
[minMSE idx] = min(RiMSE)
RiMin = Ri(:,idx)
plot(PhenoZILtest,"o")
hold on 
plot(RiPre(:,idx),".")
title('Mean square error values of the Ridge Regression')
xlabel('Lambda')
ylabel('Predictions')
legend('PhenoHeight','Ridge MSE')
hold off
MeanSquaredErrorRidge = RiMin(1,1)

%% Fitting with Lasso.
% Despite the good results, I tried to apply a Lasso Regression. 
lambda = (0:100)/length(PhenoHeiMLtrain);
[Las,fitinfo] = lasso(PGSMLtrain,PhenoHeiMLtrain,"Lambda",lambda,"Alpha",0.4);
LasPred = fitinfo.Intercept + PGSMLtest*Las
LasMSE = mean((LasPred - PhenoZILtest).^2)
plot(lambda,LasMSE)
title('Mean Lasso Regression')
xlabel('Lambda')
ylabel('LasMSE')
[minLasMSE,idx] = min(LasMSE)

%% Gaussian Process Regression. 
% In order to analyze the data with other types 
% of models, I applied this approach compared to 
% the Linear Regression model. 
Gaumdl = fitrgp(PGSMLtrain,PhenoHeiMLtrain,"KernelFunction","squaredexponential")
Pred = predict(Gaumdl,PGSMLtrain)
Pre = predict(Gaumdl, PGSMLtrain);  
mdlMSE = mean((PhenoHeiMLtrain - Pre).^2)
plot(PGSMLtrain,'.');
hold on
plot(Pred,'.');
hold off
title('Gaussian Process Regression');
legend("PSGM Data","Predictions");
