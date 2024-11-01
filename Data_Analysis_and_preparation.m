%% Importing all the data sets and data treatment for the PCA analysis. 
PGSData = readtable('Definitions_PGS_results.tsv', 'FileType','text','Delimiter','\t');
PGSRes = readtable('PGS_results.tsv', 'FileType','text','Delimiter','\t' );
Pheno = readtable('Phenotype_essential.tsv', 'FileType','text', 'Delimiter','\t');
% Only keep the numeric data to the PCA analysis. 
PGSPheno = Pheno(:,"ZIL_TOT_T3");
PGSResJust = PGSRes(:,3:end);

%% Data distribution analysis.PCA without normalized data. 
a = table2array(PGSResJust); % Convert into an array. 
[pcs,scrs,~,~,pctEXP] = pca(a); % Performing PCA.
figure;
pareto(pctEXP);
figure;
scatter(scrs(:,1),scrs(:,2));
xlabel('Principal Com 1');
ylabel('Principal Com 2');

%% Data distribution analysis.PCA with normalized data.  
DataN = zscore(a); % Normalization
[pcs,scrs,~,~,pctEXP] = pca(DataN);
figure;
pareto(pctEXP);
figure;
scatter(scrs(:,1),scrs(:,2));
xlabel('Principal Com 1');
ylabel('Principal Com 2');

%% Data treatment.
% This for loop has the intention to match and reorder 
% each sample between the two datasets. In Order to have 
% coherence through the samples for the Machine learning algorithms. 
PGSo= table();
Delete = [];

for i=1:numel(Pheno.SampleID);
    
    idx= find(ismember(PGSRes.SampleID, Pheno.SampleID{i}));
    if isempty(idx)==1
        Delete = [Delete; i];
    else
        PGSo= [PGSo; PGSRes(idx,:)];
    end
end 
Pheno(Delete,:)=[];
PGSo(412,:)=[];

 Pheno.ZIL_TOT_T3=str2double(Pheno.ZIL_TOT_T3);

% This task has the goal of deleting el missing data (NaN) 
% in the phenotype data set.
% Fill missing data
[PhenoCom,missingIndices] = fillmissing(Pheno(:,vartype("numeric")),"nearest");

% Display results
figure

% Plot cleaned data
plot(PhenoCom.Traject,"Color",[0 114 189]/255,"LineWidth",1.5,...
    "DisplayName","Cleaned data")
hold on

% Plot filled missing entries
plot(find(missingIndices(:,1)),PhenoCom.Traject(missingIndices(:,1)),".",...
    "MarkerSize",12,"Color",[217 83 25]/255,...
    "DisplayName","Filled missing entries")
title("Number of filled missing entries: " + nnz(missingIndices(:,1)))

hold off
legend
ylabel("Traject")
clear missingIndices

%% Prepare the data for the models. 
% The machine learning functions only 
% work with number arrays.  
PGSo2 = PGSo(:,3:end);
PGSML = table2array(PGSo2);
PhenoML = table2array(PhenoCom);
PhenoZIPml = PhenoML(:,2);
PhenoHeiML = PhenoML(:,3);