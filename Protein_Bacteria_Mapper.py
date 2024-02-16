import numpy as np
import pandas as pd
import os, sys
from sklearn.linear_model import MultiTaskLassoCV
from sklearn.feature_selection import f_regression, SelectFdr
from IPython.display import display

class Protein_Bacteria_Mapper():

    def __init__(self, gene_exp: pd.DataFrame, bact_exp: pd.DataFrame, patients: pd.DataFrame, health_profiles: list[str], genes: list[str]=None):

        # Assertions
        assert gene_exp.shape[0] == bact_exp.shape[0]
        assert gene_exp.shape[0] == patients.shape[0]

        # Assign attributes
        self.gene_exp = gene_exp.copy()
        self.n_genes = len(self.gene_exp.columns.to_list())
        self.bact_exp = bact_exp.copy()
        self.n_bact = len(self.bact_exp.columns.to_list())
        self.patients = patients.copy()
        self.health_profiles = [h for h in health_profiles]

        # Report breakdown of patient data
        print('Patients\n---------')
        print(f'Number of patients:', self.patients.shape[0])
        self.prof_patients = []
        for prof in health_profiles:
            prof_patients_df = self.patients.loc[self.patients['Diagnosis'] == prof].index.to_list()
            self.prof_patients.append(prof_patients_df)
            print(f'Number of {prof} patients:', len(prof_patients_df))
    
        # Report breakdown of bacteria 
        print('\nBacteria\n---------')
        self.domains = list(np.unique([obs[0] for obs in self.bact_exp.columns if str(obs[0])!= 'nan']))
        print('Number of Domains:', len(self.domains))
        self.phylums = list(np.unique([obs[1] for obs in self.bact_exp.columns if str(obs[0])!= 'nan']))
        print('Number of Phylum:', len(self.phylums))
        self.classes = list(np.unique([obs[2] for obs in self.bact_exp.columns if str(obs[0])!= 'nan']))
        print('Number of Classes:', len(self.classes))
        self.orders = list(np.unique([obs[3] for obs in self.bact_exp.columns if str(obs[0])!= 'nan']))
        print('Number of Orders:', len(self.orders))
        self.families = list(np.unique([obs[4] for obs in self.bact_exp.columns if str(obs[0])!= 'nan']))
        print('Number of Families:', len(self.families))
        self.genuses = list(np.unique([obs[5] for obs in self.bact_exp.columns if str(obs[0])!= 'nan']))
        print('Number of Genuses:', len(self.genuses))
        try:
            self.bact_exp = self.bact_exp.droplevel(level=['Species', 'Strain'])
        except:
            pass

        inds_to_keep = []
        if genes != None:
            self.genes = [g for g in genes]
        else:
            self.genes = [g for g in self.gene_exp.columns.to_list() if self.gene_exp[g].to_numpy().sum() > np.percentile(self.gene_exp.sum(axis=1).to_numpy(), 75)]
        for gene in self.genes:
            inds_to_keep.append(self.gene_exp.columns.to_list().index(gene))
        self.gene_exp = self.gene_exp.iloc[:, inds_to_keep]
        print('Number of Genes:', self.gene_exp.shape[1])

    def run(self, fdr_tresh: float=0.1, lasso_max_iters: int=100000):

        # Prepare features
        self.X, self.y = self._prepare_Xy()
        self.features = self.X.columns.to_list()
        self.predictors = self.y.columns.to_list()

        # Lasso regression
        self._lasso_reg(lasso_max_iters)

        # FDR test
        self._check_FDR(fdr_tresh)

        # Process
        self.validated_bact_inds_dict = {}
        self.validated_bact_dict = {}
        self.lasso_bact_dict = {}
        self.lasso_bact_inds_dict = {}
        self.fdr_bact_dict = {}
        self.fdr_bact_inds_dict = {}
        for i, (gene, lasso_bacts, fdr_bacts) in enumerate(zip(self.predictors, self.lasso_bact_inds, self.fdr_bact_inds)):
            
            self.validated_bact_inds_dict[gene] = []
            self.validated_bact_dict[gene] = []
            self.lasso_bact_dict[gene] = list(np.array(self.features)[lasso_bacts])
            self.lasso_bact_inds_dict[gene] = lasso_bacts
            self.fdr_bact_dict[gene] = list(np.array(self.features)[fdr_bacts])
            self.fdr_bact_inds_dict[gene] = fdr_bacts

            for b in lasso_bacts:
                if b in fdr_bacts:
                    self.validated_bact_inds_dict[gene].append(b)
                    self.validated_bact_dict[gene].append(self.features[b])
                        

    def _lasso_reg(self, max_iters):

        print('Fitting with MultiTaskLassoCV...')
        lasso = MultiTaskLassoCV(max_iter=max_iters, cv=self.X.shape[0], n_jobs=-1).fit(self.X, self.y)
        print('alpha found', lasso.alpha_)
        self.alpha = lasso.alpha_
        self.coef = lasso.coef_.copy()

        self.lasso_bact_inds = [[] for i in range(len(self.predictors))]
        self.lasso_bact = [[] for i in range(len(self.predictors))]
        # Iterate through genes
        for i, coefs in enumerate(self.coef):
            # Iterate through features
            for j, coef in enumerate(coefs):
                if coef != 0:
                    self.lasso_bact_inds[i].append(j)
                    self.lasso_bact[i].append(self.features[j])

    def _prepare_Xy(self):
        X = self.bact_exp.copy()
        X = pd.concat((X, pd.get_dummies(self.patients['Gender']).astype(int)), axis=1)
        X = pd.concat((X, pd.get_dummies(self.patients['Diagnosis']).astype(int)), axis=1)
        X.columns = X.columns.astype(str)
        y = self.gene_exp.copy()

        return [X, y]
        
    def _check_FDR(self, threshold):

        print('Computing FDR')
        self.fdr_bact = [[] for i in range(len(self.predictors))]
        self.fdr_bact_inds = [[] for i in range(len(self.predictors))]
        for i in range(len(self.predictors)):
            sele = SelectFdr(f_regression, alpha=threshold).fit(self.X, self.y.iloc[:,i])
            self.fdr_bact[i] = sele.get_feature_names_out() 
            if len(self.fdr_bact[i]) > 0:
                for bact in self.fdr_bact[i]:
                    self.fdr_bact_inds[i].append(self.features.index(bact))
            
