# Stability Selection class
from sklearn.linear_model import MultiTaskLasso
from sklearn.feature_selection import SelectFwe
from sklearn.feature_selection import f_regression
import pandas as pd
import numpy as np

class StabilitySelection():

    def __init__(self, X: pd.DataFrame, y: pd.DataFrame, lambda_best: float):
        """
        Initialize object with data. 

        Parameters:
        -----------
            X (pd.DataFrame):
                Dataframe containing observations as rows and predictors as columns. 

            y (pd.DataFrame)
                Dataframe containing observations as rows and targets as columns. 

            lambda_best (float):
                Best lambda determined by lasso regression cross-validation. 
        """ 
        # Assertions
        assert X.shape[0] == y.shape[0], "Must have same number of observations."

        # Attributes
        self.X = X.copy()
        self.features = self.X.columns.to_list()
        self.y = y.copy()
        self.predictors = self.y.columns.to_list()
        self.lambda_best = lambda_best

    def run(self, n_iters: int=100, frac: float=0.5, max_iter: int=100000, fwer_thresh: float=0.1, freq_thresh: float=0.6):
        """
        Run the stability selection. 

        Parameters:
        -----------
            n_iters (int):
                Number of iterations. Default is 100. 

            frac (float):
                Fraction of the data to subset. Default is 0.5.

            max_iter (int):
                Maximum number of iterations. Default is 1000000.

            freq_thresh (float):
                Frequency threshold that selection of a predictor must meet to qualify as selected. 

            fwer_thresh (float):
                Threshold to reject/accept FWER scores. 
        """

        # Get lamda_region for pertubation
        self._get_lambda_region(n_iters=n_iters)

        # Run
        self.selection_path = np.empty((n_iters, len(self.predictors), len(self.features)))
        for i in range(n_iters):
            print('iteration', i+1)

            # Step 1: Partition subset
            self._partition(frac)

            # Step 2: Perturb, Lasso, Evaluate with FWER
            self.selection_path[i] = self._lasso(reg_term=self.lambda_region[i], max_iter=max_iter, fwer_thresh=fwer_thresh)

        # Step 4: Compute frequency of selection per iteration
        self.sele_freqs = self.selection_path.sum(axis=0) / n_iters
        print(self.sele_freqs.max())
        print(self.sele_freqs[:,71])
        
        # Step 5: Select based on freq_thresh
        self.stability_bact_dict = {}
        self.stability_bact_inds_dict = {}
        for gene_ind, gene in enumerate(self.predictors):
            self.stability_bact_dict[gene] = []
            self.stability_bact_inds_dict[gene] = []
            for bact_ind, bact in enumerate(self.features):
                if self.sele_freqs[gene_ind, bact_ind] >= freq_thresh:
                    self.stability_bact_dict[gene].append(bact)
                    self.stability_bact_inds_dict[gene].append(bact_ind)

    def _partition(self, frac: float=0.5):
        """
        Partition the dataset.

        Parameters:
        -----------
            frac (float):
                Fraction of the data to subset. Default is 0.5.
        """
        # Partition
        rand_sample = np.random.randint(low=0, high=int(self.X.shape[0]-1), size=int(self.X.shape[0]*frac))
        self.X_sample = self.X.iloc[rand_sample, :]
        self.y_sample = self.y.iloc[rand_sample, :]

    def _get_lambda_region(self, n_iters: int):
        """
        Perturb the penalty term.

        Parameters:
        -----------
            n_iters (int):
                Number of iterations.
        """
        # Perturb
        self.lambda_region = (self.lambda_best*1.2 - self.lambda_best*0.8) * np.random.random(size=n_iters) + self.lambda_best*0.8

    def _lasso(self, reg_term: float, max_iter: int, fwer_thresh: float):
        """
        Run Lasso model from sklearn. 

        Parameters:
        -----------
            reg_term (float):
                Penalization term. 

            max_iter (int):
                Maximum number of iterations. 

            fwer_thresh (float):
                Threshold to reject/accept FWER scores. 
        """

        # Create model
        self.lasso = MultiTaskLasso(alpha=reg_term, max_iter=max_iter).fit(self.X_sample, self.y_sample)
        lasso_inds = [[j for j in range(len(self.lasso.coef_[i])) if self.lasso.coef_[i,j] != 0] for i in range(len(self.lasso.coef_))]

        # Evaluate 
        fwe_inds = [[] for i in range(len(self.predictors))]
        for i in range(len(self.predictors)):
            fwe = SelectFwe(score_func=f_regression, alpha=fwer_thresh).fit(self.X_sample, self.y_sample.iloc[:,i])
            fwe_inds[i] = [j for j in range(len(fwe.get_support())) if fwe.get_support()[j] == True]

        # Compare selections 
        sele_inds = np.zeros((len(self.predictors), len(self.features)))
        for gene_ind in range(len(lasso_inds)):
            for bact_ind in lasso_inds[gene_ind]:
                if bact_ind in fwe_inds[gene_ind]:
                    sele_inds[gene_ind, bact_ind] = 1
        
        return sele_inds