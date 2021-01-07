"""
This is a backward selection algorithm using AIC criterion written in Python sklearn style.
"""
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
import statsmodels.api as sm

class BackwardSelectionOLS(BaseEstimator):
    
    def fit_subset(self, X, y, feature_index):
        """Fit a linear regression model to a subset of data
        
        Parameters
        ----------
        X: dataframe, shape (n_samples, n_features)
           The input features.
        y: array, shape (n_samples,)
           The target values.
        feature_index: list
                       The names of features being used in model.
        
        Returns
        -------
        model results: dictionary
                       Returns a summary of model.
        """
        # extract features from X and add an intercept column
        X_use = X[feature_index]
        X_use.insert(0, 'const', 1)
        
        # fit linear regression on selected features and intercept
        OLSmodel = sm.OLS(y, X_use).fit()
        rsq_adj = OLSmodel.rsquared_adj
        AIC = OLSmodel.aic
        rsq = OLSmodel.rsquared
        
        return {"model":OLSmodel, "rsq_adj":rsq_adj, "rsq":rsq, "AIC":AIC, "feature_index":feature_index}
    
    
    def Backward(self, X, y, predictor_index):
        """A backward step by selecting the best model using AIC when reducing a feature
        
        Parameters
        ----------
        X: dataframe, shape (n_samples, n_features)
           The input features.
        y: array, shape (n_samples,)
           The target values.
        predictor_index: list
                         The names of features being used in model.
        
        Returns
        -------
        best_model: array
                    Returns a summary of model.
        """
        results = []
        for i in predictor_index:
            index_temp = predictor_index.copy()
            index_temp.remove(i)
            
            predictor_index_new = index_temp
            results.append(self.fit_subset(X, y, predictor_index_new))
        models = pd.DataFrame(results)
        
        # Choose the model with the lowest AIC value
        best_model = models.loc[models['AIC'].idxmin()]
        return best_model
    
    
    def BackwardSelection(self, X_data, y_data):
        """The backward Selection process
        
        Parameters
        ----------
        X_data: dataframe, shape (n_samples, n_features)
                The input features.
        y_data: array, shape (n_samples,)
                The target values.
        
        Returns
        -------
        best_model_bwd: object
                        Returns the best model after backward selection.
        predictor_index: list
                         Returns the selected features in the best model.
        """
        count = X_data.shape[1]
        
        # initialize the predictor list and the best model
        predictor_index = list(X_data.columns)
        best_model_dict = self.fit_subset(X_data, y_data, predictor_index)
        print(count, predictor_index)
        
        count = count - 1
        while count >= 0:
            model_dict_temp = self.Backward(X_data, y_data, predictor_index)
            if model_dict_temp['AIC'] <= best_model_dict['AIC']:
                best_model_dict = model_dict_temp
                predictor_index = best_model_dict['feature_index']
                print(count, predictor_index)
                count = count - 1
            else:
                break
        
        best_model_bwd = best_model_dict['model']
        return best_model_bwd, predictor_index
    
    
    def fit(self, X, y):
        """The fitting function
        
        Parameters
        ----------
        X: dataframe, shape (n_samples, n_features)
           The input features.
        y: array, shape (n_samples,)
           The target values.
        
        Returns
        -------
        self: object
              Returns self.
        """
        self.best_model_selected, self.best_predictors = self.BackwardSelection(X, y)
        
        self.is_fitted_ = True
        return self
    
    
    def predict(self, X):
        """The predicting function
        
        Parameters
        ----------
        X: dataframe, shape (n_samples, n_features)
           The input features.
        
        Returns
        -------
        self: array
              Returns predicted target values.
        """
        # extract the selected features from X and add an intercept column
        X_use_best = X[self.best_predictors]
        X_use_best.insert(0, 'const', 1)
        
        y_pred = self.best_model_selected.predict(X_use_best)
        
        check_is_fitted(self, 'is_fitted_')
        return y_pred
    


# Creating Simulated Data
from sklearn import datasets
features, target, coef = datasets.make_regression(n_samples = 100, n_features = 4, 
                                                  n_informative = 2, n_targets = 1, 
                                                  noise = 5, coef = True, random_state=0)

X = pd.DataFrame(features, columns=['Feature_1', 'Feature_2', 'Feature_3', 'Feature_4'])
y = pd.DataFrame(target, columns=['Target'])

# Applying the Backward Selection to simulated data
bwd = BackwardSelectionOLS()
bwd.fit(X, y)
y_pred = bwd.predict(X)

