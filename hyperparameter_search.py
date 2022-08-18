"""
Author: Tobias Butler
Last edited: 08/05/2022
Purpose: This module contains methods that can be used to search for a set of hyperparameters that maximizes the likelihood of some user-customized cross validation function. One of 
these methods searches along each hyperparameter independently using a type of binary search. The other method performs a bayesian style search and must be provided with previous 
hyperparameter sets and associated likelihood evaluations (as a csv file). In order to utilize the first search method, a user must call this module's initial_search() method and provide 
it with a custom cross validation function and a csv file to store hyperparameter sets and their corresponding evaluations. To use the bayesian style search, a user must call the fit()
method and provide it with several arguments, including a custom cross validation function and a csv file containing previous hyperparameter sets and evaluations. For both of these 
methods, the custom cross validation function must take as an argument a dict of hyperparameter:value pairs and return a log-likelihood value that represents the effectiveness of the 
hyperparameters. That is the only specification for the custom cross validation function, meaning that the user can create a function and search for the optimum hyperparameters of any 
machine learning model.

HOW THE BAYESIAN-STYLE SEARCH WORKS:
When the fit() method is called, the provided csv file is converted to a pandas DataFrame and then to a numpy array. A multivariate normal distribution is fit to the hyperparameter sets 
(through m.l.e.) using their associated log-likelihood values as prior probabilities. At the beginning of the search, the provided \"initial_dirichlet_process_prior\" is used to adjust 
these prior probabilities, bringing them closer together or pushing them further apart, depending on the value of \"initial_dirichlet_process_prior\". Once this distribution has been 
produced, we determine the average derivative/gradient of the provided cross validation evaluations w.r.t the hyperparameters at the center of this distribution. This is the direction 
in which we search for better hyperparameters. To search along this component, we adapt the following binary-search algorithm. We first search outwards along the search-direction with 
magnitude equal to one standard deviation of the distribution, evaluating the log-likelihood of the associated hyperparameter values, until the log-likelihood stops increasing. Then, 
we pick the largest hyperparameter set along this direction and search between it and the closest evaluated hyperparameter sets on either side. These closest sets will be reffered to as 
LEFT and RIGHT. We repeat this process, choosing the largest hyperparameter set and evaluating on either side of it (between the most-likely set and LEFT, and between the most-likely set 
and RIGHT) until our convergence requirements are met. Convergence is reached when one of two conditions are satisfied on EACH side of the most-likely hyperparameter set. The first 
condition is for the log-likelihood to decrease as we search between the most-likely set and one of the two closest sets to it along the search direction (LEFT and RIGHT). The second 
condition is for the derivative of the log-likelihood evaluations to appear to be decreasing. We say that this condition has been satisfied if extending the log-likelihood evaluated at 
LEFT OR RIGHT by the derivative on that side produces a greater log-likelihood than that associated with the most-likey set (suggesting that the derivative must decrease before we get to 
the most-likely set). The derivative on each side of the most-likely point is taken to be the average derivative between LEFT/RIGHT and the set next closest to LEFT/RIGHT on that same 
side. These convergence conditions are designed to end the search if we no longer expect to make significant progress along the current search-direction.

In more detail, given hyperparameter sets x0,x1,...,xN (in R^d) with corresponding log-likelihood evaluations f0,f1,...,fN, we compute a mean hyperparameter set mu and a covariance matrix, 
Simga, from x0,x1,...,xN using f0,f1,...,fN as prior probabilities with some float dirichlet process prior alpha, such that sum(exp(f0/alpha) + exp(f1/alpha) +...+ exp(fN/alpha)) == 1. 
Then, we determine a search direction by weighting the average derivative of the log-likelihood w.r.t. hyperparameters from each evaluated set to the newly evaluated mean. We determine 
the standard deviation of the recently computed covariance matrix in this direction and call it v. This is our initial seach vector. Given ||v|| = 1, we evaluate the 
log-likelihood, denoted as f() of r1 = mu + v, r2 = mu + 2v, r3 = mu + 4v,... and so on until we find rN | f(rN) < f(r{N-1}) and f(rN) > -inf (marking the first feasible decrease of 
log-likelihood in the direction v). If we find rN | f(rN) < f(r{N-1}) but f(rN) <= -inf (not a feasible set), then we backtrack, letting r{N+1} = (rN + r{N-1})/2, until we 
find r{N+1} | f(r{N+1}) < f(r{N}) and f(r{N+1}) > -inf. Now, assuming that f(r1) > -inf (there were any feasible sets in the direction v to begin with), we take three points, 
xM, xL, and xR (M, L, R refer to max, left, and right) from {r1,r2,r3,...} such that xM is the most likely point: f(xM) = max{f(r1),L(r2),L(r3),...}, xL is the next most likely point in 
the direction -v: f(xL) = max{f(r1),f(r2),f(r3),... | r1,r2,r3,... = xM - cv for some c > 0 and | r1,r2,r3,...!=xM}, and xR is the next most likely point in the 
direction v: f(xR) = max{f(r1),f(r2),f(r3),... | r1,r2,r3,... = xM + cv for some c > 0 and | r1,r2,r3,...!=xM}.With these three points, we can begin our binary search. If however, 
f(r1) <= -inf, then we also evaluate f(l1) = f(mu - v), f(l2) = f(mu - 2v), f(l3) = f(mu - 4v),... and so on until we find lN | f(lN) < f(l{N-1}) and f(lN) > -inf. These evaluations in 
the direction -v are necessary to obtain xL if the initial evaluations in the direction v (f(r1),f(r2),f(r3),...) only produce one feasible set (> -inf), which becomes xR.

Given xM, xL, and xR (defined above), our binary search begins with evaluation of the log-likelihood at x{L+1} = (xM+xL)/2 and at x{R+1} = (xM+xR)/2. With these new hyperparameter sets 
evaluated, we update xM, xL, and xR such that xM is the most likely point, xL is the next most likely point in the direction -v, and xR is the next most likely point in the direction v, 
like we did before. This binary search continues with x{L+2},x{L+3},...,x{L+N} and x{R+2},x{R+3},...,x{R+N}, until one of our two convergence conditions are satisfied on BOTH sides of xM. 
The first of these convergence conditions is to have f(x{L+N}) < f(xL+N-1) or f(x{R+N}) < f(xR+N-1). The second is to have f(xM) < f(x{L+N}) + ({f(x{L+N}) - f(x{L+N-1})} / ||x{L+N}-x{L+N-1}|| * ||xM-x{L+N}||) 
or f(xM) < L(x{R+N}) + ({f(x{R+N}) - L(x{R+N-1})} / ||x{R+N}-x{R_N-1}|| * ||x{M}-x{R+N}||. This notation is far from clear, especially written in a non-markup language. x{L+N} 
and x{R+N} denote the hyperparameter sets closest to the most-likely set xM in the direction -v (left) and v (right) respectively. x{L+N-1} and x{R+N-1} denote the hyperparameter 
sets closest to x{L+N} and x{R+N} in the direction -v (left) and v (right) respectively. Our intuition behind this algorithm is that the hyperparameter sets 
x{L+1},x{L+2},...,x{L+N-1},x{L+N},xM,x{R+N},x{R+N-1},...,x{R+2},x{R+1} will form something resembling a normal distribution with mean and mode xM. As our search approaches an ideal xM set 
of hyperparameters, the changes in log-likelihood from f(x{L+N-1}) to f(x{L+N}) and from f(x{R+N-1}) to f(x{R+N}) will begin to oscillate up and down based on the variance of f(). The 
first convergence condition is supposed to detect when this variance becomes large enough that our maximum log-likelihood evaluation in the current search-direction is no longer a 
reliable maximum. The second convergence condition detects when the second derivative along the current search-direction is decreasing, suggesting that the gains from searching along this 
direction are going to decline if we continue the search. Once the search has converged in the direction v, we return the log-likelihood of the most-likely set from the search and use it 
to measure the effectiveness of the current dirichlet process prior value. 

This search is repeated as many times as is required for a higher level search for the optimal dirichlet process prior to converge. The higher level search has the same convergence 
conditions as the search outlined above. The only differences are that it begins with a value provided by the user (default is 1) and that the dirichlet process prior is one-dimensional, 
so we are always search either up or down. This higher level search depends greatly on the initial dirichlet process prior value provided by the user, and generally must be run multiple 
times to narrow down a good window of values. Additionally, the more that the higher-level search is run, the more hyperparameter sets are evaluated and the greater potential that our 
distribution estimates have to discover likely sets of hyperparameters.
"""

# import modules from standard library:
import copy
import os
import sys
from typing import Callable
from ctypes import ArgumentError

# import modules from third party libraries:
import numpy as np
import torch as t
import pandas as pd

        
"""
This method runs a hyperparameter search algorithm utilizing kernel density estimations and an interpolation-search style algorithm to look for the set of hyperparameters that maximizes 
a provided cross validation function. It is designed to work with any "object" whose hyperparameters can be represented by real, continuous numbers, or by a list of real numbers.
NOTE: While the search algorithm technically works with hyperparameters that must be of type integer, such hyperparameters may get stuck at one value during the search if their sampled 
    variance is not large enough to jump the hyperparameter to a different integer. We recommend adjusting such hyperparameters to be continuous by letting decimals represent probabilities 
    between integer values (ex: 1.5 -> p(1 == 1/2) and p(2 == 1/2)).
"""
def fit(csv_filename: str, cv_function: Callable, hyperparameter_names: set, constant_hyperparameters:set = set({}), integer_hyperparameters: set = set({}), 
        list_hyperparameters:set = set({}), search_base:float = 2, search_distance:float=1, initial_dirichlet_process_prior:float = 1, num_rounds:int=1, minimum_log_likelihood: float = -np.inf, 
        minimum_search_magnitude:float = None, print_status:bool = False, dtype: type = np.float64, **kwargs):
    """
    Parameters:
    ----------
    csv_filename (str): The path of a csv file in which hyperparameter sets and their corresponding likelihood evaluations are stored. The corresponding csv file should consist of 
        num_hyperparameters+2 columns and a header row with column names. The first column is for indices (since we convert this csv to a pandas DataFrame). The next num_hyperparameters columns 
        should reference the hyperparameters that are being optimized, and the final column must be called \"log_likelihood\" and should refer to the log-likelihood evaluations produced by 
        cross validation of your model. How this file is created is up to the user, but it should contain enough hyperparameter sets to produce a full-rank covariance matrix. If enough sets are 
        not provided, then a singular covariance matrix will be produced and an error will be raised.
    
    cv_function (function/Callable): A custom method that takes a dictionary of string hyperparameter_name:value pairs and produces a float value that should represent the cross validation 
        log-likelihood of some machine-learning model given the dictionary of hyperparameters. The custom function should define, train, and evaluate your model for as many cross validation 
        folds as you would like and return the average evaluated log-likelihood.
    
    hyperparameter_names (set): A set that must hold the names of all of the hyperparameters which you want to record. Hyperparameters which you would like to remain constant can be 
        provided or ommitted. If they are provided, make sure that they also appear in \"constant_hyperparameters\". If they are ommitted, make sure that you assign them yourself in your 
        custom cross validation function \"cv_function\".
    
    constant_hyperparameters (set): A set of the names of any hyperparameters that are being recorded (exist in "hyperparameter_names") but are not to be optimized/changed at all.
    
    integer_hyperparameters (set): A set of the names of any hyperparameters that are being optimized but will only be given as integers. During the search, they will be treated as 
        continuous, but will be rounded to the nearest int before being passed to \"cv_function\".
    
    list_hyperparameters (set): A set of the names of any hyperparameters that are lists of floats or lists of integers. The associated hyperparameter values that appear in \"csv_filename\"
        must be one-dimensional lists. Any hyperparameters that are given as more than one-dimensional lists (i.e. a list of lists) will produce errors.
        
    search_base (float): The base used to search along directional vectors v. The default value is 2. When searching from an initial hyperparameter set (x), the values being added (v) 
        will be multiplied or divided by this value to produce new estimated values. That is, after evaluating the set x+v, we will either evaluate x+2v or x+v/2 depending on the 
        evaluation of x+v. TODO: explain the impact of larger or smaller values of \"search_base\"
    
    initial_dirichlet_process_prior (float): The initial value of our dirichlet process prior that acts as a prior distribution over the prior likelihoods of your recorded hyperparameter 
        sets. Larger values will concentrate log-likelihood evaluations closer together, and smaller values will push them further apart. This value has a huge impact on the effectiveness of 
        the algorithm as your custom cv_function will return values whose range depends on the number of independent samples being evaluated. Some cross validation functions may mostly return 
        values around the order of -1e2 while others around the order of -1e10 or anywhere in-between. The dirichlet process prior adjusts the relationship/similarity of these evaluations so 
        that useful distributions of the hyperparameter sets may be estimated. NOTE: We recommend that you start with a large value and decrease it as you record additional hyperparamter 
        sets. If you use too small of a value, the search will use predominantly/only the most likely set(s) to estimate a distribution. If you use too large of a value, the search will 
        not be able to narrow the window within which it is searching enough to find more likely sets.
    
    num_rounds (int): The number of rounds that the search algorithm will be run from start to finish, each time with an updated \"initial_dirichlet_process_prior\".
    
    minimum_log_likelihood (float): The smallest value of log-likelihood records that will be used in determining search directions. Hyperparameter sets that evaluate to log-likelihoods 
        less than minimum_log_likelihood will not be used when estimating the distribution of the hyperparameter sets and from the distribution a good search direction.
    
    minimum_hyperparameter_variance (float): The smallest amount of variance in the previous values of a hyperparameter that will be accepted before the algorithm searches only along that 
        hyperparameter. Searching in the direction of a single hyperparameter is necessary if it has very small variance compared to other hyperparameters, as it will not be changed if it 
        appears to have near zero variance.
          
    print_status (bool): Determines whether the status of the search algorithm is printed as it runs. This is useful for debugging or to see if it is working with some specific object and custom functions.
    
    dtype (type): The datatype to use for computations. Must be one of \"np.float64\", \"np.float32\", or \"np.float16\".
    
    kwargs (dict): Additional arguments to be provided to your custom "hyperparameter_replacement" method and your custom "cv_function" method. **kwargs will be passed to both. 
        Examples include dtypes, whether to use gpu, object constructor parameters (like the number of layers in a neural network), etc.
    
    Returns:
    ----------
    float: The best dirichlet process prior found during the search
    """
    
    # check method arguments:
    if dtype not in {np.float64, np.float32, np.float16}: raise ArgumentError("The provided argument \"dtype\" must be one of \"np.float64\", \"np.float32\", or \"np.float16\".")
    
    try:
        try:
            hyperparameter_samples: pd.DataFrame = pd.read_csv(csv_filename, header=0, index_col=0)
        except FileNotFoundError: print("CSV FILE NOT FOUND. Check that you are using the correct filename of your already existing csv file.")

        # check that the given hyperparameters match those from the given csv file
        if set(hyperparameter_samples.columns) != set(hyperparameter_names).union(set({"log_likelihood"})): raise ArgumentError("The given csv file \"hyperparameter_samples_filename\" should have the following columns: " + str(hyperparameter_names))
    
        # check that the given csv file contains at least one feasible sample (non-zero cross-validation likelihood)
        if sum(hyperparameter_samples["log_likelihood"] > minimum_log_likelihood) < 1: raise ArgumentError("The given csv file \"hyperparameter_samples_filename\" must contain at least set of hyperparameters with an evaluated log-likelihood value greater than \"minimum_log_likelihood\".")
        
        # if there are any hyperparameters given as lists of floats/ints, this block will convert each to multiple columns of individual float/ints so that the hyperparameter sets can be
        # converted to a numpy array. These "list" hyperparameters are of type "str" in the original DataFrame.
        list_amounts = [] # keeps track of how many arguments each "list" hyperparameter contains for future conversion back to "list" form
        if list_hyperparameters: # if not empty:
            temp_table = hyperparameter_samples[(set(hyperparameter_names)-set(list_hyperparameters)).union(set(constant_hyperparameters))] # temporary table to split the "list" hyperparameters into individual columns
            
            def fix_list(s: str): # short function to convert list entries from string to list of floats. We convert values to integers if necessary later on.
                s = s[1:-1].split(',')
                if s[-1] == "": s = s[:-1]
                return list(map(float, s))
            
            temp_list_hyperparameters = [] # keeps an ordered record of the new column names (replacing the old "list" columns with new individual columns)
            for key in list_hyperparameters:
                if key not in constant_hyperparameters:
                    temp_series = hyperparameter_samples[key].apply(fix_list).apply(pd.Series)
                    temp_series.columns = [key + str(n) for n in range(len(temp_series.columns))]
                    temp_list_hyperparameters.extend(temp_series.columns)
                    list_amounts.append((key, temp_series.shape[1]))
                    temp_table = pd.concat([temp_table, temp_series], axis=1)
            hyperparameter_samples = pd.concat([temp_table, hyperparameter_samples["log_likelihood"]], axis=1) # put log-likelihood at the last column in dataframe
            non_constant_hyperparameters:list = list(set(hyperparameter_samples.columns) - set({"log_likelihood"}) - set(temp_list_hyperparameters) - set(constant_hyperparameters))
            non_constant_hyperparameters.extend(list(temp_list_hyperparameters)) # put list hyperparameters after non-list for converting back to dataframe from numpy later
        else: non_constant_hyperparameters:list = list(set(hyperparameter_samples.columns) - set({"log_likelihood"}) - set(constant_hyperparameters))
        
        # begin search algorithm. After each iteration of the loop below, we update the variable "initial_dirichlet_process_prior" with the current best estimate from the search and do it again.
        mid_dirichlet_process_prior:float = initial_dirichlet_process_prior
        hyperparameter_samples_container = [hyperparameter_samples] # store the DataFrame containing our hyperparameter sets and evaluations in an array so that we can modify it outside of this method.
        for n in range(num_rounds):
            scale = 1
            # call the method to create distribution using "initial_dirichlet_process_prior" and use it to search for new sets of hyperparameter samples
            mid_likelihood = _minimal_covariance_sampling(cv_function, mid_dirichlet_process_prior, hyperparameter_samples_container, csv_filename, hyperparameter_names, constant_hyperparameters, integer_hyperparameters, non_constant_hyperparameters, list_hyperparameters, list_amounts, minimum_log_likelihood, search_base, search_distance , minimum_search_magnitude, print_status, dtype, **kwargs)
            
            # increase the dirichlet process prior and repeat
            right_dirichlet_process_prior = mid_dirichlet_process_prior*search_base
            right_likelihood = _minimal_covariance_sampling(cv_function, right_dirichlet_process_prior, hyperparameter_samples_container, csv_filename, hyperparameter_names, constant_hyperparameters, integer_hyperparameters, non_constant_hyperparameters, list_hyperparameters, list_amounts, minimum_log_likelihood, search_base, search_distance , minimum_search_magnitude, print_status, dtype, **kwargs)
            
            # if increasing the dirichlet process prior produced a MORE likely mean hyperparameter set, then continue increasing it until the likelihood decreases.
            if right_likelihood > mid_likelihood: # increasing dirichlet prior increased likelihood
                while right_likelihood > mid_likelihood: # continue to increase by *search_base until likelihood decreases
                    scale+=1
                    left_likelihood, left_dirichlet_process_prior = mid_likelihood, mid_dirichlet_process_prior
                    mid_likelihood, mid_dirichlet_process_prior = right_likelihood, right_dirichlet_process_prior
                    right_dirichlet_process_prior = right_dirichlet_process_prior*search_base**scale
                    right_likelihood = _minimal_covariance_sampling(cv_function, right_dirichlet_process_prior, hyperparameter_samples_container, csv_filename, hyperparameter_names, constant_hyperparameters, integer_hyperparameters, non_constant_hyperparameters, list_hyperparameters, list_amounts, minimum_log_likelihood, search_base, search_distance , minimum_search_magnitude, print_status, dtype, **kwargs)
            
            # if increasing the dirichlet process prior produced a LESS likely mean hyperparameter set, then decrease the dirichlet process prior it until the likelihood decreases.
            else: # increasing dirichlet prior decreased likelihood
                left_dirichlet_process_prior = mid_dirichlet_process_prior/search_base # continue to decrease by /search_base until likelihood decreases
                left_likelihood = _minimal_covariance_sampling(cv_function, left_dirichlet_process_prior, hyperparameter_samples_container, csv_filename, hyperparameter_names, constant_hyperparameters, integer_hyperparameters, non_constant_hyperparameters, list_hyperparameters, list_amounts, minimum_log_likelihood, search_base, search_distance , minimum_search_magnitude, print_status, dtype, **kwargs)
                while left_likelihood > mid_likelihood:
                    scale+=1
                    right_likelihood, right_dirichlet_process_prior = mid_likelihood, mid_dirichlet_process_prior
                    mid_likelihood, mid_dirichlet_process_prior = left_likelihood, left_dirichlet_process_prior
                    left_dirichlet_process_prior = left_dirichlet_process_prior/search_base**scale
                    left_likelihood = _minimal_covariance_sampling(cv_function, left_dirichlet_process_prior, hyperparameter_samples_container, csv_filename, hyperparameter_names, constant_hyperparameters, integer_hyperparameters, non_constant_hyperparameters, list_hyperparameters, list_amounts, minimum_log_likelihood, search_base, search_distance , minimum_search_magnitude, print_status, dtype, **kwargs)
                
            # likelihood is no longer increasing as we search outwards from the initial dirichlet process prior. Now interpolate between the mode and the points closest to it
            outer_left_likelihood, outer_left_dirichlet_process_prior = left_likelihood, left_dirichlet_process_prior
            outer_right_likelihood, outer_right_dirichlet_process_prior = right_likelihood, right_dirichlet_process_prior
            left_dirichlet_process_prior = (outer_left_dirichlet_process_prior+mid_dirichlet_process_prior)/2
            left_likelihood = _minimal_covariance_sampling(cv_function, left_dirichlet_process_prior, hyperparameter_samples_container, csv_filename, hyperparameter_names, constant_hyperparameters, integer_hyperparameters, non_constant_hyperparameters, list_hyperparameters, list_amounts, minimum_log_likelihood, search_base, search_distance , minimum_search_magnitude, print_status, dtype, **kwargs)
            right_dirichlet_process_prior = (outer_right_dirichlet_process_prior+mid_dirichlet_process_prior)/2
            right_likelihood = _minimal_covariance_sampling(cv_function, right_dirichlet_process_prior, hyperparameter_samples_container, csv_filename, hyperparameter_names, constant_hyperparameters, integer_hyperparameters, non_constant_hyperparameters, list_hyperparameters, list_amounts, minimum_log_likelihood, search_base, search_distance , minimum_search_magnitude, print_status, dtype, **kwargs)
            
            # continue to search between mode and the points closest to it until convergence it reached on both sides of the mode.
            no_left, no_right= False, False # these keep track of convergence on each side
            while True:
                if left_likelihood < outer_left_likelihood: no_left = True
                elif left_likelihood + (left_likelihood-outer_left_likelihood)/np.linalg.norm(left_dirichlet_process_prior-outer_left_dirichlet_process_prior)*np.linalg.norm(mid_dirichlet_process_prior-left_dirichlet_process_prior) > mid_likelihood: no_left=True
                if right_likelihood < outer_right_likelihood: no_right = True
                elif right_likelihood + (right_likelihood-outer_right_likelihood)/np.linalg.norm(right_dirichlet_process_prior-outer_right_dirichlet_process_prior)*np.linalg.norm(mid_dirichlet_process_prior-right_dirichlet_process_prior) > mid_likelihood: no_right=True
                if no_left and no_right:
                    if print_status: print(f"ROUND {n+1}HYPERPARAMETER SEARCH CONVERGED WITH DIRICHLET PROCESS PRIOR OF " + str(mid_dirichlet_process_prior))
                    break
                
                # if left is the most likely set, then left becomes mid and we search around it
                if left_likelihood >= right_likelihood and left_likelihood >= mid_likelihood: 
                    no_left, no_right = False, False
                    outer_right_likelihood, outer_right_dirichlet_process_prior = mid_likelihood, mid_dirichlet_process_prior
                    mid_likelihood, mid_dirichlet_process_prior = left_likelihood, outer_left_dirichlet_process_prior
                    
                    right_dirichlet_process_prior = (mid_dirichlet_process_prior+outer_right_dirichlet_process_prior)/2
                    right_likelihood = _minimal_covariance_sampling(cv_function, right_dirichlet_process_prior, hyperparameter_samples_container, csv_filename, hyperparameter_names, constant_hyperparameters, integer_hyperparameters, non_constant_hyperparameters, list_hyperparameters, list_amounts, minimum_log_likelihood, search_base, search_distance , minimum_search_magnitude, print_status, **kwargs)
                    
                    left_dirichlet_process_prior = (mid_dirichlet_process_prior+outer_left_dirichlet_process_prior)/2
                    left_likelihood = _minimal_covariance_sampling(cv_function, left_dirichlet_process_prior, hyperparameter_samples_container, csv_filename, hyperparameter_names, constant_hyperparameters, integer_hyperparameters, non_constant_hyperparameters, list_hyperparameters, list_amounts, minimum_log_likelihood, search_base, search_distance , minimum_search_magnitude, print_status, **kwargs)

                # if right is the most likely set, then right becomes mid and we search around it
                elif right_likelihood > left_likelihood and right_likelihood >= mid_likelihood:
                    no_left, no_right = False, False
                    outer_left_likelihood, outer_left_dirichlet_process_prior = mid_likelihood, mid_dirichlet_process_prior
                    mid_likelihood, mid_dirichlet_process_prior= right_likelihood, right_dirichlet_process_prior
                    
                    left_dirichlet_process_prior = (mid_dirichlet_process_prior+outer_left_dirichlet_process_prior)/2
                    left_likelihood = _minimal_covariance_sampling(cv_function, left_dirichlet_process_prior, hyperparameter_samples_container, csv_filename, hyperparameter_names, constant_hyperparameters, integer_hyperparameters, non_constant_hyperparameters, list_hyperparameters, list_amounts, minimum_log_likelihood, search_base, search_distance , minimum_search_magnitude, print_status, **kwargs)
                    
                    right_dirichlet_process_prior = (mid_dirichlet_process_prior+outer_right_dirichlet_process_prior)/2
                    right_likelihood = _minimal_covariance_sampling(cv_function, right_dirichlet_process_prior, hyperparameter_samples_container, csv_filename, hyperparameter_names, constant_hyperparameters, integer_hyperparameters, non_constant_hyperparameters, list_hyperparameters, list_amounts, minimum_log_likelihood, search_base, search_distance , minimum_search_magnitude, print_status, **kwargs)
                
                else: # mid stays the most likely and we search around it
                    if not no_left:
                        outer_left_likelihood, outer_left_dirichlet_process_prior = left_likelihood, left_dirichlet_process_prior
                        left_dirichlet_process_prior = (outer_left_dirichlet_process_prior+mid_dirichlet_process_prior)/2
                        left_likelihood = _minimal_covariance_sampling(cv_function, left_dirichlet_process_prior, hyperparameter_samples_container, csv_filename, hyperparameter_names, constant_hyperparameters, integer_hyperparameters, non_constant_hyperparameters, list_hyperparameters, list_amounts, minimum_log_likelihood, search_base, search_distance , minimum_search_magnitude, print_status, dtype, **kwargs)
                    if not no_right:
                        outer_right_likelihood, outer_right_dirichlet_process_prior = right_likelihood, right_dirichlet_process_prior
                        right_dirichlet_process_prior = (outer_right_dirichlet_process_prior+mid_dirichlet_process_prior)/2
                        right_likelihood = _minimal_covariance_sampling(cv_function, right_dirichlet_process_prior, hyperparameter_samples_container, csv_filename, hyperparameter_names, constant_hyperparameters, integer_hyperparameters, non_constant_hyperparameters, list_hyperparameters, list_amounts, minimum_log_likelihood, search_base, search_distance , minimum_search_magnitude, print_status, dtype, **kwargs)
        return mid_dirichlet_process_prior
    except MemoryError: raise MemoryError(f"There is not enough memory in ram to fit a kernel density estimate to the provided hyperparameter sets.")
    
"""
This method takes a pandas DataFrame of hyperparameter sets and their associated log-likelihoods, along with a dirichlet process prior (float) forms a multivariate normal 
distribution from the hyperparameter sets with their log-likelihoods as a prior distribution and with the dirichlet process prior as a prior over the log-likelihoods. The mean set of hyperparameters from this distribution is evaluated using a 
provided cross validation function and that log-likelihood value is recorded. Then, additional sets of hyperparameters are sampled along the direction of minimum variance from 
the mean and their log-likelihoods are also evaluated. The search along the direction of minimum variance ends when the rate at which the log-likelihood values increase starts to decrease 
(essentially when the second derivative of the cross validation function decreases).
"""
def _minimal_covariance_sampling(cv_function:Callable, dirichlet_process_prior:float, hyperparameter_samples_container:list[pd.DataFrame], csv_filename:str, hyperparameter_names:set, 
        constant_hyperparameters:set, integer_hyperparameters:set, non_constant_hyperparameters:list, list_hyperparameters:set, list_amounts:list, minimum_log_likelihood:float, 
        search_base:float, search_distance:float, minimum_search_magnitude:float, print_status:bool, dtype:type, **kwargs):
    """
    See the fit() method above for details about these method arguments.
    
    Returns:
    ----------
    float: the log-likelihood evaluation of the mode along the determined search direction. This is supposed to represent the effectiveness of the given dirichlet process prior 
    """
    if print_status: print("NEW DIRICHLET PRIOR: " + str(dirichlet_process_prior))
    
    # convert hyperparameter sets from pandas DataFrame to numpy array
    hyperparameter_samples_array:np.ndarray = hyperparameter_samples_container[0][non_constant_hyperparameters][hyperparameter_samples_container[0]["log_likelihood"] > minimum_log_likelihood].to_numpy()
    
    # Fit normal distribution to approximate prior distribution of samples. The log-likelihoods of the hyperparameter sets will be divided by these likelihood values to reduce the 
    # amount of bias present in the distribution estimate from frequently sampled sets.
    
    # check that the provided dataset of hyperparameters spans the space. This is necessary for the search to be effective, as it can only search along the space span by previously evaluated sets of hyperparameters.
    mean = np.mean(hyperparameter_samples_array, axis = 0)
    diffs = hyperparameter_samples_array - mean
    stds = np.sqrt(np.abs(np.einsum("ij,ij->j",diffs, diffs)))
    if any(np.round(stds,10) == 0): raise ArgumentError("At least one hyperparameter has zero variance. This will cause the search to fail. Sample more sets while varying these hyperparameters.")
    diffs = diffs / stds
    cov = np.dot(diffs.T, diffs) / hyperparameter_samples_array.shape[0]
    _,s,_ = np.linalg.svd(cov)
    if any(np.round(s,10) == 0): raise ArgumentError("The covariance matrix of the provided dataset of hyperparameters is singular. This will cause the search to fail. Sample more hyperparameter sets to add variance in this direction.")

    # estimate prior likelihoods using a multivariate normal distribution. This reduces bias from frequently sampled sets
    prior_log_likelihoods = (-1/2) * np.linalg.slogdet(cov)[1] + (-1/2) * np.sum(np.abs(np.dot(diffs, np.linalg.inv(cov)) * diffs), axis=1)[:,None]
    prior_likelihoods = np.exp(prior_log_likelihoods + 30 - np.max(prior_log_likelihoods))
    prior_likelihoods = prior_likelihoods / np.sum(prior_likelihoods)
    # estimate mean of multivariate normal distribution for the hyperparameter sets given the above prior and the dirichlet process prior
    original_log_likelihoods:np.ndarray = hyperparameter_samples_container[0]["log_likelihood"][hyperparameter_samples_container[0]["log_likelihood"] > minimum_log_likelihood].to_numpy()[:,None]
    adjusted_log_likelihoods = original_log_likelihoods - np.max(original_log_likelihoods) # ensure that all log-likelihoods are <= zero
    adjusted_log_likelihoods = adjusted_log_likelihoods/prior_likelihoods # apply dirichlet prior to widen or narrow the search space
    adjusted_log_likelihoods = adjusted_log_likelihoods/dirichlet_process_prior
    adjusted_log_likelihoods += 30 - np.max(adjusted_log_likelihoods)
    likelihoods = np.exp(adjusted_log_likelihoods)
    likelihoods = (likelihoods/np.sum(likelihoods))

    if print_status: print(f"Using dirichlet value of {dirichlet_process_prior}, the most-likely hyperparameter set is given weight {np.max(likelihoods)} out of 1.")
    weighted_mean = np.sum(hyperparameter_samples_array * likelihoods, axis=0)
    
    # evaluate log-likelihood of the mean
    if print_status: print("MEAN: " + str(weighted_mean))
    mean_likelihood = _evaluate_likelihood(cv_function, weighted_mean, hyperparameter_samples_container, csv_filename, hyperparameter_names, constant_hyperparameters, integer_hyperparameters, non_constant_hyperparameters, list_hyperparameters, list_amounts, print_status, **kwargs)
    if print_status: print("MEAN LIKELIHOOD: " + str(mean_likelihood))
    if np.isneginf(mean_likelihood): return mean_likelihood # return if not a feasible set -> bad dirichlet process prior
    
    # estimate covariance matrix
    diffs = weighted_mean - hyperparameter_samples_array
    cov = np.dot(diffs.T, diffs*likelihoods)
    
    # compute average directional derivative at the mean.
    directional_derivatives = (mean_likelihood-original_log_likelihoods) / np.einsum("ij,ij->i", diffs, diffs)[:,None] * diffs
    
    # divide each directional derivative by the prior_likelihoods. This reduces bias from frequently sampled areas.
    search_direction = np.sum(directional_derivatives / prior_likelihoods, axis=0)
    
    # apply dirichlet process prior
    search_direction = search_direction / np.sqrt(np.diag(cov))
    search_direction = (search_direction/np.sum(np.abs(search_direction)))
    signs = np.sign(search_direction)
    search_direction = np.abs(search_direction)**(1/dirichlet_process_prior)
    search_direction = search_direction / np.sum(search_direction)
    search_direction = search_direction*signs * np.sqrt(np.diag(cov))
    
    # apply covariance matrix so that we are initially searching one standard deviation away from the mean
    search_direction = search_direction / np.linalg.norm(search_direction)
    search_direction = search_direction * np.sqrt(np.dot(np.dot(search_direction, cov),search_direction)) * search_distance
    if minimum_search_magnitude is not None:
        search_direction = np.maximum([np.abs(search_direction), minimum_search_magnitude]) * np.sign(search_direction)

    if print_status: print("NEW SEARCH DIRECTION: " + str(search_direction))
    mid_likelihood = mean_likelihood
    mid_hyperparameter_sample = weighted_mean
    
    # evaluate hyperparameter samples in the search_direction until log-likelihood no longer increasing
    left_likelihood = minimum_log_likelihood
    count = 0
    scale = 0
    left_hyperparameter_sample = weighted_mean + search_direction
    left_likelihood = _evaluate_likelihood(cv_function, left_hyperparameter_sample, hyperparameter_samples_container, csv_filename, hyperparameter_names, constant_hyperparameters, integer_hyperparameters, non_constant_hyperparameters, list_hyperparameters, list_amounts, print_status, **kwargs)
    right_likelihood = minimum_log_likelihood
    while True:
        count+=1
        # if left is better than mid, step left again
        if left_likelihood >= mid_likelihood:
            right_likelihood, right_hyperparameter_sample = mid_likelihood, mid_hyperparameter_sample
            mid_likelihood, mid_hyperparameter_sample = left_likelihood, left_hyperparameter_sample
            left_hyperparameter_sample = mid_hyperparameter_sample + (mid_hyperparameter_sample-right_hyperparameter_sample)*search_base
            left_likelihood = _evaluate_likelihood(cv_function, left_hyperparameter_sample, hyperparameter_samples_container, csv_filename, hyperparameter_names, constant_hyperparameters, integer_hyperparameters, non_constant_hyperparameters, list_hyperparameters, list_amounts, print_status, **kwargs)
            # upon first decrease in left, if feasible, break.
            if left_likelihood <= mid_likelihood and left_likelihood > minimum_log_likelihood: break
            else: continue
        # if left is not feasible, step right.
        if left_likelihood <= minimum_log_likelihood:
            scale+=1
            left_hyperparameter_sample = (left_hyperparameter_sample + (2**scale-1)*mid_hyperparameter_sample)/2**scale
            left_likelihood = _evaluate_likelihood(cv_function, left_hyperparameter_sample, hyperparameter_samples_container, csv_filename, hyperparameter_names, constant_hyperparameters, integer_hyperparameters, non_constant_hyperparameters, list_hyperparameters, list_amounts, print_status, **kwargs)
            # upon first feasible solution, break
            if left_likelihood > minimum_log_likelihood: break
            else: continue
        else: break
    
    # do the same in right direction if right direction is still less than min -> left is worse than mid
    if right_likelihood <= minimum_log_likelihood:
        right_hyperparameter_sample = weighted_mean - search_direction
        right_likelihood = _evaluate_likelihood(cv_function, right_hyperparameter_sample, hyperparameter_samples_container, csv_filename, hyperparameter_names, constant_hyperparameters, integer_hyperparameters, non_constant_hyperparameters, list_hyperparameters, list_amounts, print_status, **kwargs)
        scale = 0
        count+=1
        while True:
            count+=1
            if right_likelihood >= mid_likelihood:
                left_likelihood, left_hyperparameter_sample = mid_likelihood, mid_hyperparameter_sample
                mid_likelihood, mid_hyperparameter_sample = right_likelihood, right_hyperparameter_sample
                right_hyperparameter_sample = mid_hyperparameter_sample + (mid_hyperparameter_sample-left_hyperparameter_sample)*search_base
                right_likelihood = _evaluate_likelihood(cv_function, right_hyperparameter_sample, hyperparameter_samples_container, csv_filename, hyperparameter_names, constant_hyperparameters, integer_hyperparameters, non_constant_hyperparameters, list_hyperparameters, list_amounts, print_status, **kwargs)
                # upon first decrease in right, if feasible, break.
                if right_likelihood <= mid_likelihood and right_likelihood > minimum_log_likelihood: break
                else: continue
            if right_likelihood <= minimum_log_likelihood: 
                scale+=1
                right_hyperparameter_sample = ((2**scale-1)*mid_hyperparameter_sample+right_hyperparameter_sample)/2**scale
                right_likelihood = _evaluate_likelihood(cv_function, right_hyperparameter_sample, hyperparameter_samples_container, csv_filename, hyperparameter_names, constant_hyperparameters, integer_hyperparameters, non_constant_hyperparameters, list_hyperparameters, list_amounts, print_status, **kwargs)
                if right_likelihood > minimum_log_likelihood: break
                else: continue
            else: break
    # make sure that left and mid are not the same
    if np.sum(np.abs(left_hyperparameter_sample)) == np.sum(np.abs(mid_hyperparameter_sample)) or np.sum(np.abs(right_hyperparameter_sample)) == np.sum(np.abs(mid_hyperparameter_sample)): 
        mid_hyperparameter_sample = (left_hyperparameter_sample+right_hyperparameter_sample)/2
        mid_likelihood = _evaluate_likelihood(cv_function, mid_hyperparameter_sample, hyperparameter_samples_container, csv_filename, hyperparameter_names, constant_hyperparameters, integer_hyperparameters, non_constant_hyperparameters, list_hyperparameters, list_amounts, print_status, **kwargs)
        count+=1
        
    # interpolate by searching between the mode and the sets next to it on either side. Repeat this until the likelihood decreases or until the rate of change of log-likelihood wrt hyperparameters is expected to decrease on BOTH sides of the mode.
    outer_left_likelihood, outer_left_hyperparameter_sample = left_likelihood, left_hyperparameter_sample
    outer_right_likelihood, outer_right_hyperparameter_sample = right_likelihood, right_hyperparameter_sample
    
    left_hyperparameter_sample = (left_hyperparameter_sample + mid_hyperparameter_sample)/2
    left_likelihood = _evaluate_likelihood(cv_function, left_hyperparameter_sample, hyperparameter_samples_container, csv_filename, hyperparameter_names, constant_hyperparameters, integer_hyperparameters, non_constant_hyperparameters, list_hyperparameters, list_amounts, print_status, **kwargs)
    
    right_hyperparameter_sample = (right_hyperparameter_sample+mid_hyperparameter_sample)/2
    right_likelihood = _evaluate_likelihood(cv_function, right_hyperparameter_sample, hyperparameter_samples_container, csv_filename, hyperparameter_names, constant_hyperparameters, integer_hyperparameters, non_constant_hyperparameters, list_hyperparameters, list_amounts, print_status, **kwargs)
    count+=2 
            
    no_left, no_right = False, False # keep track of convergence on each side
    while True: # loop
        if left_likelihood < outer_left_likelihood: no_left = True
        if left_likelihood + (left_likelihood-outer_left_likelihood)/np.linalg.norm(left_hyperparameter_sample-outer_left_hyperparameter_sample)*np.linalg.norm(mid_hyperparameter_sample-left_hyperparameter_sample) > mid_likelihood: no_left=True
        if right_likelihood < outer_right_likelihood: no_right = True
        if right_likelihood + (right_likelihood-outer_right_likelihood)/np.linalg.norm(right_hyperparameter_sample-outer_right_hyperparameter_sample)*np.linalg.norm(mid_hyperparameter_sample-right_hyperparameter_sample) > mid_likelihood: no_right=True
        if no_left and no_right: 
            if print_status: print(f"HYPERPARAMETER SEARCH IN THE FOLLOWING DIRECTION CONVERGED AFTER {count} STEPS: " + str(search_direction))
            return mid_likelihood # return log-likelihood of where the search ended. This represents how effective the dirichlet process prior was in providing a new search direction.
        
        # set on the left side is most-likely, so it becomes mid and we search around it
        if left_likelihood >= right_likelihood and left_likelihood >= mid_likelihood:
            no_left, no_right = False, False
            outer_right_likelihood, outer_right_hyperparameter_sample = mid_likelihood, mid_hyperparameter_sample
            mid_likelihood, mid_hyperparameter_sample = left_likelihood, left_hyperparameter_sample
            
            right_hyperparameter_sample = (mid_hyperparameter_sample+outer_right_hyperparameter_sample)/2
            right_likelihood = _evaluate_likelihood(cv_function, right_hyperparameter_sample, hyperparameter_samples_container, csv_filename, hyperparameter_names, constant_hyperparameters, integer_hyperparameters, non_constant_hyperparameters, list_hyperparameters, list_amounts, print_status, **kwargs)
            
            left_hyperparameter_sample = (mid_hyperparameter_sample+outer_left_hyperparameter_sample)/2
            left_likelihood = _evaluate_likelihood(cv_function, left_hyperparameter_sample, hyperparameter_samples_container, csv_filename, hyperparameter_names, constant_hyperparameters, integer_hyperparameters, non_constant_hyperparameters, list_hyperparameters, list_amounts, print_status, **kwargs)
            count+=2
        # set on the right side is most-likely, so it becomes mid and we search around it
        elif right_likelihood > left_likelihood and right_likelihood >= mid_likelihood:
            no_left, no_right = False, False
            outer_left_likelihood, outer_left_hyperparameter_sample = mid_likelihood, mid_hyperparameter_sample
            mid_likelihood, mid_hyperparameter_sample = right_likelihood, right_hyperparameter_sample
            
            left_hyperparameter_sample = (mid_hyperparameter_sample+outer_left_hyperparameter_sample)/2
            left_likelihood = _evaluate_likelihood(cv_function, left_hyperparameter_sample, hyperparameter_samples_container, csv_filename, hyperparameter_names, constant_hyperparameters, integer_hyperparameters, non_constant_hyperparameters, list_hyperparameters, list_amounts, print_status, **kwargs)
            
            right_hyperparameter_sample = (mid_hyperparameter_sample+outer_right_hyperparameter_sample)/2
            right_likelihood = _evaluate_likelihood(cv_function, right_hyperparameter_sample, hyperparameter_samples_container, csv_filename, hyperparameter_names, constant_hyperparameters, integer_hyperparameters, non_constant_hyperparameters, list_hyperparameters, list_amounts, print_status, **kwargs)
            count+=2
        else: # mid stays the most-likely and we search around it again
            if not no_left:
                outer_left_hyperparameter_sample = left_hyperparameter_sample
                left_hyperparameter_sample = (mid_hyperparameter_sample+left_hyperparameter_sample)/2
                left_likelihood = _evaluate_likelihood(cv_function, left_hyperparameter_sample, hyperparameter_samples_container, csv_filename, hyperparameter_names, constant_hyperparameters, integer_hyperparameters, non_constant_hyperparameters, list_hyperparameters, list_amounts, print_status, **kwargs)
                count+=1
            if not no_right:
                outer_right_hyperparameter_sample = right_hyperparameter_sample
                right_hyperparameter_sample = (mid_hyperparameter_sample+right_hyperparameter_sample)/2
                right_likelihood = _evaluate_likelihood(cv_function, right_hyperparameter_sample, hyperparameter_samples_container, csv_filename, hyperparameter_names, constant_hyperparameters, integer_hyperparameters, non_constant_hyperparameters, list_hyperparameters, list_amounts, print_status, **kwargs)
                count+=1
"""
This method takes a hyperparameter set as a numpy array, converts it to a dictionary that the user's custom cross validation function can take as an argument, and records the set and its 
evaluated log-likelihood in the user's csv file containing other previously evaluated hyperparameter sets.
"""
def _evaluate_likelihood(cv_function:Callable, hyperparameter_sample:np.ndarray, samples_table_container:list[pd.DataFrame], csv_filename:str, hyperparameter_names:set, 
    constant_hyperparameters:set, integer_hyperparameters:set, non_constant_hyperparameters:set, list_hyperparameters:set, list_amounts:list, print_status:bool, previous_evaluation_index:int = None, **kwargs):
    """
    See the fit() method above for details about method parameters.
    
    Returns:
    ----------
    float: The log-likelihood produced by \"cv_function\" when passed the given hyperparameters.
    """
    if print_status: print("NEW SAMPLE: " + str(hyperparameter_sample))
    
    temp_table = pd.DataFrame(columns=non_constant_hyperparameters) # create dataframe as intermediate between numpy and dict
    temp_table.loc[0] = hyperparameter_sample # put numpy hyperparameter set into table for manipulation
    expanded_hyperparameter_sample_dict = dict(temp_table.loc[0]) # contains non-constant hyperparameters with "list" type hyperparameters expanded into individual float/int objects
    if list_hyperparameters: # not empty - need to restore lists for replacement method
        num_non_list_non_constant_hyperparameters = len(hyperparameter_names)-len(set(list_hyperparameters).union(set(constant_hyperparameters)))
        for key, num in list_amounts:
            temp_table[key] = temp_table.iloc[:,num_non_list_non_constant_hyperparameters:num_non_list_non_constant_hyperparameters+num].apply(list, axis=1) # add column with list
            temp_table = pd.concat([temp_table.iloc[:,:num_non_list_non_constant_hyperparameters],temp_table.iloc[:,num_non_list_non_constant_hyperparameters+num:]], axis=1) # remove columns that were adjusted to not be list
    function_hyperparameter_sample_dict = dict(temp_table.loc[0]) # contains non-constant hyperparameters in list format
    
    # add constant hyperparameters to no-list dict
    constant_hyperparameter_dict = dict(samples_table_container[0][list(constant_hyperparameters)].loc[0])
    function_hyperparameter_sample_dict = function_hyperparameter_sample_dict | constant_hyperparameter_dict
    expanded_hyperparameter_sample_dict = expanded_hyperparameter_sample_dict | constant_hyperparameter_dict
    # check the replacement dict for integer hyperparameters
    for key in function_hyperparameter_sample_dict.keys(): # convert necessary hyperparameters to integers
        if key in integer_hyperparameters:
            if key in list_hyperparameters: function_hyperparameter_sample_dict[key] = [round(num) for num in function_hyperparameter_sample_dict[key]]
            else: function_hyperparameter_sample_dict[key] = round(function_hyperparameter_sample_dict[key])
    
    log_likelihood = cv_function(copy.deepcopy(function_hyperparameter_sample_dict), **kwargs)
    if previous_evaluation_index is not None:
        log_likelihood = (log_likelihood+samples_table_container[0]["log_likelihood"][previous_evaluation_index])/2
        samples_table_container[0] = samples_table_container[0].drop(previous_evaluation_index, axis=0) # HERE
        expanded_hyperparameter_sample_dict["log_likelihood"] = log_likelihood
        samples_table_container[0].loc[previous_evaluation_index] = expanded_hyperparameter_sample_dict
        samples_table_container[0].sort_index()
    else:
        expanded_hyperparameter_sample_dict["log_likelihood"] = log_likelihood
        samples_table_container[0].loc[len(samples_table_container[0].index)] = expanded_hyperparameter_sample_dict
    if print_status:
        print("YIELDED LOG-LIKELIHOOD: " + str(log_likelihood))
    
    # update the given csv file by recording the current hyperparameter set and its evaluated log-likelihood - best to do this now to avoid losing data in the event of an error
    temp_table = copy.deepcopy(samples_table_container[0]) # holds a temporary table that is reformatted to the original form
    if list_amounts is not None: # convert expanded hyperparameters back to "list" type objects
        num_non_list_non_constant_hyperparameters = len(hyperparameter_names)-len(set(list_hyperparameters)-(set(constant_hyperparameters)))
        for key, num in list_amounts:
            temp_table[key] = temp_table.iloc[:,num_non_list_non_constant_hyperparameters:num_non_list_non_constant_hyperparameters+num].apply(list, axis=1) # log-likelihood should be in column -1
            temp_table = pd.concat([temp_table.iloc[:,:num_non_list_non_constant_hyperparameters],temp_table.iloc[:,num_non_list_non_constant_hyperparameters+num:]], axis=1)
        temp_series = temp_table["log_likelihood"]
        temp_table = temp_table.drop(["log_likelihood"], axis=1)
        temp_table["log_likelihood"] = temp_series # puts log-likelihood column at the end
    temp_table[list(integer_hyperparameters)] = temp_table[list(integer_hyperparameters)].round(0).astype("int64")
    temp_table.to_csv(csv_filename) # overwrite previous csv file 
    
    return log_likelihood # return evaluated log-likelihood for the search

"""

""" # HERE WANT TO ADD OPTION TO SEARCH ALONG ONE FEATURE
def initial_search(csv_filename: str, cv_function: Callable, hyperparameter_names: set, constant_hyperparameters:set = set({}), integer_hyperparameters: set = set({}), 
    list_hyperparameters:set = set({}), search_base:float = 2, search_distance:float = 0.5, search_hyperparameter:str=None, num_rounds:int=1, minimum_log_likelihood: float = -np.inf, minimum_search:float=1e-4, print_status:bool = False, dtype: type = np.float64, 
    **kwargs):
    """
    
    """
    # check method arguments:
    if dtype not in {np.float64, np.float32, np.float16}: raise ArgumentError("The provided argument \"dtype\" must be one of \"np.float64\", \"np.float32\", or \"np.float16\".")
    
    try:
        try:
            hyperparameter_samples: pd.DataFrame = pd.read_csv(csv_filename, header=0, index_col=0)
        except FileNotFoundError: print("CSV FILE NOT FOUND. Check that you are using the correct filename of your already existing csv file.")

        # check that the given hyperparameters match those from the given csv file
        if set(hyperparameter_samples.columns) != set(hyperparameter_names).union(set({"log_likelihood"})): raise ArgumentError("The given csv file \"hyperparameter_samples_filename\" should have the following columns: " + str(hyperparameter_names))
    
        # check that the given csv file contains at least one feasible sample (non-zero cross-validation likelihood)
        if sum(hyperparameter_samples["log_likelihood"] > minimum_log_likelihood) < 1: raise ArgumentError("The given csv file \"hyperparameter_samples_filename\" must contain at least set of hyperparameters with an evaluated log-likelihood value greater than \"minimum_log_likelihood\".")
        
        # if there are any hyperparameters given as lists of floats/ints, this block will convert each to multiple columns of individual float/ints so that the hyperparameter sets can be
        # converted to a numpy array. These "list" hyperparameters are of type "str" in the original DataFrame.
        list_amounts = [] # keeps track of how many arguments each "list" hyperparameter contains for future conversion back to "list" form
        if list_hyperparameters: # if not empty:
            temp_table = hyperparameter_samples[(set(hyperparameter_names)-set(list_hyperparameters)).union(set(constant_hyperparameters))] # temporary table to split the "list" hyperparameters into individual columns
            
            def fix_list(s: str): # short function to convert list entries from string to list of floats. We convert values to integers if necessary later on.
                s = s[1:-1].split(',')
                if s[-1] == "": s = s[:-1]
                return list(map(float, s))
            
            temp_list_hyperparameters = [] # keeps an ordered record of the new column names (replacing the old "list" columns with new individual columns)
            for key in list_hyperparameters:
                if key not in constant_hyperparameters:
                    temp_series = hyperparameter_samples[key].apply(fix_list).apply(pd.Series)
                    temp_series.columns = [key + str(n) for n in range(len(temp_series.columns))]
                    temp_list_hyperparameters.extend(temp_series.columns)
                    list_amounts.append((key, temp_series.shape[1]))
                    temp_table = pd.concat([temp_table, temp_series], axis=1)
            hyperparameter_samples = pd.concat([temp_table, hyperparameter_samples["log_likelihood"]], axis=1) # put log-likelihood at the last column in dataframe
            non_constant_hyperparameters:list = list(set(hyperparameter_samples.columns) - set({"log_likelihood"}) - set(temp_list_hyperparameters) - set(constant_hyperparameters))
            non_constant_hyperparameters.extend(list(temp_list_hyperparameters)) # put list hyperparameters after non-list for converting back to dataframe from numpy later
        else: non_constant_hyperparameters:list = list(set(hyperparameter_samples.columns) - set({"log_likelihood"}) - set(constant_hyperparameters))
    
        # perform binary search along each axis
        for _ in range(num_rounds):
            for n in range(len(non_constant_hyperparameters)):
                # get mode, evaluate it again and average the log-likelihoods
                mid_hyperparameter_sample:np.ndarray = hyperparameter_samples.iloc[hyperparameter_samples["log_likelihood"].idxmax()][non_constant_hyperparameters].to_numpy()
                mid_likelihood = _evaluate_likelihood(cv_function, mid_hyperparameter_sample, [hyperparameter_samples], csv_filename, hyperparameter_names, constant_hyperparameters, integer_hyperparameters, non_constant_hyperparameters, list_hyperparameters, list_amounts, print_status, previous_evaluation_index=int(hyperparameter_samples["log_likelihood"].idxmax()), **kwargs)
                
                search_direction = np.zeros(shape=mid_hyperparameter_sample.shape)
                # search along preferred hyperparameter:
                if search_hyperparameter is not None:
                    try:
                        index = non_constant_hyperparameters.index(search_hyperparameter)
                    except ValueError: raise ValueError(f"The preferred hyperparameter along which to search, {search_hyperparameter} is not a hyperparameter that is being adjusted.")
                    search_direction[index] = max(np.abs(mid_hyperparameter_sample[index]*search_distance), minimum_search)
                else: # search along the n-th hyperparameter:
                    search_direction[n] = max(np.abs(mid_hyperparameter_sample[n]*search_distance), minimum_search)
                if print_status: print(f"searching along hyperparameter \"{non_constant_hyperparameters[n]}\" with initial magnitude {search_direction[n]}.")
        
                # evaluate hyperparameter samples in the search_direction until log-likelihood no longer increasing
                left_likelihood = minimum_log_likelihood
                count = 0
                scale = 0
                left_hyperparameter_sample = mid_hyperparameter_sample + search_direction
                left_likelihood = _evaluate_likelihood(cv_function, left_hyperparameter_sample, [hyperparameter_samples], csv_filename, hyperparameter_names, constant_hyperparameters, integer_hyperparameters, non_constant_hyperparameters, list_hyperparameters, list_amounts, print_status, **kwargs)
                right_likelihood = minimum_log_likelihood
                while True:
                    count+=1
                    # if left is better than mid, step left again
                    if left_likelihood >= mid_likelihood:
                        right_likelihood, right_hyperparameter_sample = mid_likelihood, mid_hyperparameter_sample
                        mid_likelihood, mid_hyperparameter_sample = left_likelihood, left_hyperparameter_sample
                        left_hyperparameter_sample = mid_hyperparameter_sample + (mid_hyperparameter_sample-right_hyperparameter_sample)*search_base
                        left_likelihood = _evaluate_likelihood(cv_function, left_hyperparameter_sample, [hyperparameter_samples], csv_filename, hyperparameter_names, constant_hyperparameters, integer_hyperparameters, non_constant_hyperparameters, list_hyperparameters, list_amounts, print_status, **kwargs)
                        # upon first decrease in left, if feasible, break.
                        if left_likelihood <= mid_likelihood and left_likelihood > minimum_log_likelihood: break
                        else: continue
                    # if left is not feasible, step right.
                    if left_likelihood <= minimum_log_likelihood:
                        scale+=1
                        left_hyperparameter_sample = (left_hyperparameter_sample + (2**scale-1)*mid_hyperparameter_sample)/2**scale
                        left_likelihood = _evaluate_likelihood(cv_function, left_hyperparameter_sample, [hyperparameter_samples], csv_filename, hyperparameter_names, constant_hyperparameters, integer_hyperparameters, non_constant_hyperparameters, list_hyperparameters, list_amounts, print_status, **kwargs)
                        # upon first feasible solution, break
                        if left_likelihood > minimum_log_likelihood: break
                        else: continue
                    else: break
                
                # do the same in right direction if right direction is still less than min -> left is worse than mid (mid has not changed)
                if right_likelihood <= minimum_log_likelihood:
                    right_hyperparameter_sample = mid_hyperparameter_sample - search_direction
                    right_likelihood = _evaluate_likelihood(cv_function, right_hyperparameter_sample, [hyperparameter_samples], csv_filename, hyperparameter_names, constant_hyperparameters, integer_hyperparameters, non_constant_hyperparameters, list_hyperparameters, list_amounts, print_status, **kwargs)
                    scale = 0
                    count+=1
                    while True:
                        count+=1
                        if right_likelihood >= mid_likelihood:
                            left_likelihood, left_hyperparameter_sample = mid_likelihood, mid_hyperparameter_sample
                            mid_likelihood, mid_hyperparameter_sample = right_likelihood, right_hyperparameter_sample
                            right_hyperparameter_sample = mid_hyperparameter_sample + (mid_hyperparameter_sample-left_hyperparameter_sample)*search_base
                            right_likelihood = _evaluate_likelihood(cv_function, right_hyperparameter_sample, [hyperparameter_samples], csv_filename, hyperparameter_names, constant_hyperparameters, integer_hyperparameters, non_constant_hyperparameters, list_hyperparameters, list_amounts, print_status, **kwargs)
                            # upon first decrease in right, if feasible, break.
                            if right_likelihood <= mid_likelihood and right_likelihood > minimum_log_likelihood: break
                            else: continue
                        if right_likelihood <= minimum_log_likelihood: 
                            scale+=1
                            right_hyperparameter_sample = ((2**scale-1)*mid_hyperparameter_sample+right_hyperparameter_sample)/2**scale
                            right_likelihood = _evaluate_likelihood(cv_function, right_hyperparameter_sample, [hyperparameter_samples], csv_filename, hyperparameter_names, constant_hyperparameters, integer_hyperparameters, non_constant_hyperparameters, list_hyperparameters, list_amounts, print_status, **kwargs)
                            if right_likelihood > minimum_log_likelihood: break
                            else: continue
                        else: break
                # make sure that left and mid are not the same
                if np.sum(np.abs(left_hyperparameter_sample)) == np.sum(np.abs(mid_hyperparameter_sample)) or np.sum(np.abs(right_hyperparameter_sample)) == np.sum(np.abs(mid_hyperparameter_sample)): 
                    mid_hyperparameter_sample = (left_hyperparameter_sample+right_hyperparameter_sample)/2
                    mid_likelihood = _evaluate_likelihood(cv_function, mid_hyperparameter_sample, [hyperparameter_samples], csv_filename, hyperparameter_names, constant_hyperparameters, integer_hyperparameters, non_constant_hyperparameters, list_hyperparameters, list_amounts, print_status, **kwargs)
                    count+=1
                    
                # interpolate by searching between the mode and the sets next to it on either side. Repeat this until the likelihood decreases or until the rate of change of log-likelihood wrt hyperparameters is expected to decrease on BOTH sides of the mode.
                outer_left_likelihood, outer_left_hyperparameter_sample = left_likelihood, left_hyperparameter_sample
                outer_right_likelihood, outer_right_hyperparameter_sample = right_likelihood, right_hyperparameter_sample
                
                left_hyperparameter_sample = (left_hyperparameter_sample + mid_hyperparameter_sample)/2
                left_likelihood = _evaluate_likelihood(cv_function, left_hyperparameter_sample, [hyperparameter_samples], csv_filename, hyperparameter_names, constant_hyperparameters, integer_hyperparameters, non_constant_hyperparameters, list_hyperparameters, list_amounts, print_status, **kwargs)
                
                right_hyperparameter_sample = (right_hyperparameter_sample+mid_hyperparameter_sample)/2
                right_likelihood = _evaluate_likelihood(cv_function, right_hyperparameter_sample, [hyperparameter_samples], csv_filename, hyperparameter_names, constant_hyperparameters, integer_hyperparameters, non_constant_hyperparameters, list_hyperparameters, list_amounts, print_status, **kwargs)
                count+=2 
                        
                no_left, no_right = False, False # keep track of convergence on each side
                while True: # loop
                    if left_likelihood < outer_left_likelihood: no_left = True
                    if left_likelihood + (left_likelihood-outer_left_likelihood)/np.linalg.norm(left_hyperparameter_sample-outer_left_hyperparameter_sample)*np.linalg.norm(mid_hyperparameter_sample-left_hyperparameter_sample) > mid_likelihood: no_left=True
                    if right_likelihood < outer_right_likelihood: no_right = True
                    if right_likelihood + (right_likelihood-outer_right_likelihood)/np.linalg.norm(right_hyperparameter_sample-outer_right_hyperparameter_sample)*np.linalg.norm(mid_hyperparameter_sample-right_hyperparameter_sample) > mid_likelihood: no_right=True
                    if no_left and no_right: 
                        if print_status: print(f"HYPERPARAMETER SEARCH IN THE FOLLOWING DIRECTION CONVERGED AFTER {count} STEPS: " + str(search_direction))
                        break
                    
                    # set on the left side is most-likely, so it becomes mid and we search around it
                    if left_likelihood >= right_likelihood and left_likelihood >= mid_likelihood:
                        no_left, no_right = False, False
                        outer_right_likelihood, outer_right_hyperparameter_sample = mid_likelihood, mid_hyperparameter_sample
                        mid_likelihood, mid_hyperparameter_sample = left_likelihood, left_hyperparameter_sample
                        
                        right_hyperparameter_sample = (mid_hyperparameter_sample+outer_right_hyperparameter_sample)/2
                        right_likelihood = _evaluate_likelihood(cv_function, right_hyperparameter_sample, [hyperparameter_samples], csv_filename, hyperparameter_names, constant_hyperparameters, integer_hyperparameters, non_constant_hyperparameters, list_hyperparameters, list_amounts, print_status, **kwargs)
                        
                        left_hyperparameter_sample = (mid_hyperparameter_sample+outer_left_hyperparameter_sample)/2
                        left_likelihood = _evaluate_likelihood(cv_function, left_hyperparameter_sample, [hyperparameter_samples], csv_filename, hyperparameter_names, constant_hyperparameters, integer_hyperparameters, non_constant_hyperparameters, list_hyperparameters, list_amounts, print_status, **kwargs)
                        count+=2
                    # set on the right side is most-likely, so it becomes mid and we search around it
                    elif right_likelihood > left_likelihood and right_likelihood >= mid_likelihood:
                        no_left, no_right = False, False
                        outer_left_likelihood, outer_left_hyperparameter_sample = mid_likelihood, mid_hyperparameter_sample
                        mid_likelihood, mid_hyperparameter_sample = right_likelihood, right_hyperparameter_sample
                        
                        left_hyperparameter_sample = (mid_hyperparameter_sample+outer_left_hyperparameter_sample)/2
                        left_likelihood = _evaluate_likelihood(cv_function, left_hyperparameter_sample, [hyperparameter_samples], csv_filename, hyperparameter_names, constant_hyperparameters, integer_hyperparameters, non_constant_hyperparameters, list_hyperparameters, list_amounts, print_status, **kwargs)
                        
                        right_hyperparameter_sample = (mid_hyperparameter_sample+outer_right_hyperparameter_sample)/2
                        right_likelihood = _evaluate_likelihood(cv_function, right_hyperparameter_sample, [hyperparameter_samples], csv_filename, hyperparameter_names, constant_hyperparameters, integer_hyperparameters, non_constant_hyperparameters, list_hyperparameters, list_amounts, print_status, **kwargs)
                        count+=2
                    else: # mid stays the most-likely and we search around it again
                        if not no_left:
                            outer_left_hyperparameter_sample = left_hyperparameter_sample
                            left_hyperparameter_sample = (mid_hyperparameter_sample+left_hyperparameter_sample)/2
                            left_likelihood = _evaluate_likelihood(cv_function, left_hyperparameter_sample, [hyperparameter_samples], csv_filename, hyperparameter_names, constant_hyperparameters, integer_hyperparameters, non_constant_hyperparameters, list_hyperparameters, list_amounts, print_status, **kwargs)
                            count+=1
                        if not no_right:
                            outer_right_hyperparameter_sample = right_hyperparameter_sample
                            right_hyperparameter_sample = (mid_hyperparameter_sample+right_hyperparameter_sample)/2
                            right_likelihood = _evaluate_likelihood(cv_function, right_hyperparameter_sample, [hyperparameter_samples], csv_filename, hyperparameter_names, constant_hyperparameters, integer_hyperparameters, non_constant_hyperparameters, list_hyperparameters, list_amounts, print_status, **kwargs)
                            count+=1
                if search_hyperparameter is not None: break
    except MemoryError: raise MemoryError(f"There is not enough memory in ram to fit a kernel density estimate to the provided hyperparameter sets.")