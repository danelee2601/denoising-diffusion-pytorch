import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def rbf_kernel(x1, x2, scale = 5000, sigma = 100):
    return sigma * np.exp(-1 * ((x1-x2) ** 2) / (2*scale))

def gram_matrix(xs):
    return [[rbf_kernel(x1,x2) for x2 in xs] for x1 in xs]

"""
Function for creating conditioning data.
Input:
training_image: facies array (128x128 array)
nr_samples: nr of well realizations
disp: if true, plot the wells.
Output:
well_list: list of well arrays.
"""
def well_sample(training_image,nr_samples,disp):
    n_x = training_image.shape[0]
    x = np.arange(0,n_x)
    mu_alpha = 50
    sigma_alpha = 20
    mu_beta = 0
    sigma_beta = 0.5
    lambda_s = 4
    well_list = []
    cov = gram_matrix(x)
    for i in range(nr_samples):
        s = np.random.poisson(lambda_s)+1
        plt.figure(i)
        well_mat = np.ones(np.shape(training_image))*(-1)
        for i in range(s):
            alpha = np.random.normal(mu_alpha,sigma_alpha)
            beta = np.random.normal(mu_beta,sigma_beta)
            mu = alpha + beta*x
            residual = np.random.multivariate_normal(np.zeros(len(x)),cov)
            sample = mu + residual
            inds = np.where( (sample < 128) & (sample > 0))[0]
            #print(len(inds))
            well_mat[x[inds],sample.astype(int)[inds]] = training_image[x[inds],sample.astype(int)[inds]]
        if disp:
            plt.imshow(well_mat)
            plt.show()
        well_list.append(well_mat)

    return well_list

if __name__ == '__main__':
    """
    Example:
    """

    #Load dataset. NB!!! Your path is different.
    df_facies = pd.read_csv('../dataset/facies/facies1.csv', header=None)
    #Flip facies array.
    facies_arr = np.flip(np.flip(df_facies.to_numpy()),1)
    #Plot facies array
    plt.imshow(facies_arr)
    #set nr of realiz and get well data
    nr_samples = 10
    disp = True
    well_sample(facies_arr,nr_samples,disp)