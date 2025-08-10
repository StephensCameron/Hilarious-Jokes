import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from sklearn.cluster import KMeans
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold


def main():
    """
    Main function to execute script. Basic workflow is:
    1. Load the dataset
    2. Split the dataset into training, validation, and test sets
    3. Plot explained variance against the number of components to gain insights on structure of data and optimal number of clusters.
    4. Visualize Cluster Data
    5. Define Cross-Validation Routine
        5a. Use Scikit-Learn's k-fold routine
    6. Define Model Training Function
        6a. Apply PCA to reduce dimensionality
        6b. Solve for the optimal model using a linear least-squares model
    7. Define Model Evaluation Function
        7a. Using relative error in the Frobenius norm as our measurement of error
    8. Use a Grid-Search Method to search for the optimal hyperparameters of number of folds (n_split) and model dimension (k)
    9. Train the final model using the optimal hyperparameters and evaluate it on the test set
    10. Save the model for future use
    11. Draw conclusions based on the results
    
    """
    # Load and split the data
    joke_dataset, train_dataset_full, train_dataset, val_dataset, test_dataset = load_and_split_data()

    # Print the shapes of the datasets
    print(f"Full dataset shape: {joke_dataset.shape}")
    print(f"Training dataset shape: {train_dataset_full.shape}")
    print(f"Training set shape: {train_dataset.shape}")
    print(f"Validation set shape: {val_dataset.shape}")
    print(f"Test set shape: {test_dataset.shape}")

    # Plot explained variance vs number of components
    plot_explained_variance_vs_components(train_dataset,'explained_variance_vs_components_partial.png')
    """If we apply the elbow method, we can see that the optimal number of clusters is around 4 or 5.
    One interpretation of this fact is that we have either 4 or 5 different types of jokes in the dataset."""

    plot_explained_variance_vs_components(train_dataset_full,'explained_variance_vs_components_full.png')

    plot_explained_variance_vs_components(train_dataset_full.T,'explained_variance_vs_components_full_transposed.png')
    """When we apply the same method to the transposed dataset, we are actually investigating the people who rated the jokes more than the jokes.
    From this plot, we could interpre that there are 6 or 7 different 'senses of humor' in the dataset."""

    # here I will add clustering visualizations


    ####### Predictive Model Section #######

    """
    We will first look at a linear model for predicting a joke rater's response to the last 26 jokes, based on the first 16
    This next section will combine steps 5, 6, and 7.
    """
    # Select indices for training and complementary datasets
    # We will select 16 jokes for the input (X) and the remaining jokes for the output (Y)
    selected_indices, complement_indices, JTrain_x, JTrain_y, JTest_x, JTest_y = select_train_complement_indices(
        joke_dataset, train_dataset_full, train_dataset, val_dataset, test_dataset, cross_validation=True)

    # now we can grid search for the optimal hyperparameters
    n_splits = [3, 5, 10]  # Different numbers of folds for cross-validation
    k_values = range(1, 16)  # Different numbers of components to try
    results_df = grid_search_hyperparams(k_values, n_splits, train_dataset_full, selected_indices,
            complement_indices, 'Average_Training_and_Validation_Error_vs_k.png')
    
    optimal_hyperparams = results_df.loc[results_df['Avg_Val_Error'].idxmin()]
    print(f"Optimal Hyperparameters: k={optimal_hyperparams['k']}, n_splits={optimal_hyperparams['n_splits']}, "
          f"Avg Train Error={optimal_hyperparams['Avg_Train_Error']}, Avg Val Error={optimal_hyperparams['Avg_Val_Error']}")
    
    # Optimal Hyperparameters: k=5.0, n_splits=3.0, Avg Train Error=0.352568568245995, Avg Val Error=0.4238408586782927

    """We've used the grid search to find the optimal hyperparameters for our model, so now we select 
    the optimal k and n_splits values to train our final model and evaluate it on the test set.
    Note: techically n_splits is not a hyperparameter, but grid search is still applicable to compare different k value
    while examing how different n_splits values affect the temporary model performance.. (as expected, for a very small dataset, more folds tends to be better)"""

    optimal_k = int(optimal_hyperparams['k'])
    optimal_n_splits = int(optimal_hyperparams['n_splits'])
    
    model = train_model(JTrain_x, JTrain_y, optimal_k)
    test_relative_error = evaluate_model(model, JTest_x, JTest_y)
    train_relative_error = evaluate_model(model, JTrain_x, JTrain_y)
    test_val_relative_difference = (test_relative_error - optimal_hyperparams['Avg_Val_Error']) / optimal_hyperparams['Avg_Val_Error']

    print(f"Test Relative Error: {test_relative_error}")
    print(f"Training Relative Error: {train_relative_error}")
    print(f"The model performed: {test_val_relative_difference*100:.2f} % worse on the test set than on the validation set.")

    #Test Relative Error: 0.4357707780985749
    #Training Relative Error: 0.3663375252693526
    #The Model Performed: 2.81 % worse on the test set than on the validation set.

    plot_grid_with_final_model_results(results_df, optimal_hyperparams, test_relative_error, 'final_model_test_results.png')

    """We expect the test error to be higher than the validation error, as the model has not seen the test data during training,
    though, I am surprised that the test error is only slightly higher than the validation error, it means that the model is generalizing well to unseen data.
    This is a good sign that the model is not overfitting to the training data, and that the data collected is pretty good """

    # Save the model for future use
    np.save('linear_model_by_crossvalidation.npy', model)




# Loads the dataset and data preprocessing section
def load_and_split_data():
    """
    Loads the joke dataset and splits it into training, validation, and test sets.
    
    Parameters:
    None

    Returns:
    joke_dataset (pd.DataFrame): The full dataset containing jokes.
    train_dataset_full (pd.DataFrame): The full training dataset containing 80% of the data.
    train_dataset (pd.DataFrame): The training set containing 72% of the full dataset, with 8% withheld for validation/developer set.
    val_dataset (pd.DataFrame): The validation set containing 10% of train_dataset.
    test_dataset (pd.DataFrame): The test set containing the remaining 20% of the full dataset.
    """
    
    # Load the dataset
    joke_dataset = pd.read_csv('jokeRatings.csv') #full dataset

    # split the dataset into training and testing sets
    train_size = int(0.8 * len(joke_dataset))
    train_dataset_full = joke_dataset[:train_size] # Total training dataset with 80% of data

    test_dataset = joke_dataset[train_size:] # Remaining 20% of data for testing

    # Split the training dataset into validation and training sets
    val_size = int(0.1 * len(train_dataset_full))
    val_dataset = train_dataset_full[:val_size]
    train_dataset = train_dataset_full[val_size:]

    return joke_dataset.to_numpy(), train_dataset_full.to_numpy(), train_dataset.to_numpy(), val_dataset.to_numpy(), test_dataset.to_numpy()

def plot_explained_variance_vs_components(train_dataset, filename='explained_variance_vs_components.png'):
    """
    Plots the explained variance against the number of components.
    
    Parameters:
    train_dataset (pd.DataFrame or np.ndarray): The training dataset containing features.
    filename (str): The filename to save the plot.
    
    Returns:
    None
    """

    #handles either a DataFrame or a NumPy array
    X = train_dataset.values if hasattr(train_dataset, 'values') else train_dataset

    # Normalize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Range of number of clusters
    num_clusters = range(2, 21)
    f_scores = []

    # Loop through cluster counts
    for n_clusters in num_clusters:
        # Fit KMeans
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
        cluster_labels = kmeans.fit_predict(X_scaled)

        # Perform ANOVA F-test between cluster labels and features
        f_vals, _ = f_classif(X_scaled, cluster_labels)

        # Mean F-score as a proxy for explained variance
        mean_f_score = np.nanmean(f_vals)  # Use nanmean in case of NaNs
        f_scores.append(mean_f_score)

    # Plotting
    plt.figure(figsize=(6, 3))
    plt.plot(num_clusters, f_scores, marker='o')
    plt.title('Explained Variance vs Number of Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Mean ANOVA F-score')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


def select_train_complement_indices(joke_dataset, train_dataset_full, train_dataset, val_dataset, test_dataset,
                                    n_selected=16, cross_validation:bool=True):
    """
    Randomly selects indices for training and complementary datasets based on a random selection of jokes.

    Parameters:
    joke_dataset (pd.DataFrame or np.ndarray): The full dataset containing jokes.
    train_dataset_full (pd.DataFrame or np.ndarray): The full training dataset containing 80% of the data.
    train_dataset (pd.DataFrame or np.ndarray): The training set containing 72% of the full dataset, with 8% withheld for validation/developer set.
    val_dataset (pd.DataFrame or np.ndarray): The validation set containing 10% of train_dataset.
    test_dataset (pd.DataFrame or np.ndarray): The test set containing the remaining 20% of the full dataset.
    n_selected (int): Number of jokes to select for training.
    cross_validation (bool): If True, returns the full training dataset with selected and complementary indices for cross-validation.

    Returns:
    selected_indices (list): Indices of selected jokes for training.
    complement_indices (list): Indices of complementary jokes not selected for training.
    JTrain_x (np.ndarray): Training dataset features for selected jokes.
    JTrain_y (np.ndarray): Training dataset labels for complementary jokes.
    JDev_x (np.ndarray): Validation dataset features for selected jokes.
    JDev_y (np.ndarray): Validation dataset labels for complementary jokes.
    JTest_x (np.ndarray): Test dataset features for selected jokes.
    JTest_y (np.ndarray): Test dataset labels for complementary jokes.
    """
    
    import random
    # Randomly select 16 jokes from the dataset
    random.seed(42)  # For reproducibility
    selected_indices = random.sample(range(len(joke_dataset[0])), n_selected)

    # the y values are will be in the coplement of the x values out of a range from 0 to 41
    mask = ~np.isin(np.arange(0,len(joke_dataset[0])), selected_indices)
    complement_indices = np.where(mask)[0]   # Get indices where the mask is True


    if cross_validation==True:
        # If cross-validation is enabled, we will return the full training dataset with selected and complementary indices
        # because we will split the full training dataset into k-folds later
        JTrain_x = train_dataset_full[:, selected_indices]
        JTest_x = test_dataset[:, selected_indices]

        JTrain_y = train_dataset_full[:, complement_indices]
        JTest_y = test_dataset[:, complement_indices]

        return selected_indices, complement_indices, JTrain_x, JTrain_y, JTest_x, JTest_y
    else:
        # if cross-validation is false, we will return the split train, dev, and test datasets with selected and complementary indices
        JTrain_x = train_dataset[:,selected_indices]
        JDev_x = val_dataset[:,selected_indices]
        JTest_x = test_dataset[:,selected_indices]
    
        JTrain_y = train_dataset[:,complement_indices]
        JDev_y = val_dataset[:,complement_indices]
        JTest_y = test_dataset[:,complement_indices]

    return selected_indices, complement_indices, JTrain_x, JTrain_y, JDev_x, JDev_y, JTest_x, JTest_y

def train_model(x_train, y_train, k):
    """
    Create a linear model using least squares regression.
    
    Parameters:
    x_train (numpy.ndarray): Training data features.
    y_train (numpy.ndarray): Training data labels.
    k (int): Number of components for dimensionality reduction.
    
    Returns:
    numpy.ndarray: Model coefficients.
    """

    U, S, V_T = np.linalg.svd(x_train, full_matrices=False)
    U_k = U[:, :k]
    S_k = np.diag(S[:k])
    V_T_k = V_T[:k, :]
    
    x_reduced = np.dot(U_k, np.dot(S_k, V_T_k))
    x_reduced = np.hstack((x_reduced, np.ones((x_reduced.shape[0], 1))))  # Add bias term
    
    model = np.linalg.lstsq(x_reduced, y_train, rcond=None)[0]
    
    return model

def evaluate_model(model, x_data, y_data):
    """
    Evaluate the model on validation data by calculating the relative error in the Frobenius norm.
    
    Parameters:
    model (numpy.ndarray): Model coefficients.
    x_data (numpy.ndarray): Model input data.
    y_data (numpy.ndarray): Expected model output.
    
    Returns:
    float: Relative error of the model predictions.
    """

    x_val_reduced = np.column_stack((x_data, np.ones(x_data.shape[0])))
    y_data_pred = x_val_reduced @ model
    
    relative_error = np.linalg.norm(y_data - y_data_pred, 'fro') / np.linalg.norm(y_data, 'fro')
    
    return relative_error

def cross_validate_model(data, selected_indices, complement_indices, k, n_splits=5):
    """
    Perform k-fold cross-validation on the dataset.
    
    Parameters:
    data (numpy.ndarray): Full dataset.
    selected_indices (list): Indices of selected features.
    complement_indices (list): Indices of complementary features.
    k (int): Number of components for dimensionality reduction.
    n_splits (int): Number of folds for cross-validation.
    
    Returns:
    avg_model (numpy.ndarray): Averaged model parameters across all folds.
    avg_train_error (float): Average relative error on training sets across all folds.
    avg_val_error (float): Average relative error on validation sets across all folds.
    """
    kfold = KFold(n_splits=n_splits)
    
    all_relative_errors_training = []
    all_relative_errors_validation = []
    all_models = []

    for train_index, val_index in kfold.split(data):
        X_train, X_val = data[train_index], data[val_index]

        JTrain_x = X_train[:, selected_indices]
        JTrain_y = X_train[:, complement_indices]
        JDev_x = X_val[:, selected_indices]
        JDev_y = X_val[:, complement_indices]

        model = train_model(JTrain_x, JTrain_y, k)
        train_relative_error = evaluate_model(model, JTrain_x, JTrain_y)
        val_relative_error = evaluate_model(model, JDev_x, JDev_y)

        all_models.append(model)
        all_relative_errors_training.append(train_relative_error)
        all_relative_errors_validation.append(val_relative_error)

    avg_model = np.mean(all_models, axis=0)  # Average model coefficients
    avg_train_error = np.mean(all_relative_errors_training)
    avg_val_error = np.mean(all_relative_errors_validation)

    return avg_model, avg_train_error, avg_val_error

def grid_search_hyperparams(k_values, n_splits_list, train_dataset_full, selected_indices, complement_indices,
            filename='grid_search_results.png'):
    """
    Perform a grid search over hyperparameters k_values and n_splits
    
    Parameters:
    k_values (list): List of number of components to test.
    n_splits_list (list): List of number of folds for cross-validation.
    train_dataset_full (numpy.ndarray): Full training dataset.
    selected_indices (list): Indices of selected features.
    complement_indices (list): Indices of complementary features.
    
    Returns:
    results_df (pd.DataFrame): DataFrame containing the results of the grid search.
    """
    
    results = []
    for k in k_values:
        for n_splits in n_splits_list:
            avg_model, avg_train_error, avg_val_error = cross_validate_model(
                train_dataset_full, selected_indices, complement_indices, k, n_splits
            )
            results.append((k, n_splits, avg_train_error, avg_val_error))
            #print(f'k: {k}, n_splits: {n_splits}, Avg Train Error: {avg_train_error}, Avg Val Error: {avg_val_error}')

    # now we can convert the results to a pandas dataframe for easier analysis
    results_df = pd.DataFrame(results, columns=['k', 'n_splits', 'Avg_Train_Error', 'Avg_Val_Error'])
    #print(results_df)

    # we can make some plots to visualize the results by plotting the average validation and training errors for each k, separated for each n_splits value
    fig, ax = plt.subplots(figsize=(10, 4))
    for n_splits in n_splits_list:
        subset = results_df[results_df['n_splits'] == n_splits]
        ax.plot(subset['k'], subset['Avg_Train_Error'], marker='o', label=f'Train Error (n_splits={n_splits})')
        ax.plot(subset['k'], subset['Avg_Val_Error'], marker='o', linestyle='--', label=f'Val Error (n_splits={n_splits})')

    ax.set_title('Average Training and Validation Errors vs Number of Components (k)')
    ax.set_xlabel('Number of Components (k)')
    ax.set_ylabel('Relative Error')
    ax.legend(bbox_to_anchor=(1.0, 1), loc='upper left')
    #ax.legend()
    ax.grid()
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

    return results_df

def plot_grid_with_final_model_results(results_df, optimal_hyperparams, final_model_error, filename='final_model_results.png'):
    """
    Plot the grid search results with the final model results.
    
    Parameters:
    results_df (pd.DataFrame): DataFrame containing the results of the grid search.
    optimal_hyperparams (pd.DataFrame): Optimal hyperparameters (k, n_splits).
    final_model_error (float): Relative error of the final model.
    filename (str): Filename to save the plot.

    Returns:
    None
    """
    
    fig, ax = plt.subplots(figsize=(10, 4))
    for n_splits in results_df['n_splits'].unique():
        subset = results_df[results_df['n_splits'] == n_splits]
        ax.plot(subset['k'], subset['Avg_Train_Error'], marker='o', label=f'Train Error (n_splits={n_splits})')
        ax.plot(subset['k'], subset['Avg_Val_Error'], marker='o', linestyle='--', label=f'Val Error (n_splits={n_splits})')
    #ax.axhline(final_model_error, color='red', linestyle='--', label='Final Model Error')
    ax.plot(optimal_hyperparams['k'], final_model_error, marker='*', color='blue', markersize=15, label='Final Model Test Error')

    ax.set_title('Training, Validation, and Test Errors vs Number of Components (k)')
    ax.set_xlabel('Number of Components (k)')
    ax.set_ylabel('Relative Error')
    ax.legend(bbox_to_anchor=(1.0, 1), loc='upper left')
    ax.grid()
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()




main()