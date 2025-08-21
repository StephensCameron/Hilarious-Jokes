# Hilarious-Jokes--Predictive-Model-and-Classification
This is a basic Machine-Learning analysis of hilarious-joke data that was collected in Dr. Eric Hallman's Mathematics of Scientific Computing course at NCSU. This is an extension and further analysis of these hilarious jokes.
---
## Workflow
- We first perform some data cleaning on our dataset
    - Rescale data
    - Impute missing values
- Perform exploritory data analysis
    - KMeans clustering and visual representations
    - Explore joke similarity and humor similarity
- Create predictive models
    - Split survey responses training, testing; X data (input variables), and Y data (output/predicted responses)
    - Use scikit-learn's KFold package to use cross validation to assess different model hyperparameters
    - Begin constructing Linear Regression Model:
        - First, we reduce the dimensionality of the data using Principle Component Analysis
        - Then we compare how training and validation sets compare at different dimensions using cross-validation
        - Due to the small nature of our dataset (total features ~= total survey respondants), we error on the side of reducing dimension as much as possible and select a model that projects all data onto the first two principle components
        - ![For the final model we choose to project the X data onto a two-dimensional subspace](final_model_test_results.png)
    - Begin Neural Network Model:
        - The expectation is that a 'shallow' Neural Net in combination with PCA will yield results that do not overfit the training data and still have errors in validation data that are comparable to the training errors
        - To tune hyperparameters, we use cross-validation along with a grid-search method to find the combination of hyperparameters that minimize the validation error averaged across all training-validation folds in the data
        - The optimal hyperparamters are: [n_deep_layers=2, nodes_per_layer=8, PCA_dimension=2, batch_size=8, num_epochs=50] with a relative error in the frobeneus norm of 0.467
        - the large error in this is likely due to having such a small dataset
        