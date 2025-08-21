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
        - 