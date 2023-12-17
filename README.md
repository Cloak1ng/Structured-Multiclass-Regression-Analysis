# Structured*Multiclass*Regression*Analysis
Structured Multiclass Regression: ML analysis with logistic and softmax regression. Explore decision boundaries and cost history for insightful classification visualization
 Requirements
* Python 3.x
* NumPy
* Matplotlib
*********************************************
1. Dataset Generation:
   * Generate a structured multiclass classification dataset with three classes.

2. Visualization of Multiclass Dataset:
   * Visualize the generated multiclass dataset using a scatter plot.

3. Logistic Regression (Binary Classification):
   * Apply logistic regression to the multiclass dataset as binary classification.
   * Add a bias term to the feature matrix.
   * Initialize logistic regression parameters and set hyperparameters.
   * Define sigmoid function, logistic cost function, and gradient descent for logistic regression.
   * Perform logistic regression using gradient descent.
   * Visualize the decision boundary and plot the cost history.

4. Multiclass Dataset Generation (Again):
   * Generate another multiclass classification dataset.

5. Softmax Regression (Multiclass Classification):
   * Apply softmax regression to the new multiclass dataset.
   * One-hot encode the target variable.
   * Initialize softmax regression parameters and set hyperparameters.
   * Define softmax function, softmax cost function, and gradient descent for softmax regression.
   * Perform softmax regression using gradient descent.
   * Visualize decision boundaries using pcolormesh.

6. Visualization of Softmax Decision Boundaries:
   * Visualize decision boundaries learned by softmax regression using pcolormesh.
   * Include a scatter plot of data points.

7. Cost History for Softmax Regression:
   * Plot the cost history during softmax regression optimization.

8. Print Unique Classes:
   * Print the unique classes present in the multiclass dataset.
