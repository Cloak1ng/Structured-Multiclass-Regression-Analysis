import numpy as np
import matplotlib.pyplot as plt

# Function to generate a more structured multiclass classification dataset
def generate_multiclass_classification_dataset():
    np.random.seed(42)
    num_samples_per_class = 50

    # Class 0 (Circular distribution)
    theta = np.linspace(0, 2*np.pi, num_samples_per_class)
    radius = np.random.uniform(0, 1, num_samples_per_class)
    X0 = np.c_[radius * np.cos(theta), radius * np.sin(theta)]
    y0 = np.zeros(num_samples_per_class)

    # Class 1 (Spiral distribution)
    theta = np.linspace(0, 4*np.pi, num_samples_per_class)
    radius = np.linspace(0, 1, num_samples_per_class)
    X1 = np.c_[radius * np.cos(theta), radius * np.sin(theta)]
    y1 = np.ones(num_samples_per_class)

    # Class 2 (Random distribution)
    X2 = np.random.rand(num_samples_per_class, 2)
    y2 = 2 * np.ones(num_samples_per_class)

    # Combine the classes
    X_multiclass = np.vstack([X0, X1, X2])
    y_multiclass = np.concatenate([y0, y1, y2])

    return X_multiclass, y_multiclass.astype(int)

# Visualize the multiclass classification dataset
X_multiclass, y_multiclass = generate_multiclass_classification_dataset()
plt.scatter(X_multiclass[:, 0], X_multiclass[:, 1], c=y_multiclass, cmap=plt.cm.Paired, marker='o')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Structured Multiclass Classification Dataset')
plt.show()


# Add a bias term to X
X_b = np.c_[np.ones((X.shape[0], 1)), X]

# Initialize parameters for logistic regression
theta_logistic = np.random.randn(3, 1)
# Set hyperparameters
learning_rate_logistic = 0.01
n_iterations_logistic = 1000

# Define the sigmoid function for logistic regression
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Define the logistic regression cost function
def logistic_cost_function(X, y, theta):
    m = len(y)
    predictions = sigmoid(X.dot(theta))
    cost = -1 / m * np.sum(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
    return cost

# Implement logistic regression using gradient descent
def logistic_gradient_descent(X, y, theta, learning_rate, n_iterations):
    m = len(y)
    cost_history = []

    for iteration in range(n_iterations):
        predictions = sigmoid(X.dot(theta))
        errors = predictions - y
        gradient = (1 / m) * X.T.dot(errors)
        theta = theta - learning_rate * gradient

        # Compute and store the cost for visualization
        cost = logistic_cost_function(X, y, theta)
        cost_history.append(cost)

    return theta, cost_history

# Run logistic regression using gradient descent
theta_logistic, cost_history_logistic = logistic_gradient_descent(X_b, y, theta_logistic, learning_rate_logistic, n_iterations_logistic)

# Visualize the decision boundary for logistic regression
plt.scatter(X[y == 0, 0], X[y == 0, 1], label='Class 0', marker='o')
plt.scatter(X[y == 1, 0], X[y == 1, 1], label='Class 1', marker='x')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Logistic Regression Decision Boundary')
plt.legend()

# Plot decision boundary
x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
x2_min, x2_max = X[:, 1].min(), X[:, 1].max()
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 100), np.linspace(x2_min, x2_max, 100))
X_decision_boundary = np.c_[np.ones((100*100, 1)), xx1.ravel(), xx2.ravel()]
probs = sigmoid(X_decision_boundary.dot(theta_logistic)).reshape(100, 100)
plt.contour(xx1, xx2, probs, levels=[0.5], cmap="Greys", vmin=0, vmax=1)
plt.show()

# Visualize the cost history for logistic regression
plt.plot(range(1, n_iterations_logistic + 1), cost_history_logistic, color='blue')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost History during Logistic Regression (GD)')
plt.show()

# Function to generate a multiclass classification dataset
def generate_multiclass_classification_dataset():
    np.random.seed(42)
    X = 2 * np.random.rand(100, 2)
    y = np.random.randint(0, 3, 100)
    return X, y

# Visualize the multiclass classification dataset
X_multiclass, y_multiclass = generate_multiclass_classification_dataset()
plt.scatter(X_multiclass[:, 0], X_multiclass[:, 1], c=y_multiclass, cmap=plt.cm.Paired, marker='o')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Multiclass Classification Dataset')
plt.show()

# Add a bias term to X
X_multiclass_b = np.c_[np.ones((X_multiclass.shape[0], 1)), X_multiclass]

# One-hot encode the target variable for softmax regression
def one_hot_encode(y, num_classes):
    one_hot = np.zeros((len(y), num_classes))
    one_hot[np.arange(len(y)), y] = 1
    return one_hot

num_classes = len(np.unique(y_multiclass))
y_multiclass_one_hot = one_hot_encode(y_multiclass, num_classes)

# Initialize parameters for softmax regression
theta_softmax = np.random.randn(X_multiclass_b.shape[1], num_classes)

# Set hyperparameters
learning_rate_softmax = 0.01
n_iterations_softmax = 1000

# Define the softmax function for softmax regression
def softmax(z):
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# Define the softmax regression cost function
def softmax_cost_function(X, y, theta):
    m = len(y)
    predictions = softmax(X.dot(theta))
    cost = -1 / m * np.sum(y * np.log(predictions))
    return cost

# Implement softmax regression using gradient descent
def softmax_gradient_descent(X, y, theta, learning_rate, n_iterations):
    m = len(y)
    cost_history = []

    for iteration in range(n_iterations):
        predictions = softmax(X.dot(theta))
        errors = predictions - y
        gradient = (1 / m) * X.T.dot(errors)
        theta = theta - learning_rate * gradient

        # Compute and store the cost for visualization
        cost = softmax_cost_function(X, y, theta)
        cost_history.append(cost)

    return theta, cost_history

# Run softmax regression using gradient descent
theta_softmax, cost_history_softmax = softmax_gradient_descent(X_multiclass_b, y_multiclass_one_hot, theta_softmax, learning_rate_softmax, n_iterations_softmax)

# Visualize the decision boundaries for softmax regression using pcolormesh
x1_min, x1_max = X_multiclass[:, 0].min(), X_multiclass[:, 0].max()
x2_min, x2_max = X_multiclass[:, 1].min(), X_multiclass[:, 1].max()
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 100), np.linspace(x2_min, x2_max, 100))
X_decision_boundary_softmax = np.c_[np.ones((100*100, 1)), xx1.ravel(), xx2.ravel()]
probs_softmax = softmax(X_decision_boundary_softmax.dot(theta_softmax))
y_pred_softmax = np.argmax(probs_softmax, axis=1)
plt.pcolormesh(xx1, xx2, y_pred_softmax.reshape(100, 100), cmap=plt.cm.Paired, alpha=0.8)

# Scatter plot of the points
# Scatter plot of the points
plt.scatter(X_multiclass[:, 0], X_multiclass[:, 1], c=y_multiclass, cmap=plt.cm.Paired, marker='o', edgecolor='k')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Scatter Plot of Data Points')
plt.show()


plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Softmax Regression Decision Boundaries')
plt.show()

# Visualize the cost history for softmax regression
plt.plot(range(1, n_iterations_softmax + 1), cost_history_softmax, color='blue')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost History during Softmax Regression (GD)')
plt.show()
print(np.unique(y_multiclass))
