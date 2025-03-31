# Building some simple Machine Learning algorithms from scratch


## 1. Linear Regression

The `fit` function fits the vector $\theta$ to its input parameters so that the Residual Sum of Squares 

$RSS(\theta) = \sum_{i=1}^N \left(y_i - x_i^T  \theta \right)^2 = (y - X \theta)^T (y - X  \theta)$

is minimized.

This problem can be solved analytically, with the well-known result $\theta=(X^T  X)^{-1} X^T y$, but as it is computationally-expensive to invert matrices ($ \sim \mathcal{O}(n^3)$), the gradient descent approach is preferred:

* Step 1: random initialization of $\theta_0$
* Step 2: $\theta_{k+1} = \theta_k - \lambda  \frac{\partial RSS(\theta)}{\partial \theta}$
* Step 3: continue until the termination condition is reached


## 2. Logistic Regression

Logistic Regression assumes that the logit of $p$ is a linear combination of the variables $X_1, X_2, ..., X_k$:

$$ logit(p)=\log \left(\frac{p}{1-p} \right) = \beta_0 + \beta_1 X_1 + ... + \beta_k X_k$$ 

$$\Leftrightarrow p = sigmoid(\beta_0 + \beta_1 X_1 + ... + \beta_k X_k) $$ 

We note $predictions=sigmoid(X \cdot \theta)$. The cost is computed using the formula:

$$\text{Cost} = -\frac{1}{n} \sum \left[ y \cdot \log(predictions) + (1-y) \cdot \log(1-predictions) \right]$$

Gradient descent is used to minimize the cost function.

## 3. k-Nearest Neighbors

k-Nearest Neighbors assumes that similar data points are close to each other.

Therefore the idea is to use the value of the $k$ nearest neigbors of data point $i$ to infer the target value $y_i$.

Here we use the euclidean distance:

$$d(A, B)= \sqrt{\sum_{i=1}^n \left(x_i - y_i \right)^2}$$

## 4. k-Means

k-Means is used to partition data in $k$ clusters.
The algorithm works as follows:

* Step 1: Initialize the centroids (= center of a cluster) randomly
* Step 2: Assign data points to the nearest centroid using the euclidean distance
* Step 3: Update the centroids ($n_k$ is the number of data points in the cluster $k$)

$$ C_k = \frac{1}{n_k} \sum_{i\in Cluster_k}x_i$$

* Step 4: Repeat until the centroids don't change too much anymore


## 5. Decision Tree

At each node, we choose the characteristic that best separates the dataset. Several separability criteria can be used:

* Entropy
* Gini
