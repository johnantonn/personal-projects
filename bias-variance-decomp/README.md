# bias-variance-decomp
Bias-variance decomposition for classification and regression algorithms.

![image](https://user-images.githubusercontent.com/8168416/157861049-c2fd2438-5768-41bf-a9db-0d67ae3aa2d0.png)

## Description
This repository includes two notebooks that deal with the bias-variance trade-off in the context of:
- Classification
- Regression
 
The conclusions are most obvious when comparing the bias-variance values for decision trees versus ensemble methods like random forests. However, a number of additional learning algorithms were included in this analysis to provide better insight, e.g. linear models such as linear regression display a significant amount of loss due to bias compared to variance, indicating underfitting.

## Background
The decomposition of the performance loss into its bias and variance components provides insight into learning algorithms, as these concepts are correlated to underfitting and overfitting [1]. This analysis makes use of the `mlxtend` package, which provides decomposition of the loss for two losses:
- 0-1 loss for classification
- MSE loss for regression

See references for more details.

## References
[1] [mlxtend: Bias-Variance Decomposition](https://rasbt.github.io/mlxtend/user_guide/evaluate/bias_variance_decomp/)

[2] [Pedro Domingos: A Unified Bias-Variance Decomposition and its Applications](https://homes.cs.washington.edu/~pedrod/papers/mlc00a.pdf)
