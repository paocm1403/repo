
Subset selection (SS) is a brute force method, 
which consists in exploring all the possible models of k predictors out of p, 
k = 0, ..., p, selecting the model with the largest R² for each value of k, 
and then comparing the goodness of fit of the sequence of winning models, 
M₀, M₁, ..., Mₚ, via information criteria or CV methods.
SS is computationally explosive, as it costs the estimation of 2ᵖ models.

Step-wise selection methods are forward, backward, hybrid.

Forward selection (FS) 
starts from M₀, and for k = 0, 1, ..., p 
it involves the determination of the best model with k + 1 predictors, 
as the model obtained by adding the predictor producing the largest increase in the R². 
In the end, the goodness of fit of the sequence of winning models, M₀, M₁, ..., Mₚ,
is evaluated via information criteria or cross-validation methods to determine the best model.

Backward selection (BS) 
starts from Mₚ, and for k = p, p−1, ..., 1 
it involves the determination of the best model with k − 1 predictors, 
by removing sequentially the predictor producing the largest decrease in the R². 
In the end, the goodness of fit of the sequence of winning models, Mₚ, Mₚ₋₁, ..., M₀,
is evaluated via information criteria or CV methods to determine the best model.

FS and BS cost p(p+1)/2 model estimations. 
There is therefore a strong computational advantage compared to SS. 
However, FS and BS only explore a tiny portion of the possible models, 
thus returning just a conditional optimum. 
A relevant drawback of BS is that it is impossible to estimate the full model whenever p ≥ n. 
A relevant drawback of FS is that the first steps systematically estimate models suffering from omitted variable bias. 
Importantly, both approaches take irreversible decisions: !!!
  an added or a removed predictor will never be considered to exit or enter the model again at later stages of the procedures.

Hybrid approach starts as a forward selection, but then, from step 2, 
it considers both predictor addition and predictor removal, 
choosing on the basis of the largest increase or decrease (in absolute value) in the R². 
This approach saves the same computational advantages of FS and BS compared to SS, 
while considering the possibility to reverse wrong decisions in terms of predictor addition or removal at later stages.


## RIDGE and LASSO
OLS linear regression coefficients suffer from excessive coefficient variance 
whenever the dimension p is large compared to the sample size n, because
V(β̂₍OLS₎) = (X′X)⁻¹σ²,
where X is the n × (p + 1) design matrix and σ² is the residual variance. 
When p ≥ n,   β̂₍OLS₎ = (X′X)⁻¹X′y   is not even feasible. 
For this reason, the need arises to penalize the coefficient vector norm, in order to control its magnitude.
Ridge regression involves minimizing the following penalized least squares objective:
  ∑ᵢ₌₁ⁿ ( yᵢ − β₀ − ∑ⱼ₌₁ᵖ βⱼ xᵢⱼ )² + λ ∑ⱼ₌₁ᵖ β²ⱼ

while LASSO regression (Least Absolute Shrinkage and Selection Operator) 
involves minimizing the following alternative penalized least squares objective:
  ∑ᵢ₌₁ⁿ ( yᵢ − β₀ − ∑ⱼ₌₁ᵖ βⱼ xᵢⱼ )² + λ ∑ⱼ₌₁ᵖ |βⱼ|.

Ridge regression problem does have a closed form solution, which is equal to
β̂₍RIDGE₎ = (X′X + λIₚ)⁻¹X′y.

LASSO regression coefficient estimate β̂₍LASSO₎ instead requires iterative soft-thresholding algorithms
to be computed. This occurs because ridge penalty ∑ⱼ₌₁ᵖ βⱼ² is differentiable, while LASSO penalty is not.

From the mathematical point of view, 
Ridge and LASSO objective functions are equivalent to allowing for a maximum budget s_λ 
for ∑ⱼ₌₁ᵖ βⱼ² (ridge) and for ∑ⱼ₌₁ᵖ |βⱼ| (LASSO).
Since ridge penalty is a circle, while LASSO penalty is a diamond, 
the chance that the solution minimizing the RSS, i.e.
RSS = ∑ᵢ₌₁ⁿ ( yᵢ − β₀ − ∑ⱼ₌₁ᵖ βⱼ xᵢⱼ )²,
intersects the penalty at the axes is high for LASSO and null for RIDGE.

Parameter λ is the bias-variance trade-off parameter 
which controls the degree of shrinkage of estimated coefficients. 
When λ → 0, the penalized coefficient estimates tend to the OLS estimates. 
When λ → ∞, the penalized estimates tend to zero. 
Both ridge and LASSO are not scale-equivariant: 
  typically, predictors are standardized before estimation. 
The optimal value of λ is typically selected by CV.

The main difference between Ridge and LASSO regression 
lies in the fact that LASSO regression provides variable selection, 
i.e. a sparse regression coefficient vector, while ridge regression does not. 
Instead, ridge regression controls the effect of 
small eigenvalues of the covariance matrix X′X / n onto the regression coefficient estimates.

Denoting the singular value decomposition of X as U D V′, 
and consequently the spectral decomposition of X′X as V D² V′, we get that
Xβ̂₍RIDGE₎ = (X′X + λIₚ)⁻¹X′y = ∑ⱼ₌₁ᵖ uⱼ ( dⱼ² / ( dⱼ² + λ ) ) uⱼ′ y,
while
Xβ̂₍OLS₎ = (X′X)⁻¹X′y = ∑ⱼ₌₁ᵖ uⱼ uⱼ′ y.

The fraction d²ⱼ / (d²ⱼ+λ) is called shrinkage factor, 
and shrinks down the effect of small eigenvalues, 
due to the numerical instability arising when p is large compared to n. 
Ridge regression presents an analogy with Principal Component Regression (PCR), 
in the sense that PCR imposes a jump function to the shrinkage factor: 1 for j ≤ M, 0 for j > M, 
where M is the number of retained PCs.

There is no prescribed hierarchy in terms of performance between Ridge and LASSO regression. 
Ridge tends to outperform LASSO when the true linear model involves many p with intermediately sized coeffs, 
while LASSO prevails when there are few really strong coeffs.

Under the limit case of X unit-diagonal with size n = p, it holds
β̂₍OLS,ⱼ₎ = yⱼ,
β̂₍RIDGE,ⱼ₎ = yⱼ/(1 + λ),
β̂₍LASSO,ⱼ₎ =
  
  yⱼ − λ/2     yⱼ > λ/2
⎨ yⱼ + λ/2     yⱼ < −λ/2
⎩ 0           |yⱼ| ≤ λ/2

########## Mean Smoother with fixed h VS with fixed k
- Fixed bandwidth h (running mean)
For a target point x₀, we define the local neighborhood [x₀ − h , x₀ + h]
The running mean smoother is
ŷ₀ = (∑ᵢ₌₁ⁿ yᵢI(x₀ − h ≤ xᵢ ≤ x₀ + h))/ (∑ᵢ₌₁ⁿ I(x₀ − h ≤ xᵢ ≤ x₀ + h))

- Fixed number of neighbors k (k-NN running mean)
Let dₖ(x₀) be the Euclidean distance from x₀ to its k-th nearest sample point. Replace h by dₖ(x₀):
ŷ₀ = ( ∑ᵢ₌₁ⁿ yᵢ I(x₀ − dₖ(x₀) ≤ xᵢ ≤ x₀ + dₖ(x₀)) )/ ( ∑ᵢ₌₁ⁿ I(x₀ − dₖ(x₀) ≤ xᵢ ≤ x₀ + dₖ(x₀)) ).

Difference
Fixed h: the window width is constant, so the number of points inside the window varies with local data density.
Fixed k: the number of points is constant, so the effective bandwidth dₖ(x₀) varies across x₀ (larger in sparse regions, smaller in dense regions).

### Kernel(Nadaraya–Watson) Smoother and the role of the kernel
Kernel smoother writes the estimate as a weighted local mean:
  ŷ₀ = (∑ᵢ₌₁ⁿ yᵢ K( (xᵢ − x₀)/ h ))/( ∑ᵢ₌₁ⁿ K( (xᵢ − x₀) / h ))
Role of K (kernel)
The kernel K assigns larger weights to observations with xᵢ close to x₀ and smaller weights to more distant observations, reducing sensitivity to extreme values within the local neighborhood.
The uniform (rectangular) kernel corresponds to equal weights inside the window and zero weight outside, i.e. a “box” kernel.

### Line Smoother and  LO(W)ESS as weighted least squares
Running line smoother (local linear regression)
Approximate locally around x₀ with a linear model:  y ≈ β₀ + β₁ (x − x₀)

Estimate (β̂₀ , β̂₁) by weighted least squares:
  (β̂₀ , β̂₁) = arg min over β₀,β₁ = ∑ᵢ₌₁ⁿ [yᵢ− (β₀ + β₁ (xᵢ − x₀))]² wᵢ

with weights:  wᵢ = K((xᵢ − x₀)/ h)

The fitted value at x₀ is:   ŷ₀ = β̂₀

In matrix form:    β̂₍WLS₎ = (X′ W X)⁻¹ X′ W y      where W is the diagonal matrix of kernel weights.

LO(W)ESS (locally weighted scatterplot smoother)
LOESS uses the same weighted least squares idea, but typically in a k-NN (variable bandwidth) form: 
that is, the bandwidth h varies across x₀ (for example h = dₖ(x₀)).
It commonly uses the tricube kernel. 
The parameter “span” controls the fraction of points used locally, 
while “degree” controls the local polynomial order.

### Bandwidth h and the bias–variance trade-off. Two methods to choose optimal h
Bias–variance trade-off
- As h decreases, the estimator becomes more local, variance increases; 
in the limit h → 0, ŷ₀ = y₀ (overfitting)
- As h increases, the estimator averages more points, bias increases; 
in the limit h → ∞, ŷ₀ = ȳ (underfitting)

Two practical methods to select h
• An AIC-type criterion AIC(h), based on σ̂² and tr(Wₕ).
• Cross-validation CV(h), including K-fold and LOOCV variants.

### Interpreting ksmooth / locpoly / loess patterns with respect to h, k, or d

Increasing h (or k) produces smoother fitted curves but less informative fits; 
Decreasing h (or k) produces more jagged curves and may induce overfitting.

• ksmooth: larger bandwidth h gives smoother kernel means; smaller h yields more jagged fits. 
For the same h, the rectangular (“box”) kernel tends to be more jagged than the normal kernel 
because it assigns equal weights to all included points.
• locpoly: for fixed h, a larger polynomial degree d (1, 2, 3) increases local flexibility 
and may introduce more oscillations; h remains the main smoothness control.
• loess / loess.smooth: the “span” parameter is the k-NN analogue (fraction of points used locally). 
Larger span gives smoother fits; smaller span yields more local and jagged fits. 
The “degree” parameter controls the local polynomial order (0 running mean, 1 running line, etc.).

### Multivariate nonparametric regression vs GAM vs multiple linear regression

Multiple linear regression:   Y = β₀ + ∑ⱼ₌₁ᵖ βⱼ Xⱼ + ϵ
It is highly interpretable but has low flexibility due to the linearity restriction.

Multivariate nonparametric regression:   Y = f(X₁ , … , Xₚ) + ϵ
It is very flexible, but hard to interpret, computationally demanding, and suffers from the curse of dimensionality as p increases.

Generalized Additive Model (GAM):  g(Y) = f₁(X₁) + f₂(X₂) + … + fₚ(Xₚ)
GAMs are more flexible than linear regression (nonlinear fⱼ) 
and more interpretable than fully multivariate nonparametric regression, 
since they provide p separate univariate effect plots.

### Advantage and disadvantage of GAM vs multivariate nonparametric regression; form and constraints
--> Advantage of GAM
GAMs mitigate the curse of dimensionality by decomposing the regression function 
into additive univariate smoothers, improving interpretability and feasibility for moderate sample sizes.
--> Disadvantage of GAM
The additive structure excludes interactions among regressors unless explicitly modeled, 
making GAMs less flexible than fully multivariate nonparametric models.

Form and constraints
Each fⱼ is estimated via a univariate smoother and combined additively. 
For identifiability, the following constraint is imposed:
  ∑ᵢ₌₁ⁿ fⱼ(xᵢⱼ) = 0, for j = 1, …, p

### R commands: key parameters, fitting, plots, GAM syntax and tests
• ksmooth: kernel ∈ {“box”, “normal”}, bandwidth = h.
• loess.smooth: span (fraction of points used), degree ∈ {0, 1, 2, 3}, family ∈ {“gaussian”, “symmetric” (Tukey)}.
• locpoly: bandwidth = h, degree ∈ {1, 2, 3}.
• Multivariate loess: fit loess(y ~ x₁ + x₂, span = …, degree = …), predict on a grid, visualize using persp (3D surface) and contour plots.
• GAM (mgcv::gam): model y ~ s(x₁) + s(x₂) + …; estimation via penalized maximum likelihood with smoothing parameters selected by GCV.
• gam package: parametric part estimated by WLS, nonparametric part by backfitting; allows s(·), lo(·), and poly(·).
• Testing nonlinear effects: compare a GAM with smooth terms to a model with corresponding linear terms using anova to assess the significance of nonlinear components.


######## CLUSTER ANALYSIS
# Plot
# Selected best number of clusters for a k-means output from the Plot ?
 Scree Plot --> Elbow point

# Gap Statistic
# Selected best number of clusters for a k-means output from via the Gap Statistic?
# Which estimation options mostly impact on the results?
  
Gap statistic computes the gap between the 
logarithm of the overall deviance within K clusters in the kmeans partition 
with K clusters and the logarithm of the overall deviance within K clusters 
computed on a random un-clustered dataset. 

The gap statistics is estimated by a bootstrap approach, which also allows to obtain standard deviation estimates.
The results show that K = .... is the most likely value. The most impactful estimation option is the number
of multiple starting points, which determines the solution stability, while the number of bootstrap samples is here less relevant.

### Average silhouette width (ASW), the Pearson-Gamma coefficient, and the Calinski-Harabasz index 
### for the k-means output with K = 2, . . . , 8. 
### Define the three criteria and state which value of K you would select.

The ASW is a goodness indicator of a clustering partition. 
Considering a single point i = 1, …, n, belonging to cluster Cᵢ.
We define the average distance from the points in cluster Cᵢ as
a(i) = (1/(|Cᵢ| − 1)) ∑ⱼ∈ Cᵢ,ⱼ≠ᵢ d(i, j)

and the minimum average distance from all the points in another cluster as
b(i) = min over J ≠ I--> (1 /|Cⱼ|) ∑ⱼ∈Cⱼ d(i, j)

The ASW is defined:   s(i) = ( b(i) − a(i) ) / max( a(i), b(i) ) =

⎧ 1 − a(i)/ b(i)    if a(i) > b(i)
⎨ 0                 if a(i) = b(i)
⎩ a(i)/ b(i) − 1    if a(i) < b(i)

It takes values between −1 (absence of separation) and 1 (perfect separation)

The Pearson–Gamma coeff is defined as the correlation coeff between:
• a vector of length n(n − 1) / 2 containing the distances between all pairs of points, and
• a vector of the same length taking value 0 if the two points belong to the same cluster, and 1 otherwise.
The Pearson–Gamma coeff also lies between −1 and 1

The Calinski–Harabasz index CH is defined as
CH = [ BCSS / (k − 1)] / [ WCSS / (n − k)]
where the between-cluster sum of squares is
BCSS = ∑ᵢ₌₁ᵏ nᵢ ‖cᵢ − c‖²
and the within-cluster sum of squares is
WCSS = ∑ᵢ₌₁ᵏ ∑ₓ∈ Cᵢ ‖x − cᵢ‖²

Under the normality assumption, the CH statistic follows an F distribution with (k − 1, n − k) degrees of freedom.

--> choose the point where the three criteria attain their maximum values!
  
### Hierachical clustering with Complete vs Average vs Single linkage method ?
### Merge and the height outputs for complete clustering ?
  
Hierarchical clustering starts from the units separately considered, and then at each step 
it aggregates the two clusters with the minimum distance, 
measured as the maximum distance between two units belonging to 
- different clusters (complete linkage), 
- or, the minimum distance (single linkage)
- or, the average distance (average linkage)
The merge output illustrates which units, cluster and unit, or clusters have been aggregated
at each step (e.g., at step 33 the cluster formed at step 1 and unit 21 are merged). 
The height output contains the aggregating distance at each step. 
It can be used to select the number of clusters, by searching for the maximum difference in the aggregating distance


### Confusion matrices. Comment about the three methods’ performance
### and the relative cluster shapes. Then, comment the respective adjusted rand indices. 
### Which would be the effect of data standardization?

Average linkage is doing perfectly, complete linkage is doing OK, single linkage is doing terribly. 
This is a typical pattern when data are spherical, 
Because single linkage tends to create elongated and heterogeneous clusters, due to the chaining effect, 
unlike the other two methods, which tend to obtain compact and spherical clusters.
The adjusted rand indices are the total correct classification rates, 
adjusted to obtain 0 in the case of a perfectly random partition. 
The value is 1 for average linkage, 0 for single linkage, and 0.77 (quite high) for complete linkage.
Under spherical data, standardization typically improves the complete linkage method performance, 
because it renders the maximum distance more reliable

###
The most immediate approach to high-dim clustering is the tandem approach
That is, extracting PCs (selecting the number by usual criteria) 
and then applying hierarchical or partitioning cluster methods to the pc scores. 
In principles, clustering is an unsupervised technique,
so that there is no supervisor to verify if the obtained partition is reasonable. 
However, internal validation measures like Average Silhouette Width, Pearson gamma correlation, and Calinski-Harabasz index 
allow to evaluate the clustering partition in absence of a respective target label. 
Sometimes, the obtained cluster labels can be compared to any external politomic variable, 
to verify if the obtained clustering is able to summarize it properly. 
A risk of the tandem approach is that principal components are not the latent directions where the objects are mostly clustered. 
A safe alternative in this respect may be reduced or factorial k-means.


#################
### CART
CART (Classification and Regression Trees) is a sequential optimization procedure, 
which for J = 2, 3, …, n minimizes the overall within deviance
∑ⱼ₌₁ᴶ ∑ᵢ: xᵢ ∈ Rⱼ (yᵢ − cⱼ)²
This optimization problem is NP-hard, i.e. a global optimum cannot be computed in polynomial time. 
Therefore, we first fix the number of intervals J, and then we determine the regions Rⱼ by identifying, 
for each variable, the optimal cut-off minimizing (1), where
ĉⱼ = (1 / nⱼ) ∑ᵢ: xᵢ ∈ Rⱼ yᵢ,
which is the conditional mean of yᵢ within the interval Rⱼ, 
due to the least squares property of the mean. 
The CART procedure is naturally arrested when J = n, 
or when all the xᵢ or yᵢ are equal within all the intervals.
Manual stop criteria may involve the minimum number of obs within a leaf, 
or the % of explained deviance with respect to the root node.

### Tree Pruning
Once the CART procedure stops, the maximal tree Tₘₐₓ is obtained. 
At this stage, we need to determine a posteriori the optimal number of leaves. 
This is done by minimizing, for each J = |Tₘₐₓ|, …, 1, the cost-complexity function:
  ∑ⱼ₌₁ᴶ ∑ᵢ: xᵢ∈Rⱼ (yᵢ − cⱼ)² + αJ
where α is a cost-complexity parameter. 
A grid for α is set up, and a nested collection of trees is derived,
T₍αₘₐₓ₎, … , T₍α₀₎,
where αₘₐₓ returns the root node and T₍α₀₎ returns the maximal tree Tₘₐₓ.
In the end, the optimal tree across the nested sequence is selected by CV or by using a validation set.


### grow and prune:  Classification Tree vs Regression Tree
The conditional mean of a classification tree is the prevalence of y in the j-th interval:
  ĉⱼ = (1/nⱼ) ∑ᵢ:xᵢ∈Rⱼ (yᵢ = p̂ⱼ)
which leads to the prediction  ŷᵢ =
⎧ 0   ĉⱼ ≤ 0.5
⎩ 1   ĉⱼ > 0.5.

The within deviance of a binary classification tree, where yᵢ = 0/1 and ŷᵢ = 0/1, 
is thus reduced to the misclassification error rate
∑ᵢ: xᵢ∈ Rⱼ1 (yᵢ ≠ cⱼ)

However, the error rate is non-differentiable when cⱼ = 0.5, because it equals
1 − max (cⱼ , 1 − cⱼ)
Therefore, two viable alternatives for tree growth have been proposed, which are the Gini index:
  G = ∑ⱼ₌₁ᴶ cⱼ(1 − cⱼ)
or the entropy:
  E = ∑ⱼ₌₁ᴶ [cⱼ log(cⱼ) + (1 − cⱼ) log(1 − cⱼ)]
In the pruning phase, typically also the misclassification error rate is considered as an out-of-sample performance criterion.

### R trees (package rpart)
# Interpret rpart(...)
- formula: y ~ x1 + x2 + ... (response vs predictors).
- data: dataset used to estimate the tree. 
- method:
- "anova" regression tree (continuous y),
- "class" classification tree (factor y),
- "poisson" count response.
- control = rpart.control(...):
---- minsplit: a node is split only if it contains at least minsplit observations (larger ⇒ simpler tree).
---- cp: split is kept only if it implies a relative decrease in within deviance at least equal to cp (larger ⇒ stronger pruning during growth). 

# Interpret printcp(fit) and choose the “right” tree by minimum xerror
printcp shows the nested optimal sub-trees Tⱼ obtained along the CART path.
Columns:
- nsplit: number of splits j (number of leaves − 1)
- rel error: within deviance(Tⱼ) / deviance(root), computed on training set (decreases as cp decreases)
- xerror: cv estimate of rel error (typically has a minimum)
- xstd: standard error of xerror
! Selection rule: choose the row with minimum xerror; let cp.best be the corresponding CP, then prune using cp slightly larger than cp.best!

# Interpret plotcp(fit) or rsq.rpart(fit)
- plotcp(fit): shows how xerror evolves as cp decreases / tree size increases; the preferred subtree corresponds to the minimum xerror. 
- rsq.rpart(fit): left plot shows 1 − rel error = R² on training (via CV) as cp decreases; right plot shows xerror (via 10-fold CV) and its minimum gives Tₚᵣᵤₙₑ.

# Interpret prune(fit, cp=...) and the pruned tree
- prune(fit, cp=cp₀): returns the subtree obtained by cutting back splits, keeping the complexity consistent with cp₀ (use cp₀ just above cp.best). 
- Tree reading:
--- branch length ∝ decrease in within deviance due to the split; 
--- left branch = observations satisfying the split rule; right branch = the others; 
--- each leaf reports the predicted value ĉⱼ = ȳⱼ = E(yᵢ | xᵢ ∈ Rⱼ) (regression) or predicted class (classification).


# Interpret console output of the pruned tree (node, split, n, deviance, yval)
Each row corresponds to a node; leaves are marked with *. 
- node: numeric label (genealogical order). 
- split: rule defining the node (e.g., x ≥ c). 
- n: number of observations in the node. 
- deviance: total within-node deviance (regression: SSE-like; classification: impurity based on entropy/Gini, depending on settings). 
- yval: 
---- regression: ĉⱼ = ȳⱼ; 
---- classification: predicted class (often with class proportions shown)

# Assess predictive ability out-of-sample
Use a test set:
- Regression:
---- RMSE = √( (1/n) ∑ᵢ₌₁ⁿ (yᵢ − ŷᵢ)² ),
---- RRSE = √( ∑ᵢ (yᵢ − ŷᵢ)² / ∑ᵢ (yᵢ − ȳ)² ) = √(1 − R²),
---- MAE = (1/n) ∑ᵢ |yᵢ − ŷᵢ|,
---- RAE = ∑ᵢ |yᵢ − ŷᵢ| / ∑ᵢ |yᵢ − ȳ|. 
- Classification: confusion matrix and misclassification rate
err = 1 − (∑ diagonal) / (total), plus sensitivity and specificity.


###############
### Bagging vs Random Forest
Bagging consists in generating B bootstrap samples, 
estimating a regression or classification tree on each of them using all the predictors, 
and predicting on the out-of-bag (OOB) observations. 
In the end, for each observation, final predictions are obtained by averaging out-of-bag predictions 
(simple mean for regression, majority vote for classification). 
This procedure significantly lowers the variance of a single tree. 
However, it may still suffer from overfitting, because it aggregates fully grown trees.

Unlike bagging, random forest estimates, on each bootstrap sample, 
a tree using only m predictors randomly chosen out of the p possible ones, 
with m = √p for classification and m = p / 3 for regression. 
The rest of the procedure is exactly the same. 
This operation de-correlates the trees estimated on the different bootstrap samples, 
because it prevents the systematic use of the same predictors. 
As a consequence, averaged out-of-bag predictions are expected to present a lower variance.

### AdaBoost.M1 algorithm
1) Initialize the observation weights wᵢ = 1/N, i = 1, 2, …, N.
2) For m = 1 to M:
(a) Fit a classifier Gₘ(x) to the training data using weights wᵢ.
(b) Compute  
     errₘ =  (∑ᵢ₌₁ᴺ wᵢI(yᵢ ≠ Gₘ(xᵢ))) /  (∑ᵢ₌₁ᴺ wᵢ)
(c) Compute
αₘ = log( (1 − errₘ) / errₘ ).
(d) Set
wᵢ ← wᵢ ·exp( αₘ· I(yᵢ ≠ Gₘ(xᵢ))) for  i = 1, 2, …, N.
3) Output
G(x) = sign (∑ₘ₌₁ᴹ αₘ Gₘ(x))

AdaBoost.M1 is a boosting algorithm for binary classification 
which exploits B bootstrap sets to estimate a classifier on each of those, 
where the weights attached to each observations change iteratively on each different set. 
In particular, the weights of wrongly predicted observations are inflated on each bootstrap set, in a way proportional to the estimated classifier accuracy. 
In this way, the method learns to predict the toughest observations, 
by reinforcing their presence in the bootstrap sample, which leads to exploring new tree partitions on each set. 
The final output aggregates predictions by weighting differently the OOB predictions according to the relative classifier accuracy.
When the classifier is a tree, it is needed to setup the interaction depth d, which is the maximum number of variables to (randomly) involve in the splits. 
Thus, the parameters to be tuned are the number of bootstrap samples B and the interaction depth d. 
The correct tuning of AdaBoost.M1 requires a simulation study which records the out-of-sample performance of each parameter combination.

### SLP
SLP (Single Layer Perceptron) is an acyclic network with one hidden layer. 
It is composed by an input layer with p + 1 input neurons, 
a hidden layer with M + 1 hidden neurons, an output layer with K output neurons. 
Each neuron acts like a human brain neuron: takes input signals through the dendrites, elaborates
them in a nonlinear way into the soma, transmits the results of the processing through the axon. 
The SLP is a black box because the prediction model is highly nonlinear, which makes it impossible to actually control
the impact of predictors on the target variable, i.e. the explain the causal phenomenon.

## SLP model formally. Choice of the hidden layer and the Output layer activation. Express the generic number of parameters.
The SLP model is defined as

F(t(βₖ, ϕ)) = F(β₀ₖ + ∑ₘ₌₁ᴹ βₘₖ ϕ (t(αₘ, x))) = F(β₀ₖ + ∑ₘ₌₁ᴹ βₘₖ ϕ (α₀ₘ + ∑ⱼ₌₁ᵖ αⱼₘ)xⱼ)

The connection weights from the input layer to the hidden layer αⱼₘ, j = 0, 1, …, p,   m = 1, …, M, 
and from the hidden layer to the output layer βₘₖ,   m = 0, …, M, k = 1, …, K,
connect the p + 1 input neurons (intercept or bias weight included) to the M hidden neurons,
and the M + 1 hidden neurons (intercept or bias weight included) to the K output neurons. 
The total number of parameters is just = (p + 1)M + (M + 1)K

The activation function of the hidden layer ϕ is typically logistic (to get a codomain [0, 1]) 
or hyperbolic tangent (to get a codomain [−1, 1]). 
The activation function of the output layer F depends on the domain of the target variable: 
  identity if y ∈ ℝ (regression), logistic if y ∈ {0, 1} (binary), softmax if y ∈ {0, 1, …, K} (polytomic).

### connection weights estimation
We first define the N × 1 vector of parameters 
θ = (α₁′,…,αₘ′,…, αᴹ′,β₁′) where N = (p + 1)M + (M + 1)

Recalling equation (1), we write
g(xᵢ, θ) = F( t(βₖ, ϕ) )
and we estimate the SLP by the following optimization problems.

• Regression
minimize over θ ϕ(θ), with    ϕ(θ) = ∑ᵢ₌₁ⁿ (yᵢ − g(xᵢ, θ))²
(SSE for regression);

• Binary classification
minimize over θ ϕ(θ), with   ϕ(θ) = − ∑ᵢ₌₁ⁿ [yᵢ log(g(xᵢ, θ))+(1 − yᵢ) log (1 − g(xᵢ, θ))]
(entropy for binary classification).

The multiple starting technique is needed to prevent the risk of converging to local minima, since g is a highly nonlinear function of many parameters. 
The final solution is then chosen as the minimum across the H solutions obtainedby the gradient descent method starting from H different initializers.

## Weight Decay Technique
The weight decay technique is needed to select the optimal number of hidden neurons M, 
which is the bias–variance trade-off parameter of an SLP 
(i.e. a large M leads to overfitting, a small M leads to non-informativeness).

The optimal M is obtained by solving the following optimization problem:
minimize  ϕ(θ) + λ θ′θ
over θ
where λ ∈ ℝ⁺ is a regularization parameter.

In practice, the weight decay technique requires to:
• set up a grid for λ and M;
• solve the optimization problem for each pair (λ∗, M∗) of the grid;
• estimate the out-of-sample prediction error by cross-validation or by using a single validation set;
• choose the pair (λ∗, M∗) yielding the minimum out-of-sample prediction error.

