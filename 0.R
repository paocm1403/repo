# Flexible modelling 
# --> PRO: better capturing Causal relationship under investigation. 
# ok when suspects that underlying relationship is highly JAGGED or/and IRREDUCIBLE ERROR is substantial.
# --> CONS: less interpretable and needs large n to be reliable. 


#• n extremely large, p is small --> Flex
#• p extremely large, n is small--> Inflex
#• relationship X/Y is highly non-linear --> Flex
#• Var(ϵ) is extremely high --> Flex


# Appropriate degree of flexibility  TRADE-OFF:
# --> estimation Bias decreases as model flexibility increases, 
# --> but, estimation Variance tends to increase.

Parametric method summarizes the Causal relationship Y / X
funct f is usually Linear in the parameters β₀, β₁, …, β_p
are estimated via OLS!!! --> provided that n > p and HOMOSCEDASTICITY (residual variance is constant).
The funct f may be NON-linear in X, including quadratic, cubic, or higher-order terms of one or more X variables. 
Also, can incorporate Interaction Terms, X of the form X_k′ X_k″ etc

CONS: It constrains the regression function f to a specific form. 
more interpretab, but limits complex nonlinear relationships & may introduce Estimation Bias.

 Nonparametric approach:
 - pros: allows the model to adapt more flexibly to the true structure of the data, reducing estimation bias 
 - cons: reduced interpretability of the causal relationship, needs more data for reliable estimation

######
k-fold CV method:
• Divide the data D in K exhaustive and mutually exclusive sub-samples Dk
• For each k:
1) estimate the model on the Training set: all data except the k-th fold
2) calculate CV MSE in the test set
MSEk = 1/nk Σ i∈Dk(ŷi − yi)².
• The final CV MSE estimate is the simply MSE=1/K Σ MSEk

Validation Set approach --> err estimates can be highly Variable and often Upward-biased, as model is fitted on reduced training sample
LOOCV --> error estimates are almost unbiased but very high variability, since the n training sets used at each iteration differ by only 1 ob, SO strongly correlated

#### Bootstrap technique
= a reliable way to estimate the STD of any stat learning method 
In absence of information on the underlying distrib of Y | X (and distrib of the error term ϵ), 

Z = sample of size n
Z*₍ᵦ₎ (for b = 1, …, B) = bootstrap samples  
by drawing n observations with replacement from Z.

Err_boot = (1/B) ∑ᵦ₌₁ᴮ MSE(OOBᵦ) 
with MSE(OOBᵦ) = (1/|I∖Iᵦ|) ∑ᵢ∈ᴵ∖ᴵᵦ (yᵢ − f*ᵦ(xᵢ))²

where:
xᵢ is the predictor vector for obs i,
f*ᵦ is the prediction function fitted on the b-th bootstrap sample Z*ᵦ.

Unbiased because each prediction is evaluated only on OOB obs 

#####
PCA represents a p × 1 vector of real variables x
by an r × 1 vector of artificial- uncorrelated variables z
where z is a linear combination of x  and r ≪ p
No y involved --> unsupervised

PCA used to produce a more interpretable, low-dim Representation of a high-dim data vector not feasible with a standard scatterplot matrix in large dim
PCA not used to model or predict a target variable
So, appropriateness of the chosen dim is subjective

##
PCs of x₁, x₂, …, xp  === zₘ (for m = 1, …, p) 
such that
zᵢₘ = ϕ₁ₘ xᵢ₁ + ϕ₂ₘ xᵢ₂ + … + ϕ_pₘ xᵢ_p
with constraint ∑ⱼ₌₁ᵖ ϕⱼₘ² = 1
PCs are imposed to be uncorrelated, that is, with r(zm,zm′) = 0, for each m ̸= m′.

## PCs derivation
Writing  m-th PC === zₘ = ϕₘ′ x
where ϕₘ is a p-dim coeff vector and x is the p-dim data vector

# --> Derivation of the first coefficient vector:
   ϕ₁ = arg maxϕ V(ϕ′x) = ϕ′Sϕ
under the constraint ∥ϕ∥² = 1, where S is the sample covariance matrix.
---> Applying the Lagrange multiplier method, SOLUTION EQUATION is :
  (S − λ₁ I_p) ϕ₁ = 0
--> this is solved if ϕ₁ is the first eigenvector of S, and λ₁ is the first eigenvalue of S. That is:
  z₁ = ϕ₁′ x
--> In addition, from equation, we derive
  V(z₁) = λ₁ ϕ₁′ϕ₁ = λ₁

# --> SO, Derivation of the second coefficient vector:
   ϕ2 = arg maxϕ V(ϕ′x) = ϕ′Sϕ
under the constraint ∥ϕ₂∥² = 1  and ρ(z₂, z₁) = 0
--> Since COV(z₂, z₁) = ϕ₂′ S ϕ₁ = λ₁ ϕ₂′ϕ₁ =0,  IF AND ONLY IF   ϕ₂′ϕ₁ = 0  by the Lagrange multiplier method the solution equation is
    (S − λ₂ I_p) ϕ₂ = 0 
   with ϕ₂′ϕ₁ = 0 and ϕ₂′ϕ₂ = 1
--> Equation is solved if ϕ₂ is the second eigenvector of S and λ₂ is the second eigenvalue of S.
   V(z₂) = λ₂ ϕ₂′ϕ₂ = λ₂

In the end, for m = 3, …, p, the coeff vector ϕₘ is the m-th eigenvector of S, and λₘ is its corresponding eigenvalue, with V(zₘ) = λₘ.
SO,as a consequence, the PCs zₘ, for m = 1, …, p, are uncorrelated linear combinations of the p, with the max variance possible for each m.


##
There are 3 criteria for  M ≪ p number of PCs:
1) choose the smallest M for which the proportion of Explained Variance (∑ⱼ₌₁ᴹ λⱼ)/tr(S) exceeds 70-80%.
2) Kaiser rule: if the variables are standardized --> tr(S) = p --> THEN, choose the smallest M such that λₘ > 1
3) Choose M visually, using the scree plot of ordered eigenvalues.

It is advisable to standardize the variables whenever they have different scales, 
BUT if the scale is the same, standardization may be avoided

######
PCR = dim-red method for regression problems with a large p. 
PCR replaces the original p with their first M  PCs, 
where M is typically selected through CV. 

PCs are constructed without taking response Y into account. 
Only capture maximum variance in X
As a result, the M artificial predictors, may not be strongly associated with Y.

#####
PLS = another, dim-red method for regression problems with a large p. 
Unlike PCR, PLS constructs linear combinations x₁, …, x_p 
that iteratively max their cov with y, 
while simultaneously capturing as much of the variability in X as possible.

Once the data have been standardized, 
--> 1^ direction coeffs ϕ₁m === derived from Univariate regr coeffs of y on each x
with residual vectors ϵⱼ₁ (j = 1, …, p)
representing the portions of y not explained by the first PLS component z₁ = ϕ₁′ x
--> 2^ direction coeffs ϕ₂m are then derived by regressing y on each of Orthogonalized residual vectors ϵⱼ₁.
--> same procedure can be iteratively repeated for m = 3, …, p.
-------> Once the latent components z₁, …, zₘ have been extracted --> Y is regressed on the M components !
Optimal M is selected using CV (using y to supervise the choice)


###   PLS ≠ PCR : 
In PLS latent directions constructed for max association between y and X.
Result --> M, in PLS tends to be more parsimonious than PCR, typically requiring fewer components to adequately capture y.
But, performance of the 2 methods is often similar, since the var effect usually controls the cov effect

###

The factor model is:  x = Λf + u  
where 
x is a p × 1 data vector, 
Λ is a p × m loadings matrix, 
f is an m × 1 vector of factor scores, 
u is a p × 1 vector of residuals.
Assumptions:
1) factors: have zero mean, are uncorrelated, have unit variance
E(f) = 0 and V(f) = E(ff′) = Iₘ.
2) residuals: have zero mean, are uncorrelated with the factors, have variable-specific variances
E(u) = 0, COV(u, f) = E(uf′) = 0, V(u) = E(uu′) = Ψ,   where Ψ is a diagonal matrix.

Under Assumptions, covariance matrix = Σ = ΛΛ′ + Ψ so it decomposes 
into the sum of a low-rank matrix ΛΛ′, capturing the common variation explained by the factors, 
and a diagonal matrix Ψ, representing variable-specific variability.
Σ := E(xx′) = E[(Λf + u)(Λf + u)′] = ΛE(ff′)Λ′ + ΛE(fu′) + E(uf′)Λ′ + E(uu′) = ΛΛ′ + Ψ

Applying Assumptions 1 and 2 with i ≠ j:
  
-- > V(Xᵢ) = E[(λᵢ′f + uᵢ)²] = λᵢ′E(ff′)λᵢ + E(uᵢ²) = λᵢ′λᵢ + Ψᵢᵢ,
-->  COV(Xᵢ, Xⱼ) = E[(λᵢ′f + uᵢ)(λⱼ′f + uⱼ)] = λᵢ′E(ff′)λⱼ + λᵢ′E(fuⱼ′) + E(uᵢf′)λⱼ + E(uᵢuⱼ) = λᵢ′λⱼ.

Here, λᵢ′λᵢ = communality,  Ψᵢᵢ = specificity.

#
The factor model completely accounts for the covariance between 2 variables, 
But it does not fully explain the individual variances. 
This contrasts with PCA, which is a reconstruction technique not a model.
#

The decomposition is indistinguishable under orthogonal rotations of the factor space.

PROOF: COV(x, f) = E[(Λf + u)f′] = Λ, which implies that Λᵢⱼ represents the covariance between Xᵢ and fⱼ.

The exact factor model is equivariant under scale changes and invariant under orthogonal transformations.

• Scale equivariance means that if y = Cx, then
V(y) = CΛΛ′C′ + CΨC′.

• Orthogonal invariance means that if y = Λ* f + u, where
Λ* = ΛG and G is an m × m orthogonal matrix such that GG′ = G′G = Iₘ, then:
  V(y) = Λ*Λ*′ + Ψ = ΛGG′Λ′ + Ψ = ΛΛ′ + Ψ.

##############

When the variables are measured on Different Scales, 
it is advisable to apply the factor model to the correlation matrix R = D^{-1/2} S D^{-1/2}
rather than to the covariance matrix S, where D is the diagonal matrix containing the variances of Xᵢ, for i = 1, …, p.

Estimation of the loading matrix Λ and the residual covariance matrix Ψ :
The loading matrix Λ contains pm parameters, while Ψ contributes an additional p parameters.
The exact factor model is not identifiable without imposing further constraints, 
since the decomposition of Σ is not unique. 
To ensure identifiability, we require that ΛΨ^{-1}Λ′ be diagonal, 
which enforces conditional uncorrelatedness of the factors. Under this restriction, model is identifiable provided that:
  
  pm + p − m(m + 1)/2 ≥ p(p + 1)/2.

##
PCA is not an appropriate estimator of loadings and specificities 
(except in the limiting case p → ∞), because the orthogonal complement used in PCA has dimension p − m. 
--> SO, any PCA-based estimate of Ψ is rank-deficient and inconsistent.

The principal factor method begins by estimating the communalities as:
  
  ĥᵢ² = 1 − rᵢᵢ⁻¹

where rᵢᵢ is the i-th diagonal element of R⁻¹
The loadings are then obtained by extracting the top m PCs of R − Ψ̂, which coincides with R except for its diagonal, replaced by the estimated communalities = ĥᵢ²
The specificities Ψ̂ are derived accordingly.

#
A major drawback of the principal factor method is that it is not scale-equivariant.

Conversely, The maximum likelihood method assumes x ∼ MVN(0, Σ) with Σ = ΛΛ′ + Ψ,
and yields the sample log-likelihood:
log L(x₁, …, xₙ; Σ) = −(n/2) log|2πΣ| − (1/2) ∑ⁿᵢ₌₁ ( xᵢ − μ )′ Σ⁻¹ ( xᵢ − μ )
--> this likelihood is then iteratively maximized with respect to Λ and Ψ to obtain the maximum likelihood estimates.
A key advantage of the maximum likelihood approach is that it is scale-equivariant, unlike the principal factor method.

###

Factor scores can be estimated using 
1) Bartlett’s method, which corresponds to the GLS (Generalized Least Squares) estimator:
  ^^^^^f̂ᵦ = (Λ̂′ Ψ̂⁻¹ Λ̂)⁻¹ Λ̂′ Ψ̂⁻¹ x    Bartlett’s estimator is unbiased.

2) Thomson’s estimator: an alternative Bayesian approach, estimates the posterior mean of the factor scores, 
 ^^^^^f̂ₜ = Iₘ + Λ̂′ Ψ̂⁻¹ Λ̂)⁻¹ Λ̂′ Ψ   or equivalently ^^^^ f̂ₜ = Λ̂′ Ŝ⁻¹ x   Thomson’s estimator is biased in finite samples.

## 
To improve the interpretability of the estimated factor loadings, the loading matrix can be rotated to produce a more polarized structure. 
Two well-known rotation criteria are:
1) VARIMAX, which maximizes the variance of the loadings within each column;
2) QUARTIMAX, which maximizes the overall variance of the squared loadings.

Inference on the null hypothesis H₀ : Σ = ΛΛ′ + Ψ can be conducted using Bartlett’s test, based on the statistic:
  
  W = (n − (2p + 11)/6 − 2m/3)·log |^^^Λ Λ′ + Ψ̂|

which is approximately distributed as a chi-square with
[(p − m)² − (p + m)] / 2   degrees of freedom.

Bartlett’s test is a generalized likelihood ratio test and is typically applied sequentially for m = 1, 2, …, 
until the resulting p-value becomes non-significant.
Sequential testing allows us to determine the smallest adequate number of factors!!!
  
###### Bayes’formula
given a categorical response variable Y with classes h = 1, …, K, 
and a p × 1 predictor vector X, Bayes’ theorem states that:
P(Y = h | X = x₀) = πₕ fₕ(x₀) / ∑ₗ πₗ fₗ(x₀)
where:
• x₀ is a specific realization of X;
• πₕ is the prior probability of belonging to class h;
• fₕ(x₀) is the value of the (multivariate) density of X | Y = h evaluated at x₀, that is, the likelihood or direct probability under class h;
• P(Y = h | X = x₀) is the posterior probability of belonging to class h, given the observed predictor vector x₀

## how it works
Bayes’ classifier assigns each observation x₀ 
to the class with the largest posterior probability P(Y = h | X = x₀):
  ŷ₀ = h    if and only if
h = arg maxₗ₌₁,…,K P(Y = l | X = x₀).

## gold standard
It achieves the lowest possible misclassification rate 
(which is strictly greater than zero), thanks to its reliance on the true underlying components of the data-generating mechanism.
It presupposes knowledge of 
- the prior probabilities πₗ and
- the multivariate class-conditional densities fₗ(x₀) for l = 1, …, K.
it achieves the lowest possible misclassification rate (which is strictly greater than zero), 
thanks to its reliance on the true underlying components of the data-generating mechanism.

## Error rate
The expected error rate (or misclassification rate) of Bayes’ classifier is
1 − E( maxₗ₌₁,…, P(Y = l | X = x) ).
This error increases as the population class-conditional densities exhibit greater Overlaps, 
and it also depends on the prior probabilities πₕ, for h = 1, …, K.

##
The Naive Bayes classifier assumes that the predictors are mutually Independent!!! 
so that the multivariate density of X can be expressed as the product of p univariate densities, 
each of which can be estimated by assuming a Gaussian distribution or through a nonparametric kernel method. 
However, this independence assumption becomes increasingly unrealistic as the number of predictors p grows.

## differences between Gaussian and kernel Naive Bayes.
--> Naive Bayes – Gaussian estimates, for each class h, the class-specific means and covariance matrices of the p predictors 
using only the observations with Y = h.
The multivariate density of X is then approximated as the product of p univariate Gaussian densities:
  f̂ₕ(x) = ∏ⱼ₌₁ᵖ f̂ₕ(xⱼ)   h = 1, …, K.
Here, the univariate Gaussian density f̂ₕ(xⱼ) is computed using:
  
  the class-specific mean
x̄ₕ = (1 / nₕ) ∑_{i : Yᵢ = h} xᵢ,

the class-specific covariance matrix
Σ̂ₕ = (1 / (nₕ − 1)) ∑_{i : Yᵢ = h} (xᵢ − x̄ₕ)(xᵢ − x̄ₕ)′,

where nₕ is the number of observations in class h, and the estimated prior probability is:
  π̂ₕ = nₕ / n.

All these quantities are then plugged into equation to compute the posterior probabilities.

--> Naive Bayes – kernel replaces the Gaussian assumption with a nonparametric kernel estimator 
for each univariate density f̂ₕ(xⱼ) and again forms the multivariate density as their product:
  f̂ₕ(x) = ∏ⱼ₌₁ᵖ f̂ₕ(xⱼ).
This kernel-based version typically requires a larger sample size than Naive Bayes – Gaussian, 
because it adapts more flexibly to the structure of the training data.

## k-NN
The k-NN classifier computes the Euclidean distance between each point xᵢ and all other points in the training sample. 

These distances are then sorted in increasing order, 
the k observations with the smallest distances from xᵢ
--> set of its k nearest neighbors, denoted Nᵢ.
--> then, computes the proportion of observations in Nᵢ whose label is Y = h. 
--> observation xᵢ is then assigned with a majority-vote rule.

The optimal parameter of k (k controls complexity) is selected via CV:
a larger k reduces variance but increases bias;a smaller k reduces bias but increases variance.

## k-NN sample affected
The k-NN classifier is a Nonparametric, adaptive method, 
meaning that it learns to predict the labels of Y directly from the data 
without assuming a specific functional form. 
As a result, a large sample size is required to prevent substantial structural bias 

##
Logistic regression is GLM because it imposes linearity in the conditional expectation 
through a link function g applied to the target (categorical) variable Y, 
whose conditional probability of success is
π(x) = P(Y = 1 | X = x).

Specifically, the link function is the logit:
  g(Y) = log( π(x) / [1 − π(x)] ),

where π(x) / [1 − π(x)] = P(Y = 1 | X = x) / P(Y = 0 | X = x) is the odds of Y.

The logit link (the log-odds of Y) transforms probabilities from the interval [0, 1] to the entire real line ℝ, 
which enables the meaningful estimation of regression coefficients and ensures that the fitted values correspond to valid probabilities within [0, 1].


## Derive Logit
The GLM is specified as:
  E [ log( P(Y = 1 | X = x) / P(Y = 0 | X = x) ) ] = β₀ + β₁x₁ + … + βₚxₚ.
It follows that:
  P̂(Y = 1 | X = x) = exp(β̂₀ + β̂₁x₁ + … + β̂ₚxₚ) / [ 1 + exp(β̂₀ + β̂₁x₁ + … + β̂ₚxₚ) ],
which is the logistic (or sigmoid) function mapping from ℝᵖ to [0, 1]

## logit coeff via maxim likelihood estim
The target variable Y | X = x follows a binomial distribution:
  Y | X = x ∼ Bin(n, π(x))
where π(x) = P(Y = 1 | X = x) = exp(β₀ + β₁x₁ + … + βₚxₚ) / [1 + exp(β₀ + β₁x₁ + … + βₚxₚ)].

Consequently, the likelihood of the sample (y₁, …, yₙ) is:
  L(y₁, …, yₙ) = π(x)^{∑ᵢ yᵢ} · [1 − π(x)]^{n − ∑ᵢ yᵢ}.

The log-likelihood is:
  ℓ(y₁, …, yₙ) = ∑ᵢ₌₁ⁿ yᵢ · log[π(xᵢ)] + ∑ᵢ₌₁ⁿ (1 − yᵢ) · log[1 − π(xᵢ)]
The coefficient estimates are then obtained as:
  (β̂₀, …, β̂ₚ) = arg max_{β₀, …, βₚ} ℓ(y₁, …, yₙ).
In practice, this optimization problem is solved using the Newton–Raphson algorithm.

## relevant predictor Advantage
Logistic regression quantifies the Net Multiplicative Effect of each predictor on the odds of Y, 
a feature that distinguishes it from the other classification methods.


## differences:  LDA vs Gaussian Naive Bayes
LDA is more general than Gaussian Naive Bayes 
LDA assumes a multivariate Gaussian distribution for the predictors, allowing correlations among them. 
Gaussian Naive Bayes imposes independence across predictors and models each marginal distribution separately.


## LDA vs regression, multiclass
A regression-based approach for multiclass problems (such as multinomial regression) 
requires estimating a separate regression equation for each class except one. 
LDA addresses the multiclass setting 
by solving a single algebraic problem that yields the discriminant directions.


## LDA (classificaz lineare) mechanism, two class
Starting from Bayes formula, 
- the prior probabilities πₕ = nₕ / n   for h = 1, …, K, 
- the multivariate class-conditional densities fₕ(x) =  p-dimensional Gaussian densities with mean x̄ₕ
- common covariance matrix  Σ̂ = (1 / (n − K)) ∑ₕ ∑_{i : Yᵢ = h} (xᵢ − x̄ₕ)(xᵢ − x̄ₕ)′,

Bayes’formula is applied under this Gaussian assumption, 
yielding for each class h a discriminant function δₕ(x)  for any vector x.

As all classes share the same covariance matrix --> each discriminant function is LINEAR in x.
--> observation with vector x is assigned to the class with highest discriminant function

By comparing the discriminant functions for any pair of classes, we obtain the corresponding decision boundary, 
which is LINEAR precisely because of common covariance matrix.

#### LDA Assumptions, what is p is high
For a two-class problem, when p is moderately large,
logistic regression is generally more appropriate than LDA,
because, LDA Assumpts
1) multivariate Normality for predictors... this is increasingly unrealistic as p grows. (Moreover, as the dimensionality increases, the common covariance matrix becomes progressively less stable, which in turn reduces the reliability of the estimated discriminant directions.) 
2) X | Y = h ∼ MVN(μₕ, Σ) for h = 1, …, K ...this tends to deteriorate as p increases,
since it becomes more plausible that the covariance differs across groups.


### Classification Rules
• when pop is Gaussian and n is small =   LDA
• when pop is non-Gaussian and n is small =   Logistic regression
• when decision boundary is non-linear =   k-NN
• when n is large and the p are independent = Naive Bayes - kernel
• when p are independent and Gaussian =  Naive Bayes - Gaussian
