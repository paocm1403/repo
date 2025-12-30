##
I believe that my final grade, excluding this question, 
will be 29 out of 30. Overall, I did not find the exam particularly difficult, 
and I think the questions were easier compared to the material covered during the course. 
I also believe that both my classmates and I put considerable effort into preparing for the exam, 
which is why I expect the overall distribution of grades to be fairly high. 
I estimate my own grade as 28 because I could have made a few minor inaccuracies in my answers. 
Given these considerations, I believe my result will be slightly above the class average, 
corresponding to approximately the 7th decile of the grade distribution. 
This means that around 70% of students are likely to obtain a score equal to or lower than mine, 
while about 30% are likely to achieve a higher score.

###

1) Main differences between a Cluster survey and a Stratified sample survey
In a stratified survey, the population is first divided into homogeneous subgroups (strata), 
and a random sample is drawn independently within every stratum. 
All strata are represented by design, and stratification is mainly used to increase precision.
In a cluster survey, the population is divided into clusters (PSUs), 
a subset of clusters is sampled, 
and then either all units or a subsample of units within selected clusters are observed. 
Clustering is mainly used to reduce costs, 
often at the expense of higher variance due to intra-cluster similarity.

2) Main differences between Measurement error and Sampling error

Sampling error arises because estimates are based on a sample (that is "limited") rather than a census; 
it affects the variance (precision) of estimates 
BUT, it can be quantified (SE and CI) and reduced by increasing sample size and/or improving design!

Measurement error arises from inaccuracies in data collection, so is often systematic
(poorly worded questions, respondent misreporting, interviewer effects, recall bias). 
It is a non-sampling error, it introduces bias, and does not disappear with larger samples!
In addition, it is difficult to quantify because depends on this own specificities

3) Why combinatory calculus is necessary to derive probabilities of selection
Combinatory calculus is needed to enumerate all possible samples 
that can be drawn from a population under a given design. 
Knowing how many samples are possible, and how often a given unit or cluster appears across these samples, 
allows us to compute inclusion probabilities for samples, clusters, and individual units. 
These probabilities are essential for unbiased estimation and variance calculation.

4) Simple Random Sample (SRS) in a small vs large population
In a small population, sampling fractions can be large, 
so sampling without replacement induces strong dependence 
between selections and the finite population correction (fpc) is important; 
estimates become more precise as the sample approaches a census.
In a large population, the sampling fraction is typically very small, 
dependence between draws is negligible, the fpc is close to one, 
and results from SRS with and without replacement are practically equivalent.

5) Finite population correction (fpc): definition and use
The finite population correction adjusts variance estimates to account for sampling without replacement 
from a finite population and is typically expressed as 
(N− n)/N or (N − n)/(N − 1)
It is necessary when the sampling fraction n/N is non-negligible (rule of thumb: above 5%). 
It is not necessary when the population is very large relative to the sample.

6) The Bessel correction
The Bessel correction replaces division by n with division by n−1 when estimating variance from a sample. 
It corrects for the downward bias that occurs because the sample mean is used in place 
of the true population mean.


7) Estimation of totals and means in a stratified sample
Let Nₕ and nₕ be the population and sample sizes in stratum h, and ȳₕ the sample mean in stratum h
The estimated total is
tˢᵗʳ = ∑ₕ₌₁ᴴ Nₕ ȳₕ
The estimated mean is
ȳ̂ˢᵗʳ = ∑ₕ₌₁ᴴ (Nₕ/N) ȳₕ
Both are weighted averages of stratum-level estimates.

8) Systematic sampling and ordering of the sampling frame
- Random order: systematic sampling behaves like SRS; standard SRS variance formulas apply.
- Increasing or decreasing order: 
  systematic sampling often yields lower variance than SRS 
because observations are well spread across the range; 
SRS variance formulas tend to overestimate uncertainty.
- Periodic pattern: systematic sampling can be highly inefficient or biased 
if the sampling interval matches the periodicity of the variable of interest.

9) Why with-replacement formulas are often used in practice
First, with-replacement variance estimators are much simpler and require less information 
than without-replacement estimators (which need joint inclusion probabilities).
Second, when sampling fractions are small, 
with-replacement estimators provide a good approximation with minimal loss of accuracy, 
greatly simplifying computation in complex surveys.

10) Why unit non-response is more problematic than item non-response
Unit non-response removes entire observations from the sample, 
potentially destroying representativeness and causing serious bias 
if non-respondents differ systematically from respondents.
Item non-response affects only specific variables and can often be addressed 
with imputation or weighting adjustments, 
making it generally less damaging to survey inference.

1) Difference between a “package” and a “library”
A package is a collection of R functions, data, and documentation created 
to perform specific tasks (for example, statistical analysis or sampling).
A library is the directory on system where installed packages 
are stored and from which they are loaded into an R session using library().
In short: first we install a package, then we load it from a library.

2) R code to create a data frame with two variables and four observations each
mydata <- data.frame(
  x = c(1, 2, 3, 4),
  y = c(5, 6, 7, 8)
)


3) Explanation of the command
tempid <- mstage(
  classeslong2,
  stage = list("stratified", "cluster", "stratified"),
  varnames = list("strat", "class", "studentid"),
  size = numberselect,
  method = list("", "srswor", "srswor")
)


This command selects a three-stage sample using the mstage() function:
--> "classeslong2"
The data frame containing the full population.
--> "stage = list("stratified", "cluster", "stratified")"
Specifies the sampling design at each stage:
1) First stage: stratified sampling,
2) Second stage: cluster sampling,
3) Third stage: stratified sampling within clusters.
--> "varnames = list("strat", "class", "studentid")"
Indicates the variables identifying sampling units at each stage:
-strat: strata,
-class: clusters (PSUs),
-studentid: final sampling units (SSUs).
--> "size = numberselect"
Specifies how many units are selected at each stage (strata sizes, number of clusters, and number of individuals).
--> "method = list("", "srswor", "srswor")"
Defines the sampling method at each stage:
-No method needed for defining strata,
-Simple random sampling without replacement (srswor) at the second and third stages.
--> "tempid <- "
Stores the identifiers and selection probabilities of the sampled units produced by the multi-stage design.

4) Purpose of set.seed(...)
The command set.seed(...) fixes the random number generator state 
so that random operations (such as sampling) are reproducible.
Using the same seed ensures that the same random sample is generated every time the code is run.

5) Command to generate an SRS of size n = 4 from a population of size N = 10 without replacement
sample(1:10, size = 4, replace = FALSE)

Alternatively, using the sampling package:
srswor(4, 10)

Both commands generate a simple random sample without replacement.

#######################



###############

