function
###
SURVEY SAMPLING = selecting a subset of individuals or units from a larger population to DRAW CONCLUSIONS ABOUT POPULATION ( that is INFERENCE)
Survey sampling relies on principles of probability to ensure every unit has a known chance of selection. 
Sampling theory provides formulas and principles to estimate population characteristics (such as means, totals, proportions) from sample data and to assess the uncertainty (variance) of these estimates with known CI.
-	It is alternative to a full census, saving time and cost--> still yielding useful information. 
-	key idea is to use random selection so that the sample is REPRESENTATIVE of the population and free of SELECTION BIAS. 
-	SAMPLING ERROR = the natural variation that occurs because we observe only a sample and not the entire population‼ can be quantified with formulas‼
-	Other errors like MEASUREMENT error (= inaccuracies in data collection, so is often systematic as, poorly worded questions, respondent misreporting, interviewer effects, recall bias) 
+ nonresponse error (missing data from some units) are NON-SAMPLING ERRORS (must be minimized)

•	Population =  entire set of units under study. 
•	Sample = subset of units actually selected from the population. A well-designed sample aims to mirror the population’s characteristics, on average.
•	Sampling Frame = actual database of units from which the sample is actually drawn. It is a concrete representation of the target population, such as a census, household listing. So it is an implicit scheme for generating units because 
It is used before sampling to assign known probabilities of selection. BUT, It may differ from the target population because of coverage errors (missing units, duplicates, or outdated information) A good frame should cover the target population completely; 
•	Sampling Unit =  basic element or collection of elements. In a simple design, the sampling units might be individual persons. In a multi-stage design, the primary sampling unit (PSU) could be a group like a city block or school, and secondary units (SSU) might be students within those PSUs
•	Parameter vs Statistic:  PARAMETER = numerical characteristic of the population (usually unknown, e.g. the true population mean income). STATISTIC = numerical characteristic of the sample (e.g. the sample mean income). So, WE USE SAMPLE STATISTICS TO ESTIMATE POPULATION PARAMETERS
•	Estimator and Estimate: ESTIMATOR = formula for calculating an estimate of a population parameter from sample data. For instance, the sample mean formula is an estimator of the population mean μ. ESTIMATE = the actual numerical value obtained from sample (e.g. y_bar= 52 kg as an estimate of the true mean).
•	Probability Sampling: Each unit in the population has a known, non-zero probability of selection. crucial for unbiased inference! Random sampling methods (SRS, SYS, STS, CS, etc.) are probability sampling designs.
•	Non-Probability Sampling:  No mathematic foundation. Lack theoretical foundation and reliability of probability sampling. We can validate these methods only if we have results for the full population! (Examples: Convenience sampling, Judgment sampling, Haphazard sampling, Purposing sampling, Volunteer-based sampling)

•	Inclusion Probability =  pi_i =  the probability that each unit is selected into the sample. For example, in SRS, pi_i= n/N, BUT, Inclusion probabilities may differ across units in designs with unequal selection chances!
•	Unbiasedness: An estimator is unbiased if its EXPECTED VALUE = TRUE POPULATION PARAMETER. E(theta_hat) = theta. we can derive the expected values and variances of estimators using combinatorial probability. (for example, the sample mean from SRS is an unbiased estimator of the population mean)
•	Sampling Distribution: refers to the probability distribution of a statistic over all possible random samples! As n grows, the Central Limit Theorem often guarantees that the sampling distribution of many estimators (like the sample mean or proportion) approaches a normal distribution, facilitating CI construction.

MCAR --> Missing completely at random --> Deletion, simple imputation
MAR  -->  Missing depends on observed data --> Weighting, imputation
MNAR --> Missing depends on unobserved data --> Very difficult requires models or external info
NON-RESPONSE Unit vs Item 
UNIT non-response = when a sampled unit does not participate the survey at all‼
Consequences: missing information, n shrinks, selection bias if non-response is not random (e.g. only large firms respond), Ignoring = biased estimates !
Solutions: 
-	Non-response weights adjustments
-	Post-stratification = adjust weights so that sample totals match known population totals for key variables (e.g. age, region, sector). But require knowing Population totals 
-	Ranking (calibration) = an extension of post-stratification using multiple margins simultaneously
-	PREVENTION = short questionnaires, incentives etc
ITEM non-response = when a responding unit skips one or more specific questions. So, some variables are missing, while others are observed. Less severe than unit non-response, but still a potential source of bias.
Solutions:
-	Deletion of the whole unit (often wrong‼)
-	Single imputation (Acceptable for descriptive analysis, not ideal for inference)
-	Multiple Imputation using statistical model (statistically valid under MAR)
-	Missing-data options (na.rm = TRUE for means)
--------------------------------
SAMPLE MEAN AND SAMPLE TOTAL (DEFINITIONSSS)
-	Sample mean: y_bar = ∑ y_i/n
-	Sample total: t = ∑ y_i
For all SELF-WEIGHTING DESIGNSS:   mu_hat = y_bar and T_hat = N* y_bar

VARIANCE OF THE MEAN AND STANDARD ERROR (DEFINITIONSS)
Variance of an Estimator measures its dispersion over repeated samples. SE is the square root of the variance.
For a SRS  WR (or very large population):      Var(ȳ) = S² / n
•	Pop variance (S²): if the pop is very heterogeneous, sample means will vary a lot.
•	Sample size (n): averaging over more obs reduces randomness. Each time the n increases, the sample mean decreases until zero (if ninfinite).
------------------------------------
FPC
FPC adjusts variance estimates to account for sampling WOR from a finite population and is typically expressed as (N− n)/N or    Bessel -->  (N − n)/(N − 1) 
It is necessary when the sampling fraction n/N is non-negligible (rule of thumb: above 5%)
In the formula Var(y_bar) = (S² / n) * (N− n)/N
(N− n)/N  =(1 − n/N) = fpc ‼ It reflects the fact that sampling WOR from a finite population reduces uncertainty from WR case‼ As n grows, the pool of unknown units shrinks!  --> so the variance of sample mean decrease across repeated samples!
----------------------------------------------------
BESSEL’S CORRECTION
In sample variance    s² =  ∑ (y_i – y_bar)² / (n-1)
Bessel’s correction = adjustment of using (n – 1) instead of n
(n – 1) rather than n : because the sample mean is computed from the same data. Once the mean is fixed, the data points are no longer all “free” to vary independently: knowing n – 1 values automatically “forced” the last one! (degrees of freedom theory) 
-->  variability measured from the sample tends to be too small if we divide by n. 
------------------------------
SRS
in SRS, each unit inclusion probability = n / N 
PROS: --> The sample is self-weighting w_i = N / n (same for all units = self-weighting design)
• simple and transparent
• serves as a benchmark for other designs!
• unbiased estimators for means and totals    y_bar =  ∑(in S) y_i /n ,  T_hat = N*y_bar,  t = ∑(in S) y_i
• Variance estimation is straightforward
 WR (infinite population): Var(y_bar) = s²/ n
 WOR (fpc): Var(y_bar) = (s²/ n) (1 − n / N)
 WR (infinite population): Var(t) = N² Var(y_bar)
 WOR (fpc): Var(t) = N² Var(y_bar)  
CONS
• requires a complete and accurate sampling frame
• costly for large or geographically dispersed populations
• does not exploit auxiliary information, resulting in higher variance!!!
----------------------------------------
SYS
In SYS, there is a random start where units are selected at a fixed interval along an ordered list.
Sampling interval is k = N/n (it draws n units from a population of size N). Then, we choose a random start between 1 and k, then select every k-th unit
PROS:
• If the list is RANDOMLY ORDERED with respect to the study variable Y --> SYS behaves similarly to SRS and ensures good spread over the list
• If the list has a MEANINGFUL ORDER (geographic order, sorted by size, etc.) --> SYS induces implicit STRATIFICATION, often reducing variance by guaranteeing representation  -->  more precise than SRS‼
CONS:
• Can be biased if there is HIDDEN PERIODIC PATTERN THAT MATCHES THE INTERVAL K --> systematically large or small variance
• In addition… A monotone trend in the list can also lead to over- or under-estimation, even with random start!
• Variance estimation is not easy as SRS
------------------------------------------------
STS 
In STS, population is divided into non-overlapping strata. Independent samples are drawn within each stratum. Allocation can be proportional or not.
The inclusion probability in stratum h =  n_h / N_h
The weight is the inverse:   w_h = N_h / n_h
BUT‼ :
-	in proportional alloc: every sampled unit represents the same number of pop units (=self-weighting)
-	in Non-proportional alloc: (Neyman or Optimal): weights differ by stratum! 
  PROS:
• STS really improves precision, when within-stratum variability is substantially lower ≪ than overall-population variability‼
• guarantees representation of all key subpopulations‼
• allows for stratum-specific estimates‼
CONS:
• if stratification chosen is weakly informative or unrelated to the study variable Y --> STS offers no efficiency gains over SRS and in case of non-PPS may even lead to higher variance than SRS
• requires information to define strata
• more complex design and weighting

CHOOSE n_h EFFICIENTLY (STS)
“Efficiency” = MINIMIZING THE VARIANCE of the estimator, given a fixed n (usually because of a fixed budget)
1)	PROPORTIONAL ALLOCATION            n_h = n (N_h / N)
Each stratum receives a sampling fraction that is proportional to its size. 
(If a stratum contains 30% of the population → it receives 30% of the sample)
(In practice, the within-stratum variances sigma_h are not known a priori and are therefore replaced by preliminary estimates obtained from past surveys, theory background, literature, etc.)
PRO: It is a simple and intuitive approach, “ok” when strata have similar variability
LIMIT: It does not account for neither differences in variability across strata, nor costs
2)	NEYMAN ALLOCATION      n_h = n [(N_h sigma_h) / ∑ N_h sigma_h]
More sample is allocated to strata with higher variability, because:
•	highly variable strata contribute more to the total variance
--> 	this allocation minimizes the variance of the estimator, given a fixed total sample size and equal costs per unit, but does not account for different costs
--> 	(in practice, the within-stratum variances sigma_h are not known a priori and are therefore replaced by preliminary estimates obtained from past surveys, theory background, literature, etc.)
3)	OPTIMAL ALLOCATION WITH UNEQUAL COSTS ‼
n_h = n  [(N_h sigma_h / √c_h) / ∑ (N sigma / √c)]
This allocation balances 2 factors: Variability of the stratum and Cost of sampling
-->  More sample is allocated to strata that are: Highly variable and Less expensive to sample
4)	EQUAL ALLOCATION          n_h = n / H  for all strata
PROS: very simple as each stratum receives the same sample size. It is good when Strata are very similar, and the main goal is comparisons between strata, not estimation of population totals or means
LIMITS: not variance-optimal, it ignores differences in stratum sizes and variability and costs
--------------------------------------------
CS
in CS, population is divided into clusters, we randomly select only some clusters, then observe all or some units within those clusters. Clusters are often natural groups (households, classrooms, businesses, etc)
PROS: reduce costs and useful when a “sampling frame” of individual elements is not available but a frame of clusters is‼
CONS : 
-->  CS typically has larger variance > than SRS because units within clusters are similar‼
Var(y_bar_CS) = (σ² / n) [ 1 + (m − 1) ρ ]    (m=average cluster side)
The factor 1 + (m − 1)ρ is the DEFF due to clustering‼ It shows that the loss of efficiency in CS depends on both the intra-cluster correlation and the average cluster size, and it formally demonstrates, why clustering typically increases variance relative to SRS‼
--> In addition, in EP, if clusters vary substantially in size M_j, increases the variance of estimators, even when estimation is unbiased and correctly weighted.
-->  In CS, because the independent pieces of information arise from clusters rather than from individual obs, inference must be based on nc, not on n. As a result, CIs and hypothesis tests rely on a t-distrib with df based of nc‼

Clusters can be selected with: EP or PPS
-->  EP, if clusters vary substantially in size M_j, increases the variance of estimators, even when estimation is unbiased and correctly weighted‼
-->  PPS sampling selects clusters with probability proportional to size! Giving individuals more equal inclusion chances, improving efficiency!

PPS Sampling
In unequal probability sampling, often implemented in CS context of multistage complex surveys...clusters are selected with PPS measure. 
“size” does not refer to physical size, but to an auxiliary variable correlated with the study variable y (e.g. number of households, employees, students, sales, land area) ‼ Sampling weights are defined as the inverse of the inclusion probabilities.
PPS WR: a PSU can be selected more than once, typically uses the Hansen–Hurwitz estimator, variance estimate is simpler to calculate in WR (software default is WR typically)
PPS WOR: a PSU can be selected at most once, Inclusion probabilities differ from selection probabilities, requires 1st- , 2nd- n-order inclusion probabilities‼ Estimation typically uses the Horvitz–Thompson estimator and Variance estimation is more complex (usually: Sen–Yates–Grundy)

PROS: improves efficiency when the size measure is correlated with y, is particularly effective for skewed populations, and reduces the variance of total estimates, preventing under-sampling of large PSUs, in addition: facilitates self-weighting designs, widely used in complex survey 
--> 	In 2S- PPS- CS : sampling weight = inverse inclusion probab :   w_ij =  1 / pi_i * pi_j|i 
CONS:  need reliable auxiliary size measures, complex estimation formulas variance WOR, and sensitivity to errors in the size measures.
---------------------------
- 1S- CS: we select a sample of clusters and include ALL those units  in the sample
- 2S- CS: 
Stage 1: select clusters = PSUs   (Primary Sampling Units)
Stage 2: within PSUs, select units = SSUs  (Secondary Sampling Units)
There can be additional stages (e.g. select districts, then schools within districts, then students within schools for a 3S- design)!
The sampling weight =    w_i = 1 / pi_i = (1 / pi_j1) · (1 / pi_i|j2)   This tells: how many units in the pop that sampled unit represents!
  
PROS  • more cost-efficient than 1S- CS 
• reduces respondent burden within clusters 
• flexible design adaptable to field constraints
CONS  • more complex weighting and variance estimation 
• precision depends on both number of clusters and subsampling rates 
• DEFF can be substantial

-------------------------
2P- (Double Sampling)
The idea is to measure x on a large phase-1 sample and y only on a smaller phase-2 subsample, while still using phase-1 information in estimation!
Overall inclusion probability:   pi_i = pi_i(1) · pi_i(2|1)
Sampling weight:      w_i = 1 / pi_i
If both phases are SRS:      pi_i = n² / N   (= to a single SRS of size n²)
Unlike 2S- CS, the sampling units are the same in both phases. 
There is no hierarchical structure (no clusters and elements)!
The distinction between phases refers to when and what is measured, not to different types of sampling units.
PROS:
•	helpful: when the study variable y is expensive or difficult to measure, but : auxiliary variables x are cheap
•	can be exploited to improve efficiency when auxiliar x is strong correlated with y ‼
•	allows use of ratio or regression estimators
CONS:
• more complex design and analysis
• requires modeling assumptions
• risk of inefficiency if correlation is weak!
--------------------------- 
2P– STRAT 
•	in 2P- STS, phase-1 information is not only auxiliary for estimation — it is used to define strata for phase-2 sampling (strata are unknown before phase-1, stratum membership is revealed by measuring x)‼
Unlike basic 2P: phase-2 pi_i differ by stratum and weights are not constant, even if phase-1 is SRS
•	Estimation proceeds as in STS with appropriate combined weights (reverse of pi_i) = w_i= (N/n1) * (n_h/m_h)
VERY BENEFICIAL WHEN:
•	x is strongly correlated with y
•	variance (y_within-stratum) ≪ variance (y_overall)
•	we can allocate m_h efficiently (as in Neyman allocation) 
•	 -->  so,  variance become ≪ than 1P- SRS and of 2P-nonSTS
RISK TO BE AWARE OF: rare or important groups can be oversampled!
------------
STS – CS (Stratified Cluster Sampling)
In STS-CS the pop is first divided into strata, and then clusters (PSUs) are selected within each stratum, often followed by sampling units inside the selected clusters.
STS improves precision, while CS reduces survey costs but introduces within-cluster correlation that affects variance estimation and degrees of freedom.
•	Clustering usually increases variance due to within-cluster correlation.
•	DEFF often approximated =    1 + (m − 1)ρ
Where m= number of units per PSU and ρ=ICC
•	With few PSUs, degrees of freedom are small:
-	use t distributions for CI
-	df approxim= number of PSUs − number of strata
•	RISK TO BE AWARE OF : when PSUs is a small number e.g. < 30  -->  variance estimates are unstable and normal approximations are unreliable, requiring the use of t distributions and careful interpretations!
-----------------------------------  
COMPLEX SAMPLING
Complex sampling designs combine STS, CS, and unequal probabilities across multiple stages of selection.
•	Second-stage fpc should be included if sampling fractions are non-negligible.
•	If second-stage fractions are small, WR approximation is often acceptable.
PROS: are highly flexible and cost-effective for large-scale surveys, enable nationwide or extensive data collection, and are standard practice in official statistics.
CONS: complex to design and implement, require specialized softwares, and carry a risk of incorrect inference if DEFFs are ignored

WEIGHTS ALONE ARE NOT ENOUGH
•	sampling weights alone are sufficient for correct point estimates, but not for correct variance estimation‼ Ignoring the full survey design—especially CS —leads to underestimated s.e. with overly optimistic inference!
•	weighted means via weighted.mean() give point estimates but incorrect SEs.
•	correct variance requires full design info: strata, PSUs, weights, fpc.

Software behavior (survey package)
•	Defaults to WR variance unless fpc provided‼
•	Linearizes ratio and regression estimators automatically.
•	Can report design effect (deff = TRUE).
-	deff > 1 → variance inflation (clustering)
-	deff < 1 → variance reduction (stratification)
CI
•	Large samples: estimate ± z · SE
•	Few PSUs: use t(df) with   df = PSUs − strata
•	In R, we must :  check degf() and specify df explicitly if needed
Nonresponse and calibration = Weight adjustments for nonresponse or post-stratification may increase or decrease variance

SAMPLE SIZE DETERMINATION (n)
Choosing n = depends on desired precision for the estimates and our resource constraints. 
1. decide on the parameter of interest (mean, proportion, total, etc.) and the precision measure (e.g. margin of error e for CI, or target variance)
2. choose a CI level (which gives a z-value or t for the CI)
3. use a formula relating sample size n to the precision target
4. solve for n, then adjust for population size (FPC) if necessary and consider cost/trade-offs
Practically:
-	Proportions: n0 = (z / e)² p(1 − p), we use p = 0.5 if unknown, to be conservative!
-	Usually are used WOR formula to calculate n0 but then is important to apply FPC factor and calculate final n‼
-	Inflate by DEFF for clustering or unequal weighting:
  Example:     n_design = n_SRS · DEFF
STS may reduce variance, but good practise is to be conservative and not reduce n in advance.
-	We use pilot data or prior studies to guess p in proportions cases
-	sigma is usually unknown and must be guessed from past studies, administrative data, pilot sample, etc  (in pilot sample sigma is often replaced by sample s)
-	a rough rule is sigma = range / 4 (for approximately normal data)
-	for small pilot samples, a t critical value instead of z
-	AFTER n formulas, the drawn sample MUST BE INFLATED to account for anticipated nonresponse.
required draws = n / response rate.
Example: need 500 responses, expect 80% response → draw about 625 units.
-	Sometimes CV error is specified instead of a margin of error.
-	In STS, formulas using stratum variances (Cochran) can be used to solve for n iteratively. These require assumptions or pilot estimates of sigma_h
--------------------------------------------------
Estimating Means vs Proportions
Means
•	Estimating averages is easier when the population values are similar to each other.
•	Large dispersion in values implies more uncertainty and therefore larger samples.
•	Prior information (past surveys, pilot studies) is crucial to judge how variable the population is.
Proportions 
A proportion can be seen as the mean of a binary (0/1) variable, where Yi=1 if the unit has the characteristic of interest and Yi=0 otherwise
Var(Y)= p(1 − p), which is maximized at p = 0.5
If p_hat is the sample proportion:     Var(p_hat) = [ p(1 − p) / n ] (1 − n / N)
•	Proportions are hardest to estimate when the population is evenly split (around 50%).
•	Rare characteristics (very small or very large proportions) generally need smaller samples for the same precision.
•	When the true proportion is unknown, planners assume the most conservative case p=0.5
------------------------------------
Ratio Estimator
Auxiliary variable x_i correlated with y_i, and X = ∑ x_i is known 
If y =approx= B x (relationship roughly proportional and passing through the origin)
--> it is more efficient to estimate Y using the known X rather than relying only on the mean y_bar ‼
PROS:
If x and y are positively correlated and the relationship is close to proportional:
•	variance is reduced by factor (1-rho)^2, If rho is close to 1 → large gain
•	estimates are “anchored” to the known X, 
•	under- or over-representation of large-x units in the sample, automatically corrected (ratio!)
BIAS‼
•	ratio estimator is not exactly unbiased, due to the nonlinearity of the ratio.
The bias is typically O(1 / n) and negligible for moderate or large samples. BUT, If the model y = Bx holds exactly, the estimator is unbiased!
•	If rho near 0 --> little or no gain (ratio may even be worse)
--------------------------------------------------
Regression Estimator
Here the auxiliary variable x_i is linearly related to y_i, but the relationship does not necessarily pass through the origin (Ratio Estimator case)
It generalizes ratio estimation and can substantially reduce variance when ρ² is high.
-----------------------------------------------
MULTIPLE AUXILIARY VARIABLES
--> lead to the Generalized regression estimator (GREG)
--> can be applied under SRS, STS, CS using appropriate weights
--> in STS: separate regression (by stratum) may further improve efficiency if relationships differ across strata‼
PROS:
The regression estimator is approximately unbiased:
- if b is estimated, bias is of order O(1 / n) and negligible for moderate n.
- if beta is known in advance, the estimator is exactly unbiased and more efficient‼

---------------
Sampling Weights
The sampling weight of unit i is the number of population units it represents‼ It is the inverse of the inclusion probability:  w_i = 1 / pi_i, where  pi_i= prob(unit i is selected)
weights serve two roles:
1.	point estimation: expand the sample to estimate totals, means, and proportions.
Example: Horvitz–Thompson estimator of a total = T_hat = ∑ {in S} w_i y_i
2.	adjustment: weights are MODIFIED for unequal selection probabilities and compensate for nonresponse or known population totals 
calibration  -->  The resulting analysis weights usually should sum to N
Sampling weights correct point estimates  BUT, correct variance estimation additionally requires full knowledge of DEFFs and fpc ‼
•	ignoring weights in a non-self-weighting design leads to biased estimates
•	weights alone are not sufficient for variance estimation: CS and STS must also be specified, in particular, treating weighted data as SRS usually underestimates standard errors when CS is present!
  
DESIGN WEIGHTS
Weights implied by the sampling design (formulas are always the inverse of inclusion prob pi_i):
•	SRS and SYS   = N / n 
-	STS: (for each i in stratum h)    = N_h / n_h   
-	1S- CS (EP)   = Nc / nc   
-	2S- CS (EP)    = (Nc / nc) * (Mi / mi)
-	STS-CS: =   (Nc_h / nc_h) * (M_hi) / m_hi)
-	2P- STS:  = (N/n1) * (n1_h/n2_h)
-	2P – STS- CS  = (N_h/nc_h)*(M_hi/m_hi)
•	multi-stage cluster sampling:    w_i = (1 / pi_j(1) )*(1 / pi_i|j (2))
•	PPS: units sampled with higher probability receive smaller weights. PPS does not produce a self-weighting design, but it does only if the subsampling within selected PSUs is proportional to the size measure used for PPS‼
•	If all pi_i are equal, the design is self-weighting

Self-weighting vs non-self-weighting
•	Self-weighting: all w_i are equal (e.g. SRS, proportional STS)
unweighted means = weighted means‼
•	Non-self-weighting: w_i vary (e.g. oversampling large units)
Ignoring weights causes biased estimates toward over-represented units‼

Using weights in estimation
•	Total: T_hat = ∑ w_i  y_i
•	Mean: mu_hat = (∑ w_i y_i) / (∑ w_i)
•	Proportion: is a weighted mean of a 0/1 indicator
•	Ratio: (∑ w_i y_i) / (∑ w_i x_i)

---------------------
Formulas: mean, total , variance of the mean

SRS (WOR)
y_bar = (∑y_i) / n
T_hat = N * y_bar
Var(y_bar_hat) = (1 − n/N)* (s² / n)

SYS (WOR) Approximations of SRS!
y_bar = (∑ y_i) / n
T_hat = N * y_bar
Var(y_bar_hat) = (1 − n/N) * (s² / n)

STS (SRSWOR at each stratum)  h = 1,…,H  where W_h = N_h / N
y_bar = ∑ W_h * y_bar_h
T_hat = ∑ N_h * y_bar_h
Var(y_bar_hat) = ∑ W²_h * (1 − n_h/N_h) *(s²_h/n_h)

1S–CS (EP) (WOR) 
Nc = # of clusters in N, nc = # of clusters sampled, 
t_i = ∑ y_ij ,   t_bar = ∑ t_i /nc    s²_t = (1/(nc−1)) ∑(t_i −t_bar)²
T_hat = (Nc / nc)* ∑ t_i
y_bar = T_hat / N
Var(y_bar_hat) = (1/N²) * Nc² * (1 − nc/Nc) *(s²_t /nc)

2S–CS (EP) (WOR) 
M_i = SSU within cluster i,   m_i = SSU sampled,   y_bar_i = (∑ y_ij) / m_i
Mean(ratio) = y_bar = (∑ M_i * y_bar_i) / (∑ M_i)
T_hat = (Nc / nc) * ∑ M_i * y_bar_i
Var(y_bar_hat) = s²_r / ( nc * M²_bar )  --------> WR approximation of WOR
where
s²_r = (1/(nc−1)) ∑ M²_i (y_bar_i − y_bar_hat)²
M_bar = ∑ M_i /nc

2S–CS (PPS) (WOR) Notazione: pi_i= prob. inclusione cluster i,   pi_j|i = prob. SSU j given i
pi_ij = pi_i  * pi_j|i 
T_hat = ∑∑ y_ij / pi_ij  --->  Total (Horvitz–Thompson)   
y_bar = ( ∑∑ y_ij / pi_ij )  / ( ∑∑ 1 / pi_ij )
Var(y_bar_hat) = (1/ N_hat²) * (1/(nc(nc−1))) * ∑ (u_i - u_bar)²  --------> WR approximation of WOR
where u_i= t_hat_i / pi_i,      u_bar = (1/nc) ∑ u_i

2P – (SRS–SRS) Notazione: n₁ = fase 1, n₂ = fase 2
y_bar = y_bar(2)
T_hat = N * y_bar(2)
Var(y_bar_hat) = (1 − n_1/N)* (s_1² / n_1) + (n_1 − n_2)/n_1 * (s_2² / n_2)

2P–STS 
y_bar = ∑ W_h * y_bar_h(2)
T_hat = ∑ N_h * y_bar_h(2)
Var(y_bar_hat) = ∑ W_h² * [(1 − n_1h  /N_h) * (s²_1h / n_1h) + (n_1h − n_2h)/n_1h * (s²_2h / n_2h)]

STS–2S- CS  In each stratum, a sample of clusters is selected; within each selected cluster, a sample of SSUs is drawn. (If all SSUs were obs, this would reduce to STS–1S–CS = census within clusters)
y_bar = T_hat / N = ∑ W_h * y_bar_hat_h  where     y_bar_hat_h =T_hat_h / N_h
T_hat_h = (Nc_h / nc_h) * ∑ t_hat_ih = (Nc_h / nc_h) * ∑ (M_ih * y_bar_ih)
Var(y_bar_hat) = ∑ W_h² * (1 / N_h²) * [ Nc_h² * (1 − nc_h/ Nc_h) *(s_hat_th² / nc_h) + (Nc_h/ nc_h) * ∑ (1 − m_ih / M_ih) * M_ih² * (s_ih² / m_ih)]


The Malaria in The Gambia study uses a STS-3S- PPS- CS design; unbiased estimation requires sampling weights, and variance estimation is simplified using WR approximations.
