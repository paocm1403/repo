function    Mock exam + R
##
I believe that my final grade, will be around 28. I did not find the exam particularly difficult, I think the questions were easier compared to the course material. 
I also believe that both my classmates and I put considerable effort into preparing for the exam, which is why I expect the overall distribution of grades to be fairly high. 
I estimate my own grade as 28 because I could have made a few minor inaccuracies in my answers. 
I''ll try to guess my result will be a bit above the class average, corresponding to approximately the 7th decile of the grade distribution. 
This means that around 70% of students are likely to obtain a score = to or < than mine, while about 30% are likely to achieve a > score. 
But, I don’t know my classmates well so, my assessment has wide confidence interval…
###

1) CS vs STS
in STS, N is first divided into homogeneous subgroups (strata), and a SRS is drawn independently within every stratum. All strata are represented (by design)
STS --> used to increase precision.
in a CS, N is divided into clusters (PSUs), a subset of clusters is sampled, and then either all units or a subsample of units inside selected clusters are observed! 
CS --> used to reduce costs, often at the expense of HIGHER VARIANCE due to intra-cluster similarity

2) Measurement error vs Sampling error
-Sampling err --> because estimates are based on a sample (that is "LIMITED") rather than a census
... so it is a natural variation who affects the variance (precision) of estimates , BUT, it can be quantified (SE and CI) and reduced by increasing n (sample size) and/or improving design!
-Measurement err --> from inaccuracies in data collection, so is often systematic (poorly worded questions, respondent misreporting, interviewer effects, recall bias) 
It is a non-sampling err, it introduces bias, and does not disappear with larger samples! In addition, it is difficult to quantify because depends on this own specificities

3) Combinatory calculus, to derive probabilities of selection?
  Combinatory calculus enumerates all possible samples that can be drawn from a N, under a given design... and also it calculates how often a given unit or cluster appears across these samples! 
  --> These allow us to compute INCLUSION PROBABILITY for samples, clusters, and individual units. So, it Is essential for: unbiased estimation and variance calculation

4) SRS in a small vs large population
- in small N, sampling fraction n/N might be large‼ so sampling WOR is not good, and  fcp factor is important (estimates become more precise as the sample approaches census)
- in large N, sampling fraction is typically very small, dependence between draws is negligible (WOR formulas could be used), the fpc is close to 1

5) FPC: definition and use
FPC adjusts variance estimates to account for sampling WOR from a finite population. It is = (N−n)/N or using Bessel =(N−n)/(N−1) 
It is necessary when the sampling fraction n/N is non-negligible (rule of thumb: above 5%)
In the formula Var(y_bar) = (S²/n)*(N− n)/N
--> (N−n)/N = (1− n/N) = fpc‼ It reflects the fact that sampling WOR from a finite population reduces uncertainty from WR case‼ 
As n grows, the pool of unknown units shrinks! --> so the variance of sample mean decrease across repeated samples!
  
6) Bessel correction
In sample variance    s² = ∑(y_i – y_bar)² / (n-1)
Bessel’s correction is = adjustment of using (n – 1) instead of n
(n–1) rather than n : because the sample mean is computed from the same data!! 
Once the mean is fixed, points are no longer all “free” to vary independently: 
knowing n – 1 values automatically “forced” the last one! (degrees of freedom theory) 
-->  variability from the sample would have been too small if we had divided by n

7) Estimation of mean and total in STS
y_bar(STS) = ∑ (N_h/N) * y_bar_h
t_hat(STS) = ∑ N_h  * y_bar_h

8) SYS and ordering
If the list has a MEANINGFUL ORDER (geographic order, sorted by size, etc.)  
-->  SYS induces implicit STRATIFICATION, often reducing variance by guaranteeing representation  
-->  SO, more precise than SRS‼ (because observations are well spread across the range)

9) Why WR formulas are often used in practice
- WR variance estimators are much simpler and require less information than WOR estimators (which need JOINT inclusion probabilities)
- in addition, when n/N fractions are really small, WR estimators provide a good approximation, enormous simplifying computation

10) Why unit non-response is more problematic than item non-response?
-Unit non-response removes entire obs from n  -->  potentially destroying representativeness and causing serious bias (if non-respondents differ systematically from respondents)‼
-Item non-response affects only specific variables and can often be addressed with imputation or weighting adjustments!  -->  generally less damaging for inference

------------
1) Difference between a “package” and a “library”
package = collection of R functions and data... created to perform "specific tasks"
library = location where installed packages are stored to be used in R sessions using library()
In short: first we install a package, then we load it from library!
  
2) R CODE TO CREATE A DATA-FRAME WITH 2 VARIABLES AND 4 OBSERVATIONS EACH
d <- data.frame(x = c(1, 2, 3, 4), y = c(5, 6, 7, 8))

3) R Explanation
tempid <- mstage(classeslong2, stage = list("stratified", "cluster", "stratified"),varnames = list("strat", "class", "studentid"), size = numberselect, method = list("", "srswor", "srswor"))

This command selects a 3Stages- sample using the mstage() function...
-classeslong2.......  full population
-stage = list("stratified", "cluster", "stratified").....designs at each stage: first= STS,  second=CS,  third=STS within clusters
-varnames = list("strat", "class", "studentid").... units at each stage:  strat = strata, class = clusters (PSUs), studentid = final sampling units (SSUs)
-size = numberselect....  how many units at each stage (strata sizes, number of clusters, number of individuals)
-method = list("", "srswor", "srswor").....sampling method at each stage: first= no method needed for defining strata because is just partitioning, second=srswor, third=srswor
-tempid <-  .... name of the produced 3-stage design performed

4) Purpose of set.seed(...)
It fixes the random numbers generator:
- Without set.seed() results of random numbers change every time
- With set.seed(123) results of random numbers are the same every time (reproducibility)

5) generate an SRS n = 4 from N = 10 WOR
pop <- 1:10
sample (pop, size = 4, replace = FALSE)

6) Generate an SRS of size n=4 from N=10 WOR and make the result reproducible
set.seed(123)
pop <- 1:10
sample(pop, size = 4, replace = FALSE)

7) Select an SRS n=6, N=15, WOR, using the sampling package!
library(sampling)
s <- srswor(6, 15)
(1:15)[s == 1]

8) SYS with EP, n=10 , N=100
library(sampling)
pik <- inclusionprobabilities(rep(1,100), 10)
s <- UPsystematic(pik)
(1:100)[s == 1]

9) SYS with PPS, n=5
size <- c(10, 20, 30, 25, 15)
pik <- inclusionprobabilities(size, 3)
UPsystematic(pik)

10) STS-  with proportional allocation by region
library(sampling)
index <- strata(agpop, stratanames = "region", size = c(103, 21, 135, 41), method = "srswor")
sample_str <- getdata(agpop, index)

-------------

11) What type of sampling design is used in the following code?
  strata(agpop, "region", size = c(103,21,135,41), method="srswor")

-STS WOR
-With independent SRS within each stratum

12) Select a 1S- CS of 5 clusters with EP
library(sampling)
cluster(data = classes, clustername = "class",  size = 5,    method = "srswor")

13) Select 5 clusters with PPS and WR
sample(1:nrow(classes), size = 5,  replace = TRUE, prob = classes$class_size)

13) Select a 2S- CS... Stage-1: select 5 clusters, Stage-2: select 4 units per cluster (SRSWOR)
library(sampling)
tempid <- mstage(classeslong,
                 stage = list("cluster","stratified"),
                 varnames = list("class","studentid"),
                 size = list(5, rep(4,5)),
                 method = list("srswor","srswor"))
sample2 <- getdata(classeslong, tempid)[[2]]

14) Select a 2S- CS with PPS at stage-1 and SRS at stage-2.
pik <- list(classes$class_size / sum(classes$class_size),
            4 / classeslong$class_size)
tempid <- mstage(classeslong,
                 stage = list("cluster","stratified"),
                 varnames = list("class","studentid"),
                 size = list(5, rep(4,5)),
                 method = list("systematic","srswor"),
                 pik = pik)
15) Identify the sampling design 
mstage(data,
       stage = list("cluster","stratified"),
       method = list("systematic","srswor"))
Answer
- 2S - CS
- SYS (PPS) at stage-1
- SRS WOR at stage-2

16) How are sampling weights defined in 2S- CS?
The sampling weight is the inverse of the overall inclusion probability   w_ij = 1 / (pi_i * pi_j|i)         
In R:
sample$weight <- 1 / sample$Prob

17) Estimate the mean of a variable using a two-phase sample in Vietnam survey
library(survey)           
data(vietnam)
vietnam$indexp2 <- vietnam$p2sample == 1
dphase2 <- twophase(  id = list(~1, ~1), weights = list(~phase1wt, ~phase2wt), subset = ~indexp2,data = vietnam,  method = "simple")
svymean(~vietnam, dphase2)

--------------------

SRS                 sample ()
SRS (package)       srswor()
SYS                 UPsystematic()
STS                 strata()
1S- CS              cluster()
2S – CS             mstage()
2P                  twophase()
