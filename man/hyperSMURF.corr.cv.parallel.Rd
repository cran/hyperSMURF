\name{hyperSMURF.corr.cv.parallel}
\alias{hyperSMURF.corr.cv.parallel}

\title{
hyperSMURF cross-validation with embedded correlation-based feature selection
}
\description{
This function implements the automated cross-validation procedure with hyperSMURF (hyper-ensemble SMote Undersampled Random Forests), using at the same time a correlation-based feature selection to select the best features to train the hyper-ensemble.
}
\usage{
hyperSMURF.corr.cv.parallel(data, y, kk = 5, n.part = 10, fp = 1, 
   ratio = 1, k = 5, ntree = 10, mtry = 5, n.feature = 0, seed = 0, 
   fold.partition = NULL, ncores = 0, file = "")
}
\arguments{
  \item{data}{
a data frame or matrix with the  data
}
  \item{y}{
a factor with the labels. 0:majority class, 1: minority class.
}
  \item{kk}{
number of folds (def: 5)
} 
  \item{n.part}{
number of partitions (def. 10)
}
  \item{fp}{
multiplicative factor for the SMOTE oversampling of the minority class
If fp<1 no oversampling is performed.
}
  \item{ratio}{
ratio of the #majority/#minority
}
  \item{k}{
number of the nearest neighbours for SMOTE oversampling (def. 5)
}
  \item{ntree}{
number of trees of the base learner random forest
}
  \item{mtry}{
number of the features to randomly selected by the decision tree of each base random forest
}
  \item{n.feature}{
number of the features to be selected in the training set according to the absolute value of the correlation coefficient.
If 0 (def), the top 5\% are selected.
}
  \item{seed}{
initialization seed for the random generator (if set to 0(def.) no initialization is performed)
}
  \item{fold.partition}{
vector of size nrow(data) with values in interval \eqn{[0,kk)}. The values indicate the fold of the cross validation of each example. If NULL (default) the folds are randomly generated.
}
  \item{ncores}{
number of cores. If 0, the max number of cores - 1 is selected
}
  \item{file}{
 name of the file where the cross-validated hyperSMURF models will be saved. If file=="" (def.) no model is saved.
}
}
\details{
The cross-validation is performed by randomly constructing the folds (parameter \code{fold.partition} = NULL) or using a set of predefined folds listed in the parameter vector \code{fold.partition}. The cross validation is performed by training and testing in parallel the base random forests. To this end the parameter \code{ncores} allows to choose the number of cores to be used. Note that by selecting a large number of cores a larger primary memory is needed, and this can be an issue if the data to be analyzed are relatively large with respect to the available RAM memory.
At each step of the cross validation a subset of features is selected on the training set by choosing the features most correlated (according to the Pearson correlation) with the response variable and then the selected features are used to train and test the hyper-ensemble.
}
\value{
a vector with the cross-validated hyperSMURF probabilities (hyperSMURF scores).
}


\seealso{
\code{\link{hyperSMURF.cv}}, \code{\link{hyperSMURF.cv.parallel}}
}
\examples{
d <- imbalanced.data.generator(n.pos=10, n.neg=160, n.features=7, 
                                n.inf.features=1, sd=0.3, seed=1);
if (Sys.info()['sysname']!="Windows")
    res<-hyperSMURF.corr.cv.parallel (d$data, d$labels, kk=2, n.part=2, fp=1, ratio=2, k=5, 
	      ntree=5, mtry=2, n.feature=3, seed = 1, fold.partition=NULL, ncores=2, file="");
}
