\name{hyperSMURF.cv.parallel}
\alias{hyperSMURF.cv.parallel}

\title{
hyperSMURF cross-validation -- parallel implementation
}
\description{
Automated cross validation of hyperSMURF (hyper-ensemble SMote Undersampled Random Forests) with both training and testing phase parallelized. 
}
\usage{
hyperSMURF.cv.parallel(data, y, kk = 5, n.part = 10, fp = 1, ratio = 1, k = 5, 
   ntree = 10, mtry = 5, seed = 0, fold.partition = NULL, ncores = 0, file = "")
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
  \item{seed}{
initialization seed for the random generator (if set to 0(def.) no inizialization is performed)
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
The cross-validation is performed by randomly constructing the folds (parameter \code{fold.partition} = NULL) or using a set of predefined folds listed in the parameter \code{fold.partition}. The cross validation is performed by training and testing in parallel the base random forests. More precisely for each training set constructed at each step of the cross validation a separated random forest is trained in each of the \code{n.part} partitions of the data, by oversampling the minority class (parameter \code{fp}) and undersampling the majority class (parameter \code{ratio}). The random forest parameters \code{ntree} and \code{mtry} are the same for all the random forest of the hyper-ensemble.
The parameter \code{ncores} allows to choose the number of cores to be used. Note that the selection of a large number of cores when data to be analyzed ara large can be an issue if the available RAM memory is relatively small.

}
\value{
a vector with the cross-validated hyperSMURF probabilities (hyperSMURF scores).
}

\seealso{
\code{\link{hyperSMURF.cv}}, \code{\link{hyperSMURF.corr.cv.parallel}}
}
\examples{
# construction of a synthetic unbalanced data set
d <- imbalanced.data.generator(n.pos=10, n.neg=300, sd=0.1);
if (requireNamespace("foreach", quietly = TRUE) 
          && requireNamespace("doParallel", quietly = TRUE)) 
   res<-hyperSMURF.cv.parallel (d$data, d$labels, kk=2, n.part=3, fp=1, ratio=1, 
        k=3, ntree=7, mtry=2, seed = 1, fold.partition=NULL, ncores=2, file="");
}
