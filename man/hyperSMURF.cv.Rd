\name{hyperSMURF.cv}
\alias{hyperSMURF.cv}

\title{
hyperSMURF cross-validation
}
\description{
Automated cross validation of hyperSMURF (hyper-ensemble SMote Undersampled Random Forests)
}
\usage{
hyperSMURF.cv(data, y, kk = 5, n.part = 10, fp = 1, ratio = 1, 
k = 5, ntree = 10, mtry = 5, cutoff = c(0.5, 0.5), thresh = FALSE, 
                       seed = 0, fold.partition = NULL, file = "")
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
number of trees of the base learner random forest (def. 10)
}
  \item{mtry}{
number of the features to randomly selected by the decision tree of each base random forest (def. 5)
}
  \item{cutoff}{
a numeric vector of length 2. Cutoff for respectively the majority and minority class.
This parameter is meaningful when used with the thresholded version of hyperSMURF parameter (\code{thresh}=TRUE)
}
  \item{thresh}{
logical. If TRUE the thresholded version of hyperSMURF is executed (def: FALSE)
}
  \item{seed}{
initialization seed for the random generator. If set to 0(def.) no initialization is performed
}
  \item{fold.partition}{
vector of size nrow(data) with values in interval [0,kk). The values indicate the fold of the cross validation of each example. 
If NULL (default) the folds are randomly generated.
}
  \item{file}{
name of the file where the cross-validated hyperSMURF models will be saved. If file=="" (def.) no model is saved.
}
}
\details{
The cross-validation is performed by randomly constructing the folds (parameter \code{fold.partition} = NULL) or using a set of predefined folds listed in the parameter \code{fold.partition}. The cross validation is performed by training and testing in sequence the base random forests. More precisely for each training set constructed at each step of the cross validation a separated random forest is trained sequentially for each of the \code{n.part} partitions of the data, by oversampling the minority class (parameter \code{fp}) and undersampling the majority class (parameter \code{ratio}). The random forest parameters \code{ntree} and \code{mtry} are the same for all the random forest of the hyper-ensemble.
}
\value{
a vector with the cross-validated hyperSMURF probabilities (hyperSMURF scores).
}

\seealso{
\code{\link{hyperSMURF.train}}, \code{\link{hyperSMURF.test}}
}
\examples{
d <- imbalanced.data.generator(n.pos=10, n.neg=300, sd=0.3);
res<-hyperSMURF.cv (d$data, d$labels, kk=2, n.part=3, fp=1, ratio=1, k=3, ntree=7, 
                    mtry=2, seed = 1, fold.partition=NULL);
}

\references{
M. Schubach, M. Re, P.N. Robinson and G. Valentini Imbalance-Aware Machine Learning for Predicting Rare and Common Disease-Associated Non-Coding Variants,
Scientific Reports, Nature Publishing, 7:2959, 2017.
}
