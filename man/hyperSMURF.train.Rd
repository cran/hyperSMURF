\name{hyperSMURF.train}
\alias{hyperSMURF.train}
\title{
hyperSMURF training
}
\description{
A hyperSMURF model is trained on a given data set. Training data are partitioned, and each RF is separately trained on each partition by SMOTE oversampling of the positives (minority class examples) and undersampling of the negatives (majority class examples). 
Each RF is trained sequentially
}
\usage{
hyperSMURF.train(data, y, n.part = 10, fp = 1, ratio = 1, k = 5, ntree = 10, 
                 mtry = 5, cutoff = c(0.5, 0.5), seed = 0, file = "")
}

\arguments{
  \item{data}{
a data frame or matrix with the train data. Rows: examples; columns: features
}
  \item{y}{
a factor with the labels. 0:majority class, 1: minority class.
}
  \item{n.part}{
number of partitions (def. 10)
}
  \item{fp}{
multiplicative factor for the SMOTE oversampling of the minority class. 
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
number of the features to randomly selected by the decision tree of each base random forest (def.5)
}
  \item{cutoff}{
a numeric vector of length 2. Cutoff for respectively the majority and minority class.
This parameter is meaningful when used with the thresholded version of hyperSMURF (parameter \code{thresh}=TRUE)
}
  \item{seed}{
initialization seed for the random generator. If set to 0(def.) no initialization is performed
}
  \item{file}{
name of the file where the cross-validated hyperSMURF models will be saved. If file=="" (def.) no model is saved.
}
}
\details{
A different random forest is trained on each partition of the training set.
If \code{npos} and \code{nneg} are the the number of respectively the positive and negative examples, for each partition of the training data \code{fp*npos} new synthetic positives constructed by the SMOTE algorithm are added to the training set. The number of negatives is set to \code{ratio*(fp*npos + npos)}. If no enough negatives are available in the partition, then all the negatives in the partition are used to train the base RF associated to the partition. 
}
\value{
A list of trained RF models. Each element of the list is a \code{randomForest} objects of the homonymous package.
}

\seealso{
\code{\link{hyperSMURF.test}}
}
\examples{
train <- imbalanced.data.generator(n.pos=20, n.neg=1000, 
          n.features=10, n.inf.features=2, sd=1, seed=1);
HSmodel <- hyperSMURF.train(train$data, train$label, n.part = 5, fp = 1, ratio = 2);
}

\references{
M. Schubach, M. Re, P.N. Robinson and G. Valentini Imbalance-Aware Machine Learning for Predicting Rare and Common Disease-Associated Non-Coding Variants,
Scientific Reports, Nature Publishing, 7:2959, 2017.
}
