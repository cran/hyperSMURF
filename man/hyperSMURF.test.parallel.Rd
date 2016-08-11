\name{hyperSMURF.test.parallel}
\alias{hyperSMURF.test.parallel}

\title{
Test of a hyperSMURF model -- parallel version
}
\description{ 
A hyperSMURF model is tested on a given data set.
Predictions are performed in parallel: more precisely each RF of the hyperensemble
is executed independently and in parallel and the scores are finally averaged.
}
\usage{
hyperSMURF.test.parallel(data, HSmodel, ncores = 0)
}

\arguments{
  \item{data}{
a data frame or matrix with the test data. Rows: exaples; columns: features
}
  \item{HSmodel}{
a list including the trained random forest models. The models have been trained with \code{\link{hyperSMURF.train.parallel}} or with \code{\link{hyperSMURF.train}}
}
  \item{ncores}{
number of cores used for the parallel execution. If 0, the max number of cores - 1 is selected
}
}

\value{
a named vector with the computed probabilities for each example (hyperSMURF score)
}


\seealso{
\code{\link{hyperSMURF.test}}, \code{\link{hyperSMURF.train.parallel}}, \code{\link{hyperSMURF.train}}
}
\examples{
train <- imbalanced.data.generator(n.pos=10, n.neg=500, 
           n.features=12, n.inf.features=2, sd=0.2, seed=1);
if (requireNamespace("foreach", quietly = TRUE) && 
    requireNamespace("doParallel", quietly = TRUE)) 
    HSmodel <- hyperSMURF.train.parallel(train$data, train$label, 
	             n.part = 5, fp = 2, ratio = 3, k = 5, ncores=2);
test <- imbalanced.data.generator(n.pos=10, n.neg=500, 
       n.features=12, n.inf.features=2, sd=0.2, seed=2);
if (requireNamespace("foreach", quietly = TRUE) && 
     requireNamespace("doParallel", quietly = TRUE)) {
       res <- hyperSMURF.test.parallel(test$data, HSmodel, ncores=2);
       y <- ifelse(test$labels==1,1,0);
       pred <- ifelse(res>0.5,1,0);
       table(pred,y);
}
}
