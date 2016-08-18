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
a data frame or matrix with the test data. Rows: examples; columns: features
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
train <- imbalanced.data.generator(n.pos=10, n.neg=200, 
           n.features=10, n.inf.features=2, sd=0.2, seed=1);
if (Sys.info()['sysname']!="Windows")
    HSmodel <- hyperSMURF.train.parallel(train$data, train$label, 
	             n.part = 4, fp = 1, ratio = 2, k = 3, ncores=2);
test <- imbalanced.data.generator(n.pos=10, n.neg=200, 
       n.features=10, n.inf.features=2, sd=0.2, seed=2);
if (Sys.info()['sysname']!="Windows") {
       res <- hyperSMURF.test.parallel(test$data, HSmodel, ncores=2);
       y <- ifelse(test$labels==1,1,0);
       pred <- ifelse(res>0.5,1,0);
       table(pred,y);
}
}
