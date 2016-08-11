\name{hyperSMURF.test}
\alias{hyperSMURF.test}

\title{
Test of a hyperSMURF model 
}
\description{ 
A hyperSMURF model is tested on a given data set.
Predictions of each RF of the hyperensemble are performed sequentially and the scores of each ensemble are finally averaged.
}
\usage{
hyperSMURF.test(data, HSmodel)
}

\arguments{
  \item{data}{
a data frame or matrix with the test data. Rows: exaples; columns: features
}
  \item{HSmodel}{
a list including the trained random forest models. The models have been trained with \code{\link{hyperSMURF.train.parallel}} or with \code{\link{hyperSMURF.train}}
}
}

\value{
a named vector with the computed probabilities for each example (hyperSMURF score)
}

\seealso{
\code{\link{hyperSMURF.test.parallel}}, \code{\link{hyperSMURF.train.parallel}}, \code{\link{hyperSMURF.train}}
}
\examples{
train <- imbalanced.data.generator(n.pos=20, n.neg=1000, 
       n.features=10, n.inf.features=2, sd=0.1, seed=1);
HSmodel <- hyperSMURF.train(train$data, train$label, 
               n.part = 5, fp = 1, ratio = 2, k = 5);
test <- imbalanced.data.generator(n.pos=20, n.neg=1000, 
        n.features=10, n.inf.features=2, sd=0.1, seed=2);
res <- hyperSMURF.test(test$data, HSmodel);
y <- ifelse(test$labels==1,1,0);
pred <- ifelse(res>0.5,1,0);
table(pred,y);
}
