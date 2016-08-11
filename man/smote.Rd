\name{smote}
\alias{smote}
\title{
SMOTE oversampling
}
\description{
Function to  oversample by SMOTE the minority class
}
\usage{
smote(data, fp = 1, k = 5)
}
\arguments{
  \item{data}{
data frame or matrix of data including only the minority class. Rows: examples; columns: features
}
  \item{fp}{
multiplicative factor for the SMOTE oversampling of the minority class (def=1). If fp<1 no oversampling is performed.
}
  \item{k}{
number of the nearest neighbours for SMOTE oversampling (def. 5)
}
}
\details{
If n is the number of examples of the minority class,  then fp*n new synthetic examples are generated according to the SMOTE algorithm and returned in addition to the original set of positives.
If fp<1 no new data are generated and the original data set is returned
}
\value{
a data frame including the original minority class examples plus the SMOTE oversampled data
}

\seealso{
\code{\link{smote_and_undersample}}
}
\examples{
d <- imbalanced.data.generator(n.pos=20, n.neg=1000, n.features=12, n.inf.features=2, sd=1, seed=1);
res <- smote(d$data[d$label==1,],  fp = 2, k = 3);
}
