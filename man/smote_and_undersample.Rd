\name{smote_and_undersample}
\alias{smote_and_undersample}
\title{
SMOTE oversampling and undersampling
}
\description{
Function to both oversample by SMOTE the minority class and undersample the majority class
}
\usage{
smote_and_undersample(data, y, fp = 1, ratio = 1, k = 5)
}
\arguments{
  \item{data}{
a data frame or matrix. Rows: examples; columns: features
}
  \item{y}{
a factor with the labels. 0:majority class, 1: minority class.
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
}
\details{
If n is the number of examples of the minority class,  then fp*n new synthetic examples are generated according to the SMOTE algorithm and ratio*(fp*n + n) negative examples are undersampled form the majority class.
}
\value{
A list with two entries:
\item{X}{a data frame including the original minority class examples plus the SMOTE oversampled and undersampled data}
\item{Y}{a factor with the labels of the data frame}
}

\seealso{
\code{\link{smote}}
}
\examples{
d <- imbalanced.data.generator(n.pos=20, n.neg=1000, n.features=12, n.inf.features=2, sd=1, seed=1);
res <- smote_and_undersample(d$data, d$label, fp = 2, ratio = 3);
}
