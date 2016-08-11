\name{do.stratified.cv.data.from.folds}
\alias{do.stratified.cv.data.from.folds}
\title{
Construction of folds for cross-validation from predefined folds
}
\description{
The function generates data for cross-validation from pre-computed folds
}
\usage{
do.stratified.cv.data.from.folds(examples, positives, folds, k = 10)
}
\arguments{
  \item{examples}{
vector of integer: indices of the examples
}
  \item{positives}{
vector of integer: Indices of the positive examples. The indices refer to the indices of \code{examples}
}
  \item{folds}{
vector of indices : its length is equal to examples, with values in the interval \eqn{[0,kk)}. The value indicates the partition in the cross validation step of the class
}
  \item{k}{
number of folds (def = 10)
}
}
\details{
The folds are separated for positive and negative examples. The elements included in each fold are obtained from the vector of fold indices \code{folds}.
}
\value{
a list with two components;
\item{fold.non.positives }{a list with \code{k}  components. Each component is a vector with the indices of the non positive elements of the fold}
\item{old.positives }{a list with \code{k} components. Each component is a vector with the indices of the positive elements of the fold}
}

\seealso{
\code{\link{do.stratified.cv.data}}
}
\examples{
do.stratified.cv.data.from.folds(1:100, 1:10, folds=sample(rep((0:4),20)), k = 5)
}
