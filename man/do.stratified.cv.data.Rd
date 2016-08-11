\name{do.stratified.cv.data}
\alias{do.stratified.cv.data}

\title{
Construction of random folds for cross-validation
}
\description{
The function randomly generates fold data for cross-validation
}
\usage{
do.stratified.cv.data(examples, positives, k = 10, seed = 0)
}
\arguments{
  \item{examples}{
vector of integer: indices of the examples
}
  \item{positives}{
vector of integer: Indices of the positive examples. The indices refer to the indices of \code{examples}
}
  \item{k}{
number of folds (def = 10)
}
  \item{seed}{
seed of the random generator (def=0). If is set to 0 no initializitation is performed
}
}
\details{
he folds are separated for positive and negative examples. The elements included in each fold are obtained by random sampling the data.
}
\value{
a list with two components;
\item{fold.non.positives }{a list with \code{k}  components. Each component is a vector with the indices of the non positive elements of the fold}
\item{old.positives }{a list with \code{k} components. Each component is a vector with the indices of the positive elements of the fold}
}
\seealso{
\code{\link{do.stratified.cv.data.from.folds}}
}
\examples{
do.stratified.cv.data(1:100, 1:20, k = 10)
}
