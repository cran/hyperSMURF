\name{imbalanced.data.generator}
\alias{imbalanced.data.generator}
\title{
Synthetic imbalanced data generator
}
\description{
A variable number of minority and majority class examples are generated. All the features of the majority class are distributed according 
to a gausian distributin with mean=0 and sd=1. Of the overall n.features, n.inf. features of the minority class are distributed according to a gaussian centered in 1 with standard deviation sd.
}
\usage{
imbalanced.data.generator(n.pos=100, n.neg=2000, 
   n.features=10, n.inf.features=2, sd=1, seed=0)
}
\arguments{
  \item{n.pos}{
number of positive (minority clsss) examples (def. 100)
}
  \item{n.neg}{
number of negative (majority class) examples  (def. 2000)
}
  \item{n.features}{
total number of features (def. 10)
}
  \item{n.inf.features}{
number of informative features (def. 2)
}
  \item{sd}{
standard deviation of the informative features (def.1)
}
  \item{seed}{
intialization seed for the random number generator. If 0 (def) current clock time is used.
}
}

\value{
A list with two elements:
\item{data}{the matrix of the synthetic data having pos+n.neg rows and n.features columns}
\item{labels}{a factor with the labels of he examples: 1 for minority and 0 for majority class}
}

\examples{
imbalanced.data.generator(n.pos=10, n.neg=200, n.features=6, n.inf.features=2, sd=1)
}
