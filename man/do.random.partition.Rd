\name{do.random.partition}
\alias{do.random.partition}
\title{
Random partition of the data
}
\description{
Performs a random partition of the indices that refer to a given data set (data frame or matrix)
}
\usage{
do.random.partition(n.ex, n.partitions, seed = 0)
}
\arguments{
  \item{n.ex}{
number of indices to be partitioned
}
  \item{n.partitions}{
number of partitions
}
  \item{seed}{
seed for the random generator
}
}
\details{
The partition of the data is performed using only the indices of the data not the data itself
}
\value{
a list with \code{n.partitions} elements. Each element stores the indices of the partition.
}

\examples{
do.random.partition(100, 10)
}
