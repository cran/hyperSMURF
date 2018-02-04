
# hyperSMURF 1.0
# August 2016


library(randomForest);
library(unbalanced);

# Function to both oversample by SMOTE the minority class and undersample the majority class
# Input:
# data : data frame or matrix of data
# y : factor with the labels of the classes, 0 for the majority and 1 for the minority class
# fp : multiplicative factor for the SMOTE oversampling of the minority class (def. = 1) 
#      If n is the number of examples of the minority class,  then fp*n new synthetic examples are generated 
# ratio : ratio of the #majority/#minority (def.=1)
# k : number of the nearest neighbours for the SMOTE algorithm (def. = 5).
# Output:
# A list with two entries:
# - X: a data frame including the original minority class examples plus the SMOTE oversampled and undersampled data
# - Y: a factor with the labels of the data frame 
smote_and_undersample <- function(data, y, fp=1, ratio=1, k=5) {
  perc.over <- fp * 100;
  perc.under <- (100.0 * ratio * (perc.over+100))/perc.over;
  return (ubSMOTE(as.data.frame(data), y, perc.over = perc.over, k = k, perc.under = perc.under));
}

# Function to  oversample by SMOTE the minority class
# Input:
# data : data frame or matrix of data including only the minority class
# fp : multiplicative factor for the SMOTE oversampling of the minority class (def=1). 
#     If n is the number of examples of the minority class,  then fp*n new synthetic examples are generated 
#     If fp<1 no new data are generated and the original data set is returned
# k : number of the nearest neighbours for the SMOTE algorithm (def. = 5).
# Output: 
# a data frame including the original minority class examples plus the SMOTE oversampled data
smote <- function(data, fp=1, k=5) {
  if (fp<1)
     return(data);
  perc.over <- fp * 100;
  y <- as.factor(rep(1,nrow(data)));
  return (ubSMOTE(as.data.frame(data), y, perc.over = perc.over, k = k, perc.under = 0)$X);
}

# Performs a random partition of the indices
# Input:
# n.ex : The number of indices to be partitioned
# n.partitions : number of partitions
# seed : seed for the random generator
# Output:
# a list with n.partitions elements. Each element store the indices of the partition
do.random.partition <- function(n.ex, n.partitions, seed=0) {
   if (seed!=0)
     set.seed(seed);
   part <- vector(mode="list", length=n.partitions);
   m.per.part <- round(n.ex/n.partitions);
   shuffled <- sample(1:n.ex);
   start=1;
   end=m.per.part;
   for (i in 1:(n.partitions-1)) {     
     part[[i]] <- shuffled[start:end];
	 start <- start+m.per.part;
	 end <- end+m.per.part;
   }
   part[[n.partitions]] <- shuffled[start:n.ex];
   return(part);
}

# hyperSMURF training
# Input:
# data : a data frame or matrix with the training data
# y : a factor with the labels. 0:majority class, 1: minority class.
# n. part : number of partitions
# fp : multiplicative factor for the SMOTE oversampling of the minority class
# ratio : ratio of the #majority/#minority
# k : number of the nearest neighbours for SMOTE
# ntree : number of trees of the rf
# mtry : number of the features to be randomly selected by the rf
# cutoff : a numeric vector of length 2. Cutoff for respectively the majority and minority class
#          This parameter is meaningful when used with the thresholded version of hyperSMURF in the testing phase,
#          i.e. with hyperSMURF.test.thresh
# seed : initialization seed for the random generator
# file : name of the file where the hyperSMURF model will be saved. If file=="" (def.) no model is saved.
# Output:
# A list of trained RF models. Each element of the list is a randomForest object.
hyperSMURF.train <- function (data, y, n.part=10, fp=1, ratio=1, k=5, ntree=10, mtry=5, cutoff=c(0.5,0.5), seed = 0, file="")  {

   data.min <- as.data.frame(data[y==1,]);  # only data of the minority class
   data <- data[-which(y==1),];  # only data of the majority class
   gc();
   n.data <- nrow(data);
   rf.list <- vector(mode="list", n.part);
   
   # Construct the random partitions of majority examples
   
   part <- do.random.partition(n.data, n.partitions=n.part, seed=seed);
   
   for (i in 1:n.part)  {
   
      # SMOTE oversampling
	  data.min.over <- smote(data.min, fp=fp, k=k);
	  n.data.min.over <- nrow(data.min.over);
	  
	  
	  # Majority undersampling and construction of the training set
	  n.maj <- ratio*n.data.min.over;
	  ind.part <- part[[i]];
	  # indices of the majority class
	  if (length(ind.part) >= n.maj)	  
	     indices.maj <- ind.part[1:n.maj]
	  else
	     indices.maj <- ind.part;		  
	  data.train <- rbind(data.min.over, data[indices.maj,]);
	  y <- as.factor(c(rep(1,n.data.min.over), rep(0, length(indices.maj))));
      rf.list [[i]] <-randomForest(data.train, y, ntree = ntree, mtry = mtry, cutoff = cutoff);
	  rm(data.train, data.min.over); gc();
	  cat("Training of ensemble ", i, "done.\n");
   }
   if (file!="")
     save(rf.list, file=file);
   return(rf.list);
}

# hyperSMURF training (parallel multi-core version)
# The training of each RF is performed independently and using parallel computation
# Input:
# data : a data frame or matrix with the training data
# y : a factor with the labels. 0:majority class, 1: minority class.
# n. part : number of partitions
# fp : multiplicative factor for the SMOTE oversampling of the minority class
# ratio : ratio of the #majority/#minority
# k : number of the nearest neighbours for SMOTE
# ntree : number of trees of the rf
# mtry : number of the features to be randomly selected by the rf
# cutoff : a numeric vector of length 2. Cutoff for respectively the majority and minority class
# seed : initialization seed for the random generator
# ncores: number of cores to be used for parallel computation. If 0 (def) all minus one the available cores are used.
# file : name of the file where the hyperSMURF will be saved. If file=="" (def.) no model is saved.
# Output:
# A list of trained RF models. Each element of the list is a randomForest object.
hyperSMURF.train.parallel <- function (data, y, n.part=10, fp=1, ratio=1, k=5, ntree=10, mtry=5, cutoff=c(0.5,0.5), seed = 0, ncores=0, file="")  {

   data.min <- as.data.frame(data[y==1,]);  # only data of the minority class
   data <- data[-which(y==1),];  # only data of the majority class
   gc();
   n.data <- nrow(data);
   rf.list <- vector(mode="list", n.part);
   
   # Construct the random partitions of majority examples
   part <- do.random.partition(n.data, n.partitions=n.part, seed=seed);
   
   # loading parallel libraries and setting of the number of cores
   #library(doParallel);
   #library(foreach);
   if (ncores == 0) {
    n.cores <- detectCores(); 
    if (n.cores > 2)
	  ncores <- n.cores - 1
	else
	  ncores <- n.cores;
   }
   registerDoParallel(cores = ncores);
   
   data.train<-data.min.over <-0;
   gc();
   i=0;
   rf.list <- foreach(i = 1:n.part, .packages="randomForest", .inorder=FALSE) %dopar% {
                       cat("Training of ensemble ", i, " started \n");	 
                       rm(data.train); gc(); 
                       # SMOTE oversampling
	                   data.min.over <- smote(data.min, fp=fp, k=k);
	                   n.data.min.over <- nrow(data.min.over);	  
	                   # Majority undersampling and construction of the training set
	                   n.maj <- ratio*n.data.min.over;
	                   ind.part <- part[[i]];
	                   # indices of the majority class
	                   if (length(ind.part) >= n.maj)	   
	                      indices.maj <- ind.part[1:n.maj]
	                   else
	                      indices.maj <- ind.part;  	   
	                   data.train <- rbind(data.min.over, data[indices.maj,]);
	                   y <- as.factor(c(rep(1,n.data.min.over), rep(0, length(indices.maj))));
                       randomForest(data.train, y, ntree = ntree, mtry = mtry, cutoff = cutoff);		  
   } # end foreach
   stopImplicitCluster();
   gc();
   if (file!="")
     save(rf.list, file=file);
   return(rf.list);
}


# hyperSMURF test
# Input:
# data : a data frame or matrix with the test data
# HSmodel: a list including the trained random forest models
# Output:
# a named vector with the computed probabilities for each example (HyeprSMURF score)
hyperSMURF.test <- function (data, HSmodel)  {

  n.models <- length(HSmodel);
  n.ex <- nrow(data);
  prob <- numeric(n.ex);
  
  for (i in 1:n.models) {
     prob <- prob + predict(HSmodel[[i]], data, type="prob")[,2];
	 gc();
  }
   
  prob <- prob/n.models;
  names(prob) <- rownames(data);
  return(prob);
} 

# hyperSMURF test - thresholded version
# The threshold is embedded in the HSmodel according to the cutoff parameter set in the training phase.
# Input:
# data : a data frame or matrix with the test data
# HSmodel: a list including the trained random forest models
# Output:
# a named vector with the computed probabilities for each example (HyeprSMURF thresholded score)
hyperSMURF.test.thresh <- function (data, HSmodel)  {

  n.models <- length(HSmodel);
  n.ex <- nrow(data);
  prob <- numeric(n.ex);
  
  for (i in 1:n.models) {
	    lab <- predict(HSmodel[[i]], data, type="response");
		p <- ifelse(lab==1, 1, 0);
	    prob <- prob + p;
	    gc();
  }
   
  prob <- prob/n.models;
  names(prob) <- rownames(data);
  return(prob);
} 
  
# hyperSMURF test: parallelized multi-core version
# Predictions are performed in parallel: more precisely each RF of the hyperensemble
# is executed independently and in parallel and the scores are finally averaged.
# Input:
# data : a data frame or matrix with the test data
# HSmodel: a list including the trained random forest models
# ncores: number of cores. If 0, the max number of cores - 1 is selected
# Output:
# a named vector with the computed probabilities for each example (hyperSMURF score)
hyperSMURF.test.parallel <- function (data, HSmodel, ncores=0)  {
  
  #library(doParallel);
  #library(foreach);
  if (ncores == 0) {
    n.cores <- detectCores(); 
    if (n.cores > 2)
	  ncores <- n.cores - 1
	else
	  ncores <- n.cores
  }
 
  registerDoParallel(cores = ncores);
  
  n.models <- length(HSmodel);
  n.ex <- nrow(data);
  prob <- numeric(n.ex);
  i=0;
  prob <- foreach(i = 1:n.models, .combine = "+",  .packages="randomForest", .inorder=FALSE) %dopar% { 
                                     gc();
                                     predict(HSmodel[[i]], data, type="prob")[,2];  
  }
  stopImplicitCluster();
  gc();
  average.prob <- prob/n.models;
  names(average.prob) <- rownames(data);
  return(average.prob);
} 

# hyperSMURF cross-validation
# Input:
# data : a data frame or matrix with the  data
# y : a factor with the labels. 0:majority class, 1: minority class.
# kk : number of folds (def: 5)
# n. part : number of partitions
# fp : multiplicative factor for the SMOTE oversampling of the minority class
#      If fp<1 no oversampling is performed.
# ratio : ratio of the #majority/#minority
# k : number of the nearest neighbours for SMOTE
# ntree : number of trees of the rf
# mtry : number of the features to be randomly selected by the rf
# cutoff : a numeric vector of length 2. Cutoff for respectively the majority and minority class
#          This parameter is meaningful when used with the thresholded version of hyperSMURF (parameter thresh=TRUE)
#          i.e. with hyperSMURF.test.thresh
# thresh : logical. If TRUE the thesholded version of hyperSMURF is exectuted (def: FALSE)
# seed : initialization seed for the random generator ( if set to 0(def.) no inizialization is performed)
# fold.partition: vector of size nrow(data) with values in interval [0,kk). The values indicate the fold of the cross validation of each example. If NULL (default) the folds are randomly generated.
# file : name of the file where the cross-validated hyperSMURF models will be saved. If file=="" (def.) no model is saved.
# Output:
# a vector with the cross-validated hyperSMURF probabilities (hyperSMURF scores).
hyperSMURF.cv <- function (data, y, kk=5, n.part=10, fp=1, ratio=1, k=5, ntree=10, mtry=5, cutoff=c(0.5,0.5), thresh=FALSE, seed = 0, fold.partition=NULL, file="")  {
  
  n.data <- nrow(data);
  indices.positives <- which(y == 1) ;
  scores <- numeric(n.data);
  names(scores) <- rownames(data);
  
  if (is.null(fold.partition)) {
    cat("Creating new folds\n")
    folds <- do.stratified.cv.data(1:n.data, indices.positives, k=kk, seed=seed);
  } else {
    cat("Using given folds\n")
    folds <- do.stratified.cv.data.from.folds(1:n.data,indices.positives,fold.partition,k=kk);
  }
  
  for (i in 1:kk)  {     
	 # preparation of the training and test data
	 ind.test <- c(folds$fold.positives[[i]], folds$fold.non.positives[[i]]);
     ind.pool.pos <- integer(0);
	 ind.pool.neg <- integer(0);
     for (j in 1:kk)
	   if (j!=i)  {
 	     ind.pool.pos <- c(ind.pool.pos, folds$fold.positives[[j]]);
 	     ind.pool.neg <- c(ind.pool.neg, folds$fold.non.positives[[j]]);
 	   }
	 data.train <- data[c(ind.pool.pos, ind.pool.neg),];
	 y.train <- as.factor(c(rep(1, length(ind.pool.pos)), rep (0, length(ind.pool.neg))));
	 # training	 
	 cat("Starting training on Fold ", i, "...\n");
	 HS <- hyperSMURF.train (data.train, y.train, n.part=n.part, fp=fp, ratio=ratio, k=k, ntree=ntree, mtry=mtry, cutoff=cutoff, seed = seed);
	 rm(data.train); gc();
	 # test
	 data.test <- data[ind.test,];
	 cat("Starting test on Fold ", i, "...\n");
	 if (thresh)
	    scores[ind.test] <- hyperSMURF.test.thresh(data.test, HS)
	 else
	    scores[ind.test] <- hyperSMURF.test(data.test, HS);
	 cat("End test on Fold ", i, ".\n");
	 rm(data.test); 
	 if (file=="")
	    rm(HS)
	 else
	    HS.list <- c(HS.list, HS);	 
	 gc(); 
	 cat("Fold ", i, " done -----\n");
  }
  if (file != "")
    save(HS.list, file);
  return(scores);
}


# hyperSMURF cross-validation (parallel version)
# Cross validation of hyperSMURF with both training and testing phase parallelized
# Input:
# data : a data frame or matrix with the  data
# y : a factor with the labels. 0:majority class, 1: minority class.
# kk : number of folds (def: 5)
# n. part : number of partitions
# fp : multiplicative factor for the SMOTE oversampling of the minority class
#      If n is the number of positives, then n*fp novel positive examples are computed through the SMOTE algorithm. 
#      If fp<1 no oversampling is performed.
# ratio : ratio of the #majority/#minority
# k : number of the nearest neighbours for SMOTE
# ntree : number of trees of the rf
# mtry : number of the features to be randomly selected by the rf
# seed : initialization seed for the random generator ( if set to 0(def.) no inizialization is performed)
# fold.partition: vector of size nrow(data) with values in interval [0,kk). The values indicate the fold of the cross validation of each example. If NULL (default) the folds are randomly generated.
# ncores: number of cores. If 0, the max number of cores - 1 is selected
# file : name of the file where the cross-validated hyperSMURF models will be saved. If file=="" (def.) no model is saved.
# Output:
# a vector with the cross-validated hyperSMURF probabilities (ReMM scores).
# Note: currently the thresholded version of hyperSMURF is not available in the parallel implementation.
hyperSMURF.cv.parallel <- function (data, y, kk=5, n.part=10, fp=1, ratio=1, k=5, ntree=10, mtry=5, seed = 0, fold.partition=NULL, ncores=0, file="")  {
  
  # loading parallel libraries and initialization
  #library(doParallel);
  #library(foreach);
  
  n.data <- nrow(data);
  indices.positives <- which(y == 1);
  scores <- numeric(n.data);
  names(scores) <- rownames(data);
  
  if (is.null(fold.partition)) {
    cat("Creating new folds\n")
    folds <- do.stratified.cv.data(1:n.data, indices.positives, k=kk, seed=seed);
  } else {
    cat("Using given folds\n")
    folds <- do.stratified.cv.data.from.folds(1:n.data,indices.positives,fold.partition,k=kk);
  }
  
  for (i in 1:kk)  {     
	 # preparation of the training and test data
	 ind.test <- c(folds$fold.positives[[i]], folds$fold.non.positives[[i]]);
     ind.pool.pos <- integer(0);
	 ind.pool.neg <- integer(0);
     for (j in 1:kk)
	   if (j!=i)  {
 	     ind.pool.pos <- c(ind.pool.pos, folds$fold.positives[[j]]);
 	     ind.pool.neg <- c(ind.pool.neg, folds$fold.non.positives[[j]]);
 	   }
	 data.train <- data[c(ind.pool.pos, ind.pool.neg),];
	 y.train <- as.factor(c(rep(1, length(ind.pool.pos)), rep (0, length(ind.pool.neg))));
	 # training	 
	 cat("Starting training on Fold ", i, "...\n");
	 HS <- hyperSMURF.train.parallel (data.train, y.train, n.part=n.part, fp=fp, ratio=ratio, k=k, ntree=ntree, mtry=mtry, seed = seed, ncores=ncores);
	 rm(data.train); gc();
	 # test
	 data.test <- data[ind.test,];
	 cat("Starting test on Fold ", i, "...\n");
	 scores[ind.test] <- hyperSMURF.test.parallel(data.test, HS, ncores=ncores);
	 cat("End test on Fold ", i, ".\n");
	 rm(data.test); 
	 if (file=="")
	    rm(HS)
	 else
	    HS.list <- c(HS.list, HS);	 
	 gc(); 
	 cat("Fold ", i, " done -----\n");
  }
  if (file != "")
    save(HS.list, file);
  return(scores);
}



# hyperSMURF cross-validation with correlation-based feature selection(parallel version)
# At each step of the cross validatton a subset of features is selected on the traning set by choosing the features most correlated with 
# the response variable and then used to train the ensembles
# Input:
# data : a data frame or matrix with the  data
# y : a factor with the labels. 0:majority class, 1: minority class.
# kk : number of folds (def: 5)
# n. part : number of partitions
# fp : multiplicative factor for the SMOTE oversampling of the minority class
#      If fp<1 no oversampling is performed.
# ratio : ratio of the #majority/#minority
# k : number of the nearest neighbours for SMOTE
# ntree : number of trees of the rf
# mtry : number of the features to be randomly selected by the rf
# n.feature : number of the features to be selected in the training set according to the absolute value of the correlation coefficient.
#            If 0 (def), the top 5% are selected.
# seed : initialization seed for the random generator ( if set to 0(def.) no inizialization is performed)
# fold.partition: vector of size nrow(data) with values in interval [0,kk). The values indicate the fold of the cross validation of each example. If NULL (default) the folds are randomly generated.
# ncores: number of cores. If 0, the max number of cores - 1 is selected
# file : name of the file where the cross-validated hyperSMURF models will be saved. If file=="" (def.) no model is saved.
# Output:
# a vector with the cross-validated hyperSMURF probabilities (hyperSMURF scores).
# Note: currently the thresholded version of hyperSMURF is not available in the parallel implementation.
hyperSMURF.corr.cv.parallel <- function (data, y, kk=5, n.part=10, fp=1, ratio=1, k=5, ntree=10, mtry=5, n.feature=0, seed = 0, fold.partition=NULL, ncores=0, file="")  {
  
  # loading parallel libraries and initialization
  #library(doParallel);
  #library(foreach);
  
  n.data <- nrow(data);
  indices.positives <- which(y == 1);
  scores <- numeric(n.data);
  names(scores) <- rownames(data);
  
  if (is.null(fold.partition)) {
    cat("Creating new folds\n")
    folds <- do.stratified.cv.data(1:n.data, indices.positives, k=kk, seed=seed);
  } else {
    cat("Using given folds\n")
    folds <- do.stratified.cv.data.from.folds(1:n.data,indices.positives,fold.partition,k=kk);
  }
  
  for (i in 1:kk)  {     
	 # preparation of the training and test data
	 cat("Starting preparation data for Fold ", i, "...\n");
	 ind.test <- c(folds$fold.positives[[i]], folds$fold.non.positives[[i]]);
     ind.pool.pos <- integer(0);
	 ind.pool.neg <- integer(0);
	 gc();
     for (j in 1:kk)
	   if (j!=i)  {
 	     ind.pool.pos <- c(ind.pool.pos, folds$fold.positives[[j]]);
 	     ind.pool.neg <- c(ind.pool.neg, folds$fold.non.positives[[j]]);
 	   }
	 data.train <- data[c(ind.pool.pos, ind.pool.neg),];
	 gc();
	 # correlation based feature selection
	 cat("Starting feature selection on Fold ", i, "...\n");
	 if (n.feature==0) {
	    n.feat <- ncol(data.train);
		n.feature <- ceiling((n.feat*5)/100);
	 }
	 y.train <- as.factor(c(rep(1, length(ind.pool.pos)), rep (0, length(ind.pool.neg))));
	 yy <- ifelse(y.train==1,1,0);
	 yy <-  matrix (yy,ncol=1);
	 res <- cor(yy,data.train);
	 if (is.null(colnames(data.train)))
	    colnames(data.train) <- 1:(ncol(data.train));
     feat.names <- colnames(data.train);
     res <- as.numeric(res);
     names(res) <- feat.names;
	 ind.selected <- order(abs(res), decreasing=TRUE)[1:n.feature];
	 cat("Selected features : \n", colnames(data.train)[ind.selected], "\n");
	 data.train <- data.train[,ind.selected];
	 gc();
	 # training	 
	 cat("Starting training on Fold ", i, "...\n");
	 HS <- hyperSMURF.train.parallel (data.train, y.train, n.part=n.part, fp=fp, ratio=ratio, k=k, ntree=ntree, mtry=mtry, seed = seed, ncores=ncores);
	 rm(data.train); gc();
	 # test
	 data.test <- data[ind.test,ind.selected];
	 cat("Starting test on Fold ", i, "...\n");
	 scores[ind.test] <- hyperSMURF.test.parallel(data.test, HS, ncores=ncores);
	 cat("End test on Fold ", i, ".\n");
	 rm(data.test); 
	 if (file=="")
	    rm(HS)
	 else
	    HS.list <- c(HS.list, HS);	 
	 gc(); 
	 cat("Fold ", i, " done -----\n");
  }
  if (file != "")
    save(HS.list, file);
  return(scores);
}



######################################################################
# Function to generate data for the stratified cross-validation.
# Input:
# examples : indices of the examples (a vector of integer)
# positives: vector of integer. Indices of the positive examples. The indices refer to the indices of examples
# k : number of folds (def = 10)
# seed : seed of the random generator (def=0). If is set to 0 no initiazitation is performed
# Ouptut:
# a list with 2 two components
#   - fold.non.positives : a list with k components. Each component is a vector with the indices of the non positive elements of the fold
#   - fold.positives : a list with k components. Each component is a vector with the indices of the positive elements of the fold
# N.B.: in both elements indices refer to the values of the examples vector	 	 
do.stratified.cv.data <- function(examples, positives, k=10, seed=0) {
  
  if (seed!=0)
     set.seed(seed);
  fold.non.positives <- fold.positives <- list();
  for (i in 1:k) {
    fold.non.positives[[i]] <- integer(0);
    fold.positives[[i]] <- integer(0);
  }
  # examples <- 1:n;
  non.positives <- setdiff(examples,positives);
  # non.positives <- examples[-positives];
  non.positives <- sample(non.positives);
  positives <- sample(positives);
  n.positives <- length(positives);
  resto.positives <- n.positives%%k;
  n.pos.per.fold <- (n.positives - resto.positives)/k;
  n.non.positives <- length(non.positives);
  resto.non.positives <- n.non.positives%%k;
  n.non.pos.per.fold <- (n.non.positives - resto.non.positives)/k;
  j=1; 
  if (n.non.pos.per.fold > 0)
    for (i in 1:k) {
      fold.non.positives[[i]] <- non.positives[j:(j+n.non.pos.per.fold-1)];
      j <- j + n.non.pos.per.fold;
    }
  j.pos=1;  
  if (n.pos.per.fold > 0)
    for (i in 1:k) {
      fold.positives[[i]] <- positives[j.pos:(j.pos+n.pos.per.fold-1)];
      j.pos <- j.pos + n.pos.per.fold;
    }
  
  if (resto.non.positives > 0)
    for (i in k:(k-resto.non.positives+1)) {
      fold.non.positives[[i]] <- c(fold.non.positives[[i]], non.positives[j]);
      j <- j + 1;
    }
  
  if (resto.positives > 0) 
    for (i in 1:resto.positives) {
      fold.positives[[i]] <- c(fold.positives[[i]], positives[j.pos]);
      j.pos <- j.pos + 1;
    }
  
  return(list(fold.non.positives=fold.non.positives, fold.positives=fold.positives));
}

######################################################################
# Function to generate data for cross-validation from pre-computed folds
# Input:
# examples : indices of the examples (a vector of integer)
# positives: vector of integer. Indices of the positive examples. The indices refer to the indices of examples
# folds: vector of length equal to examples, with values in interval [0,kk). The value indicates the partition in the cross validation step of the class
# k : number of folds (def = 10)
# Ouptut:
# a list with 2 two components
#   - fold.non.positives : a list with k components. Each component is a vector with the indices of the non positive elements of the fold
#   - fold.positives : a list with k components. Each component is a vector with the indices of the positive elements of the fold
# N.B.: in both elements indices refer to the values of the examples vector	 
do.stratified.cv.data.from.folds <- function(examples, positives, folds, k=10) {
  
  non.positives <- setdiff(examples,positives);  
  fold.non.positives <- fold.positives <- list();
  
  for (i in 1:k) {
    fold.non.positives[[i]] <- integer(0);
    fold.positives[[i]] <- integer(0);   
    fold.positives[[i]] <- positives[folds[positives]==i-1];
    fold.non.positives[[i]] <- non.positives[folds[non.positives]==i-1];
  }
  
  return(list(fold.non.positives=fold.non.positives, fold.positives=fold.positives)); 
}

######################################################################
# Function to generate synthetic imbalanced data
# A variable number of minority and majority class examples are generated. All the features of the majority class are distributed according 
# to a gausian distributin with mean=0 and sd=1. Of the ovreall n.features n.inf. features of the minority class are distributed according to a gaussian centered in 1 with standard deviation sd.
# Input:
# n.pos: number of positive (minority clsss) examples (def. 100)
# n.neg: number of negative (majority class) examples  (def. 2000)
# n.feaures: total number of features (def. 10)
# n.inf.features: number of informative features (def. 2)
# sd: standard deviation of the informative features (def.1)
# seed: intialization seed for the random number generator. If 0 (def) current clock time is used.
# Output:
# A list with two elements:
# data: the matrix of the synthetic data having pos+n.neg rows and n.features columns
# labels: a factor with the labels of he examples: 1 for minority and 0 for majority class.
# construction of a synthetic unbalanced data set
imbalanced.data.generator <- function(n.pos=100, n.neg=2000, n.features=10, n.inf.features=2, sd=1, seed=0) {
  if (seed!=0)
     set.seed(seed);
  class0 <- matrix(rnorm(n.neg*n.features, mean=0, sd=1), nrow=n.neg);
  class1 <-matrix(rnorm(n.pos*n.inf.features, mean=1, sd=sd), nrow=n.pos);
  classr1<-matrix(rnorm(n.pos*(n.features-n.inf.features), mean=0, sd=1), nrow=n.pos);
  class1 <- cbind(class1,classr1);
  data <- rbind(class1,class0);
  labels<-factor(c(rep(1,n.pos),rep(0,n.neg)), levels=c("1","0"));
  return (list(data=data, labels=labels));
}







