## March 30, 2018 ##
## CS 7641 - Machine Learning ##
## Create plots for Assignment 3 - Unsupervised Learning and Dimensionality Reduction ##

library(dplyr)
library(magrittr)


setwd('C:/Users/challan/Desktop/Personal/GeorgiaTech/Machine Learning/Assignment 3/CS7641A3 - Wine/CS7641-A3-Unsupervised-Learning-and-Dimensionality-Reduction-master')

## Part 1: Run the clustering algorithms on your datasets and describe what you see

## vv Global Variables vv ##
{  datasets <- c('BASE',
               'PCA',
               'ICA',
               'RP',
               'RF')
  
  datanames <- c('cancer',
                 'wine')
  
  algos <- c('gmm', 'km')
  
  optimals <- data.frame(datasets = rep(datasets, length(datanames) * 2),
                         datanames = c(rep(datanames[1], length(datasets) * 2), rep(datanames[2], length(datasets) * 2)),
                         clust_algo = rep(algos, length(datasets) * length(datanames)))
  
  optimals %<>% arrange(datanames, clust_algo, datasets)
  
  optimals$value <- c(3, 3, 8, 7, 6,
                       3, 4, 8, 4, 4,
                       4, 4, 4, 4, 4,
                       7, 7, 7, 7, 7)
}


clust_plots <- function(data){
  oldwd <- getwd()
  print (getwd())
  setwd(paste0(oldwd, '/output/', data, '/'))
  
  clusters = 2:10
  
  wine_gmm_line = optimals %>% filter(datasets == data, datanames == 'wine', clust_algo == 'gmm') %>% select(value) %>% as.numeric()
  wine_km_line = optimals %>% filter(datasets == data, datanames == 'wine', clust_algo == 'km') %>% select(value) %>% as.numeric()
  
  cancer_gmm_line = optimals %>% filter(datasets == data, datanames == 'cancer', clust_algo == 'gmm') %>% select(value) %>% as.numeric()
  cancer_km_line = optimals %>% filter(datasets == data, datanames == 'cancer', clust_algo == 'km') %>% select(value) %>% as.numeric()
  
  base_km_sse <- read.csv('SSE.csv')
  base_gmm_ll <- read.csv('logliklihood.csv')
  
  jpeg(paste0('../plots/', data, ' - Elbow Method - K Means.jpg'))
  
  plot(clusters, base_km_sse$cancer.SSE..left., type = 'o',
       ylim = c(min(base_km_sse[ , -1]), max(base_km_sse[ , -1])),
       xlab = 'Number of Clusters',
       ylab = 'Sum Of Squared Errors',
       col = 613,
       main = paste0('K Means', ifelse(data == 'BASE', '', paste0(' - ', data)), '\nElbow Method'))
  lines(clusters, base_km_sse$wine.SSE..left., col = 374, type = 'o')
  legend(x = 'topright', legend = c('Breast Cancer', 'Wine Quality'),
         lty = 1, col = c(613, 374),
         cex = .6)
  abline(v = cancer_km_line, lty = 2, col = 613)
  abline(v = wine_km_line, lty = 2, col = 374)
  
  dev.off()
  
  
  jpeg(paste0('../plots/', data, ' - Elbow Method - GMM.jpg'))
  
  plot(clusters, base_gmm_ll$cancer.log.likelihood, type = 'o',
       ylim = c(min(base_gmm_ll[ , -1]), max(base_gmm_ll[ , -1])),
       xlab = 'Number of Clusters',
       ylab = 'LogLikelihood',
       col = 613,
       main = paste0('GMM', ifelse(data == 'BASE', '', paste0(' - ', data)), '\nElbow Method'))
  lines(clusters, base_gmm_ll$wine.log.likelihood, col = 374, type = 'o')
  legend(x = 'topleft', legend = c('Breast Cancer', 'Wine Quality'),
         lty = 1, col = c(613, 374),
         cex = .6)
  abline(v = cancer_gmm_line, lty = 2, col = 613)
  abline(v = wine_gmm_line, lty = 2, col = 374)
  
  dev.off()
  
  jpeg(paste0('../plots/', data, ' - Cluster Validation - GMM.jpg'))
  
  cancer_MI <- read.csv('cancer adjMI.csv')
  print (getwd())
  wine_MI <- read.csv('wine adjMI.csv')
  
  cancer_MI %$% plot(clusters, t(.[1, -1]), type = 'o',
       ylim = c(min(.[ , -1]), max(.[ , -1])),
       xlab = 'Number of Clusters',
       ylab = 'Adjusted Mutual Information',
       col = 613,
       main = paste0('GMM', ifelse(data == 'BASE', '', paste0(' - ', data)), '\nClusters Validation against Labels using AMI'))
  wine_MI %$% lines(clusters, t(.[1, -1]), type = 'o', col = 374)
  legend(x = 'topright', legend = c('Breast Cancer', 'Wine Quality'),
         lty = 1, col = c(613, 374),
         cex = .6)
  
  dev.off()
  
  
  jpeg(paste0('../plots/', data, ' - Cluster Validation - K Means.jpg'))
  
  cancer_MI %$% plot(clusters, t(.[2, -1]), type = 'o',
                     ylim = c(min(.[ , -1]), max(.[ , -1])),
                     xlab = 'Number of Clusters',
                     ylab = 'Adjusted Mutual Information',
                     col = 613,
                     main = paste0('K Means', ifelse(data == 'BASE', '', paste0(' - ', data)), '\nClusters Validation against Labels using AMI'))
  wine_MI %$% lines(clusters, t(.[2, -1]), type = 'o', col = 374)
  legend(x = 'topright', legend = c('Breast Cancer', 'Wine Quality'),
         lty = 1, col = c(613, 374),
         cex = .6)
  
  dev.off()
  
  setwd(oldwd)
}

for(data in datasets){
  clust_plots(data)
}

clust_plots_2d <- function(data){
  stopifnot(data %in% datasets)
  
  for(name in datanames){
    
    title = NA
    if(name == 'cancer'){
      title <- 'Breast Cancer'
    }
    else{
      title <- 'Wine Quality' 
    }
    
    twod <- read.csv(paste0('./output/', data, '/', name, '2D.csv'))
    
    jpeg(paste0('./output/plots/', data, name, ' GMM 2D.jpg'))
    
    twod %$% plot(x, y,
                  col = gmm_cluster + 10,
                  pch = 20,
                  xaxt = 'n', yaxt = 'n',
                  xlab = '', ylab = '',
                  main = paste0(title, ifelse(data == 'BASE', '', data), '\nGMM Clusters - 2D Projection'))
    
    dev.off()
    
    jpeg(paste0('./output/plots/', data, name, ' KM 2D.jpg'))
    
    twod %$% plot(x, y,
                  col = km_cluster + 10,
                  pch = 20,
                  xaxt = 'n', yaxt = 'n',
                  xlab = '', ylab = '',
                  main = paste0(title, ifelse(data == 'BASE', '', data), '\nKMeans Clusters - 2D Projection'))
				  
	jpeg(paste0('./output/plots/', data, name, ' Orig 2D.jpg'))
    
    
	twod %$% plot(x, y,
                  col = target + 10,
                  pch = 20,
                  xaxt = 'n', yaxt = 'n',
                  xlab = '', ylab = '',
                  main = paste0(title, ifelse(data == 'BASE', '', data), '\nOriginal Clusters - 2D Projection'))
    
    dev.off()
  }
}

for(data in datasets){
  clust_plots_2d(data)
}

## First we read in the dimensionality reduction data for each of our data sets

dim_red_plots <- function(data){
  stopifnot(data %in% datanames)
  
  pca_cutoff <- .8
  ica_cutoff <- .2
  rf_cutoff <- .9
  
  title = NA
  if(data == 'cancer'){
    title <- 'Breast Cancer'
  }
  else{
    title <- 'Wine Quality' 
  }
  
  pca <- read.csv(paste0('./output/PCA/', data, ' scree.csv'), header = F, col.names = c('num_components', 'explained_var'))
  ica <- read.csv(paste0('./output/ICA/', data, ' scree.csv'), header = F, col.names = c('num_components', 'kurtosis'))
  rp_recon <- read.csv(paste0('./output/RP/', data, ' scree2.csv'), col.names = c('num_components', paste0('recon_error_', 1:10)))
  rf <- read.csv(paste0('./output/RF/', data, ' scree.csv'), header = F, col.names = c('num_components', 'feature_importance'))
  
  
  jpeg(paste0('./output/plots/PCA ', data,  '.jpg'))
  
  pca %<>% mutate(percent_var = cumsum(explained_var)/sum(explained_var))
  pca %$% plot(num_components, percent_var, type = 'o',
               ylim = c(0, 1),
               col = 'Green',
               xlab = 'Number of Components',
               ylab = 'Cumulative Explained Variance',
               main = paste0(title, '\nPCA Number of Components'))
  abline(h = pca_cutoff, col = 'gray', lty = 2)
  
  dev.off()
  
  jpeg(paste0('./output/plots/ICA ', data, '.jpg'))
  
  ica_cutoff_ind <- which(ica$kurtosis == max(ica$kurtosis))
  ica %$% barplot(kurtosis,
                  xlab = 'Number of Components',
                  ylab = 'Kurtosis',
                  col = c(rep('Green', ica_cutoff_ind - 1), 613, rep('Green', nrow(rf) - ica_cutoff_ind)),
                  main = paste0(title, '\nICA Number of Components'),
                  names.arg = .$num_components)
  
  dev.off()
  
  ## Both datasets here happen to have their overall lowest reconstruction error on the seventh iteration of RP
  ## If this wasn't true, I would split this into two functions and specify the min column for each
  
  jpeg(paste0('./output/plots/RP ', data, '.jpg'))
  
  rp_recon %$% plot(num_components, recon_error_7, type = 'o',
                    ylim = c(0, 1),
                    col = 'Green',
                    xlab = 'Number of Components',
                    ylab = 'Reconstruction Error',
                    main = paste0(title, '\nRandom Projection Number of Components'))
  abline(h = ica_cutoff, col = 'gray', lty = 2)
  
  dev.off()
  
  jpeg(paste0('./output/plots/RF ', data, '.jpg'))
  
  rf %<>% mutate(percent_imp = cumsum(feature_importance)/sum(feature_importance))
  rf_cutoff_ind <- which(rf$percent_imp > rf_cutoff)[1]
  
  rf %$% barplot(feature_importance,
                 xlab = 'Features',
                 ylab = 'Feature Importance',
                 col = c(rep(613, rf_cutoff_ind), rep('Green', nrow(rf) - rf_cutoff_ind)),
                 main = paste0(title, '\nRandom Forest Feature Selection'))
  
  dev.off()
}

for(data in datanames){
  dim_red_plots(data)
}


## NN ##

# This gets maximum mean_test_scores for dim_red data
{
optimal_nn_arch <- c("{'NN__alpha': 0.01, 'NN__hidden_layer_sizes': (12, 12, 12)}", 
                     "{'NN__alpha': 1.0, 'NN__hidden_layer_sizes': (18, 18)}")
names(optimal_nn_arch) <- c('cancer', 'wine')

cancer_best_hl <- '(12, 12, 12)'
cancer_best_alpha <- .01
cancer_bests <- c()
cancer_overall_bests <- c()

wine_best_hl <- '(18, 18)'
wine_best_alpha <- 1
wine_bests <- c()
wine_overall_bests <- c()

cancer_pca_best <- 6
cancer_ica_best <- 2
cancer_rp_best <- 8
cancer_rf_best <- 6

wine_pca_best <- 9
wine_ica_best <- 7
wine_rp_best <- 10
wine_rf_best <- 7
  
  
  
cancer_nn <- read.csv('./BASE/cancer NN bmk.csv')
wine_nn <- read.csv('./BASE/wine NN bmk.csv')

cancer_best <- cancer_nn %>% filter(param_NN__alpha == cancer_best_alpha, 
                     param_NN__hidden_layer_sizes == cancer_best_hl) %>%
              select(mean_test_score) %>% as.numeric()
wine_best <- wine_nn %>% filter(param_NN__alpha == wine_best_alpha, 
                     param_NN__hidden_layer_sizes == wine_best_hl) %>%
              select(mean_test_score) %>% as.numeric()

cancer_bests <- c(cancer_bests, cancer_best)
wine_bests <- c(wine_bests, wine_best)
cancer_overall_bests <- c(cancer_overall_bests, max(cancer_nn$mean_test_score))
wine_overall_bests <- c(wine_overall_bests, max(wine_nn$mean_test_score))


cancer_nn <- read.csv('./output/PCA/cancer dim red.csv')
wine_nn <- read.csv('./output/PCA/wine dim red.csv')

cancer_best <- cancer_nn %>% filter(param_NN__alpha == cancer_best_alpha, 
                                    param_NN__hidden_layer_sizes == cancer_best_hl,
                                    param_pca__n_components == cancer_pca_best) %>%
  select(mean_test_score) %>% as.numeric()
wine_best <- wine_nn %>% filter(param_NN__alpha == wine_best_alpha, 
                                    param_NN__hidden_layer_sizes == wine_best_hl,
                                    param_pca__n_components == wine_pca_best) %>%
  select(mean_test_score) %>% as.numeric()

cancer_bests <- c(cancer_bests, cancer_best)
wine_bests <- c(wine_bests, wine_best)
cancer_overall_bests <- c(cancer_overall_bests, max(cancer_nn$mean_test_score))
wine_overall_bests <- c(wine_overall_bests, max(wine_nn$mean_test_score))


cancer_nn <- read.csv('./output/ICA/cancer dim red.csv')
wine_nn <- read.csv('./output/ICA/wine dim red.csv')

cancer_best <- cancer_nn %>% filter(param_NN__alpha == cancer_best_alpha, 
                                    param_NN__hidden_layer_sizes == cancer_best_hl,
                                    param_ica__n_components == cancer_ica_best) %>%
  select(mean_test_score) %>% as.numeric()
wine_best <- wine_nn %>% filter(param_NN__alpha == wine_best_alpha, 
                                    param_NN__hidden_layer_sizes == wine_best_hl,
                                    param_ica__n_components == wine_ica_best) %>%
  select(mean_test_score) %>% as.numeric()

cancer_bests <- c(cancer_bests, cancer_best)
wine_bests <- c(wine_bests, wine_best)
cancer_overall_bests <- c(cancer_overall_bests, max(cancer_nn$mean_test_score))
wine_overall_bests <- c(wine_overall_bests, max(wine_nn$mean_test_score))


cancer_nn <- read.csv('./output/RP/cancer dim red.csv')
wine_nn <- read.csv('./output/RP/wine dim red.csv')

cancer_best <- cancer_nn %>% filter(param_NN__alpha == cancer_best_alpha, 
                                    param_NN__hidden_layer_sizes == cancer_best_hl,
                                    param_rp__n_components == cancer_rp_best) %>%
  select(mean_test_score) %>% as.numeric()
wine_best <- wine_nn %>% filter(param_NN__alpha == wine_best_alpha, 
                                    param_NN__hidden_layer_sizes == wine_best_hl,
                                    param_rp__n_components == wine_rp_best) %>%
  select(mean_test_score) %>% as.numeric()

cancer_bests <- c(cancer_bests, cancer_best)
wine_bests <- c(wine_bests, wine_best)
cancer_overall_bests <- c(cancer_overall_bests, max(cancer_nn$mean_test_score))
wine_overall_bests <- c(wine_overall_bests, max(wine_nn$mean_test_score))


cancer_nn <- read.csv('./output/RF/cancer dim red.csv')
wine_nn <- read.csv('./output/RF/wine dim red.csv')

cancer_best <- cancer_nn %>% filter(param_NN__alpha == cancer_best_alpha, 
                                    param_NN__hidden_layer_sizes == cancer_best_hl,
                                    param_filter__n == cancer_rf_best) %>%
  select(mean_test_score) %>% as.numeric()
wine_best <- wine_nn %>% filter(param_NN__alpha == wine_best_alpha, 
                                    param_NN__hidden_layer_sizes == wine_best_hl,
                                    param_filter__n == wine_rf_best) %>%
  select(mean_test_score) %>% as.numeric()

cancer_bests <- c(cancer_bests, cancer_best)
wine_bests <- c(wine_bests, wine_best)
cancer_overall_bests <- c(cancer_overall_bests, max(cancer_nn$mean_test_score))
wine_overall_bests <- c(wine_overall_bests, max(wine_nn$mean_test_score))
}
cancer_bests
wine_bests
cancer_overall_bests
wine_overall_bests
# Plots max overall and max with baseline params
{

jpeg('./output/plots/wine NN dim red and base - baseline.jpg', height = 240)  

barplot(wine_bests, names.arg = datasets, 
        col = c('gray', rep('darkgreen', 4)),
        ylab = 'Average Validation Accuracy',
        main = 'Wine Quality\nNeural Network Performance with Baseline Parameters')

dev.off()

jpeg('./output/plots/cancer NN dim red and base - baseline.jpg', height = 240)  

barplot(cancer_bests, names.arg = datasets, 
        col = c('gray', rep('darkgreen', 4)),
        ylab = 'Average Validation Accuracy',
        main = 'Breast Cancer\nNeural Network Performance with Baseline Parameters')

dev.off()

jpeg('./output/plots/wine NN dim red and base - optimized.jpg', height = 240)  

barplot(wine_overall_bests, names.arg = datasets, 
        col = c('gray', rep('darkgreen', 4)),
        ylab = 'Average Validation Accuracy',
        main = 'Wine Quality\nNeural Network Performance with Optimized Parameters')

dev.off()

jpeg('./output/plots/cancer NN dim red and base - optimized.jpg', height = 240)  

barplot(cancer_overall_bests, names.arg = datasets, 
        col = c('gray', rep('darkgreen', 4)),
        ylab = 'Average Validation Accuracy',
        main = 'Breast Cancer\nNeural Network Performance with Optimized Parameters')

dev.off()
}




# This gets maximum mean_test_scores for cluster centers and cluster probs
{
  optimal_nn_arch <- c("{'NN__alpha': 0.01, 'NN__hidden_layer_sizes': (12, 12, 12)}", 
                       "{'NN__alpha': 1.0, 'NN__hidden_layer_sizes': (18, 18)}")
  names(optimal_nn_arch) <- c('cancer', 'wine')
  
  cancer_best_hl <- '(12, 12, 12)'
  cancer_best_alpha <- .01
  cancer_bests <- c()
  cancer_overall_bests <- c()
  
  wine_best_hl <- '(18, 18)'
  wine_best_alpha <- 1
  wine_bests <- c()
  wine_overall_bests <- c()
  
  cancer_pca_best <- 6
  cancer_ica_best <- 2
  cancer_rp_best <- 8
  cancer_rf_best <- 6
  
  wine_pca_best <- 9
  wine_ica_best <- 7
  wine_rp_best <- 10
  wine_rf_best <- 7
  
  
  ## K Means
  
  cancer_nn <- read.csv('./output/BASE/cancer cluster KMeans.csv')
  wine_nn <- read.csv('./output/BASE/wine cluster KMeans.csv')
  
  cancer_best <- cancer_nn %>% filter(param_NN__alpha == cancer_best_alpha, 
                                      param_NN__hidden_layer_sizes == cancer_best_hl, 
                                      param_km__n_clusters == 5) %>%
    select(mean_test_score) %>% as.numeric()
  wine_best <- wine_nn %>% filter(param_NN__alpha == wine_best_alpha, 
                                      param_NN__hidden_layer_sizes == wine_best_hl,
                                      param_km__n_clusters == 7) %>%
    select(mean_test_score) %>% as.numeric()
  
  cancer_bests <- c(cancer_bests, cancer_best)
  wine_bests <- c(wine_bests, wine_best)
  cancer_overall_bests <- c(cancer_overall_bests, max(cancer_nn$mean_test_score))
  wine_overall_bests <- c(wine_overall_bests, max(wine_nn$mean_test_score))
  
  
  cancer_nn <- read.csv('./output/PCA/cancer cluster KMeans.csv')
  wine_nn <- read.csv('./output/PCA/wine cluster KMeans.csv')
  
  cancer_best <- cancer_nn %>% filter(param_NN__alpha == cancer_best_alpha, 
                                      param_NN__hidden_layer_sizes == cancer_best_hl,
                                      #param_pca__n_components == cancer_pca_best,
                                      param_km__n_clusters == 8) %>%
    select(mean_test_score) %>% as.numeric()
  wine_best <- wine_nn %>% filter(param_NN__alpha == wine_best_alpha, 
                                      param_NN__hidden_layer_sizes == wine_best_hl,
                                      #param_pca__n_components == wine_pca_best,
                                      param_km__n_clusters == 8) %>%
    select(mean_test_score) %>% as.numeric()
  
  cancer_bests <- c(cancer_bests, cancer_best)
  wine_bests <- c(wine_bests, wine_best)
  cancer_overall_bests <- c(cancer_overall_bests, max(cancer_nn$mean_test_score))
  wine_overall_bests <- c(wine_overall_bests, max(wine_nn$mean_test_score))
  
  
  cancer_nn <- read.csv('./output/ICA/cancer cluster KMeans.csv')
  wine_nn <- read.csv('./output/ICA/wine cluster KMeans.csv')
  
  cancer_best <- cancer_nn %>% filter(param_NN__alpha == cancer_best_alpha, 
                                      param_NN__hidden_layer_sizes == cancer_best_hl,
                                      #param_ica__n_components == cancer_ica_best,
                                      param_km__n_clusters == 4) %>%
    select(mean_test_score) %>% as.numeric()
  wine_best <- wine_nn %>% filter(param_NN__alpha == wine_best_alpha, 
                                      param_NN__hidden_layer_sizes == wine_best_hl,
                                      #param_ica__n_components == wine_ica_best,
                                      param_km__n_clusters == 7) %>%
    select(mean_test_score) %>% as.numeric()
  
  cancer_bests <- c(cancer_bests, cancer_best)
  wine_bests <- c(wine_bests, wine_best)
  cancer_overall_bests <- c(cancer_overall_bests, max(cancer_nn$mean_test_score))
  wine_overall_bests <- c(wine_overall_bests, max(wine_nn$mean_test_score))
  
  
  cancer_nn <- read.csv('./output/RP/cancer cluster KMeans.csv')
  wine_nn <- read.csv('./output/RP/wine cluster KMeans.csv')
  
  cancer_best <- cancer_nn %>% filter(param_NN__alpha == cancer_best_alpha, 
                                      param_NN__hidden_layer_sizes == cancer_best_hl,
                                      #param_rp__n_components == cancer_rp_best,
                                      param_km__n_clusters == 4) %>%
    select(mean_test_score) %>% as.numeric()
  wine_best <- wine_nn %>% filter(param_NN__alpha == wine_best_alpha, 
                                      param_NN__hidden_layer_sizes == wine_best_hl,
                                      #param_rp__n_components == wine_rp_best,
                                      param_km__n_clusters == 6) %>%
    select(mean_test_score) %>% as.numeric()
  
  cancer_bests <- c(cancer_bests, cancer_best)
  wine_bests <- c(wine_bests, wine_best)
  cancer_overall_bests <- c(cancer_overall_bests, max(cancer_nn$mean_test_score))
  wine_overall_bests <- c(wine_overall_bests, max(wine_nn$mean_test_score))
  
  
  cancer_nn <- read.csv('./output/RF/cancer cluster KMeans.csv')
  wine_nn <- read.csv('./output/RF/wine cluster KMeans.csv')
  
  cancer_best <- cancer_nn %>% filter(param_NN__alpha == cancer_best_alpha, 
                                      param_NN__hidden_layer_sizes == cancer_best_hl,
                                      #param_filter__n == cancer_rf_best,
                                      param_km__n_clusters == 4) %>%
    select(mean_test_score) %>% as.numeric()
  wine_best <- wine_nn %>% filter(param_NN__alpha == wine_best_alpha, 
                                      param_NN__hidden_layer_sizes == wine_best_hl,
                                     #param_filter__n == wine_rf_best,
                                      param_km__n_clusters == 5) %>%
    select(mean_test_score) %>% as.numeric()
  
  cancer_bests <- c(cancer_bests, cancer_best)
  wine_bests <- c(wine_bests, wine_best)
  cancer_overall_bests <- c(cancer_overall_bests, max(cancer_nn$mean_test_score))
  wine_overall_bests <- c(wine_overall_bests, max(wine_nn$mean_test_score))

  
  
  ## GMM
  
  cancer_nn <- read.csv('./output/BASE/cancer cluster GMM.csv')
  wine_nn <- read.csv('./output/BASE/wine cluster GMM.csv')
  
  cancer_best <- cancer_nn %>% filter(param_NN__alpha == cancer_best_alpha, 
                                      param_NN__hidden_layer_sizes == cancer_best_hl, 
                                      param_gmm__n_components == 5) %>%
    select(mean_test_score) %>% as.numeric()
  wine_best <- wine_nn %>% filter(param_NN__alpha == wine_best_alpha, 
                                      param_NN__hidden_layer_sizes == wine_best_hl,
                                      param_gmm__n_components == 7) %>%
    select(mean_test_score) %>% as.numeric()
  
  cancer_bests <- c(cancer_bests, cancer_best)
  wine_bests <- c(wine_bests, wine_best)
  cancer_overall_bests <- c(cancer_overall_bests, max(cancer_nn$mean_test_score))
  wine_overall_bests <- c(wine_overall_bests, max(wine_nn$mean_test_score))
  
  
  cancer_nn <- read.csv('./output/PCA/cancer cluster GMM.csv')
  wine_nn <- read.csv('./output/PCA/wine cluster GMM.csv')
  
  cancer_best <- cancer_nn %>% filter(param_NN__alpha == cancer_best_alpha, 
                                      param_NN__hidden_layer_sizes == cancer_best_hl,
                                      #param_pca__n_components == cancer_pca_best,
                                      param_gmm__n_components == 8) %>%
    select(mean_test_score) %>% as.numeric()
  wine_best <- wine_nn %>% filter(param_NN__alpha == wine_best_alpha, 
                                      param_NN__hidden_layer_sizes == wine_best_hl,
                                      #param_pca__n_components == wine_pca_best,
                                      param_gmm__n_components == 8) %>%
    select(mean_test_score) %>% as.numeric()
  
  cancer_bests <- c(cancer_bests, cancer_best)
  wine_bests <- c(wine_bests, wine_best)
  cancer_overall_bests <- c(cancer_overall_bests, max(cancer_nn$mean_test_score))
  wine_overall_bests <- c(wine_overall_bests, max(wine_nn$mean_test_score))
  
  
  cancer_nn <- read.csv('./output/ICA/cancer cluster GMM.csv')
  wine_nn <- read.csv('./output/ICA/wine cluster GMM.csv')
  
  cancer_best <- cancer_nn %>% filter(param_NN__alpha == cancer_best_alpha, 
                                      param_NN__hidden_layer_sizes == cancer_best_hl,
                                      #param_ica__n_components == cancer_ica_best,
                                      param_gmm__n_components == 3) %>%
    select(mean_test_score) %>% as.numeric()
  wine_best <- wine_nn %>% filter(param_NN__alpha == wine_best_alpha, 
                                      param_NN__hidden_layer_sizes == wine_best_hl,
                                      #param_ica__n_components == wine_ica_best,
                                      param_gmm__n_components == 9) %>%
    select(mean_test_score) %>% as.numeric()
  
  cancer_bests <- c(cancer_bests, cancer_best)
  wine_bests <- c(wine_bests, wine_best)
  cancer_overall_bests <- c(cancer_overall_bests, max(cancer_nn$mean_test_score))
  wine_overall_bests <- c(wine_overall_bests, max(wine_nn$mean_test_score))
  
  
  cancer_nn <- read.csv('./output/RP/cancer cluster GMM.csv')
  wine_nn <- read.csv('./output/RP/wine cluster GMM.csv')
  
  cancer_best <- cancer_nn %>% filter(param_NN__alpha == cancer_best_alpha, 
                                      param_NN__hidden_layer_sizes == cancer_best_hl,
                                      #param_rp__n_components == cancer_rp_best,
                                      param_gmm__n_components == 6) %>%
    select(mean_test_score) %>% as.numeric()
  wine_best <- wine_nn %>% filter(param_NN__alpha == wine_best_alpha, 
                                      param_NN__hidden_layer_sizes == wine_best_hl,
                                      #param_rp__n_components == wine_rp_best,
                                      param_gmm__n_components == 9) %>%
    select(mean_test_score) %>% as.numeric()
  
  cancer_bests <- c(cancer_bests, cancer_best)
  wine_bests <- c(wine_bests, wine_best)
  cancer_overall_bests <- c(cancer_overall_bests, max(cancer_nn$mean_test_score))
  wine_overall_bests <- c(wine_overall_bests, max(wine_nn$mean_test_score))
  
  
  cancer_nn <- read.csv('./output/RF/cancer cluster GMM.csv')
  wine_nn <- read.csv('./output/RF/wine cluster GMM.csv')
  
  cancer_best <- cancer_nn %>% filter(param_NN__alpha == cancer_best_alpha, 
                                      param_NN__hidden_layer_sizes == cancer_best_hl,
                                      #param_filter__n == cancer_rf_best,
                                      param_gmm__n_components == 7) %>%
    select(mean_test_score) %>% as.numeric()
  wine_best <- wine_nn %>% filter(param_NN__alpha == wine_best_alpha, 
                                      param_NN__hidden_layer_sizes == wine_best_hl,
                                      #param_filter__n == wine_rf_best,
                                      param_gmm__n_components == 7) %>%
    select(mean_test_score) %>% as.numeric()
  
  cancer_bests <- c(cancer_bests, cancer_best)
  wine_bests <- c(wine_bests, wine_best)
  cancer_overall_bests <- c(cancer_overall_bests, max(cancer_nn$mean_test_score))
  wine_overall_bests <- c(wine_overall_bests, max(wine_nn$mean_test_score))
  
}

{
  axis_names <- merge(datasets, c('KM', 'GMM')) %$% paste0(x, '\n', y)
  jpeg('./output/plots/wine NN clust dist and base - baseline.jpg', height = 240)  
  
  barplot(wine_bests, names.arg = axis_names, 
          cex.names = .8,
          col = c('gray', rep('darkgreen', 4)),
          ylab = 'Average Validation Accuracy',
          main = 'Wine Quality - Cluster Distance/Probability Only\nNeural Network Performance with Baseline Parameters')
  
  dev.off()
  
  jpeg('./output/plots/cancer NN clust dist and base - baseline.jpg', height = 240)  
  
  barplot(cancer_bests, names.arg = axis_names, 
          cex.names = .8,
          col = c('gray', rep('darkgreen', 4)),
          ylab = 'Average Validation Accuracy',
          main = 'Breast Cancer - Cluster Distance/Probability Only\nNeural Network Performance with Baseline Parameters')
  
  dev.off()
  
  jpeg('./output/plots/wine NN clust dist and base - optimized.jpg', height = 240)  
  
  barplot(wine_overall_bests, names.arg = axis_names, 
          cex.names = .8,
          col = c('gray', rep('darkgreen', 4)),
          ylab = 'Average Validation Accuracy',
          main = 'Wine Quality - Cluster Distance/Probability Only\nNeural Network Performance with Optimized Parameters')
  
  dev.off()
  
  jpeg('./output/plots/cancer NN clust dist and base - optimized.jpg', height = 240)  
  
  barplot(cancer_overall_bests, names.arg = axis_names,
          cex.names = .8,
          col = c('gray', rep('darkgreen', 4)),
          ylab = 'Average Validation Accuracy',
          main = 'Breast Cancer - Cluster Distance/Probability Only\nNeural Network Performance with Optimized Parameters')
  
  dev.off()
}

cancer_bests
wine_bests
cancer_overall_bests
wine_overall_bests





# This gets maximum mean_test_scores for data with cluster assignments
{
  optimal_nn_arch <- c("{'NN__alpha': 0.01, 'NN__hidden_layer_sizes': (12, 12, 12)}", 
                       "{'NN__alpha': 1.0, 'NN__hidden_layer_sizes': (18, 18)}")
  names(optimal_nn_arch) <- c('cancer', 'wine')
  
  cancer_best_hl <- '(12, 12, 12)'
  cancer_best_alpha <- .01
  cancer_bests <- c()
  cancer_overall_bests <- c()
  
  wine_best_hl <- '(18, 18)'
  wine_best_alpha <- 1
  wine_bests <- c()
  wine_overall_bests <- c()
  
  cancer_pca_best <- 6
  cancer_ica_best <- 2
  cancer_rp_best <- 8
  cancer_rf_best <- 6
  
  wine_pca_best <- 9
  wine_ica_best <- 7
  wine_rp_best <- 10
  wine_rf_best <- 7
  
  
  ## K Means
  
  cancer_nn <- read.csv('./output/BASE/cancer cluster KM CLUSTER AND DATA.csv')
  wine_nn <- read.csv('./output/BASE/wine cluster KM CLUSTERS AND DATA.csv')
  
  cancer_best <- cancer_nn %>% filter(param_NN__alpha == cancer_best_alpha, 
                                      param_NN__hidden_layer_sizes == cancer_best_hl, 
                                      param_km__n_components == 5) %>%
    select(mean_test_score) %>% as.numeric()
  wine_best <- wine_nn %>% filter(param_NN__alpha == wine_best_alpha, 
                                      param_NN__hidden_layer_sizes == wine_best_hl,
                                      param_km__n_components == 7) %>%
    select(mean_test_score) %>% as.numeric()
  
  cancer_bests <- c(cancer_bests, cancer_best)
  wine_bests <- c(wine_bests, wine_best)
  cancer_overall_bests <- c(cancer_overall_bests, max(cancer_nn$mean_test_score))
  wine_overall_bests <- c(wine_overall_bests, max(wine_nn$mean_test_score))
  
  
  cancer_nn <- read.csv('./output/PCA/cancer cluster KM CLUSTER AND DATA.csv')
  wine_nn <- read.csv('./output/PCA/wine cluster KM CLUSTERS AND DATA.csv')
  
  cancer_best <- cancer_nn %>% filter(param_NN__alpha == cancer_best_alpha, 
                                      param_NN__hidden_layer_sizes == cancer_best_hl,
                                      #param_pca__n_components == cancer_pca_best,
                                      param_km__n_components == 8) %>%
    select(mean_test_score) %>% as.numeric()
  wine_best <- wine_nn %>% filter(param_NN__alpha == wine_best_alpha, 
                                      param_NN__hidden_layer_sizes == wine_best_hl,
                                      #param_pca__n_components == wine_pca_best,
                                      param_km__n_components == 8) %>%
    select(mean_test_score) %>% as.numeric()
  
  cancer_bests <- c(cancer_bests, cancer_best)
  wine_bests <- c(wine_bests, wine_best)
  cancer_overall_bests <- c(cancer_overall_bests, max(cancer_nn$mean_test_score))
  wine_overall_bests <- c(wine_overall_bests, max(wine_nn$mean_test_score))
  
  
  cancer_nn <- read.csv('./output/ICA/cancer cluster KM CLUSTER AND DATA.csv')
  wine_nn <- read.csv('./output/ICA/wine cluster KM CLUSTERS AND DATA.csv')
  
  cancer_best <- cancer_nn %>% filter(param_NN__alpha == cancer_best_alpha, 
                                      param_NN__hidden_layer_sizes == cancer_best_hl,
                                      #param_ica__n_components == cancer_ica_best,
                                      param_km__n_components == 4) %>%
    select(mean_test_score) %>% as.numeric()
  wine_best <- wine_nn %>% filter(param_NN__alpha == wine_best_alpha, 
                                      param_NN__hidden_layer_sizes == wine_best_hl,
                                      #param_ica__n_components == wine_ica_best,
                                      param_km__n_components == 7) %>%
    select(mean_test_score) %>% as.numeric()
  
  cancer_bests <- c(cancer_bests, cancer_best)
  wine_bests <- c(wine_bests, wine_best)
  cancer_overall_bests <- c(cancer_overall_bests, max(cancer_nn$mean_test_score))
  wine_overall_bests <- c(wine_overall_bests, max(wine_nn$mean_test_score))
  
  
  cancer_nn <- read.csv('./output/RP/cancer cluster KM CLUSTER AND DATA.csv')
  wine_nn <- read.csv('./output/RP/wine cluster KM CLUSTERS AND DATA.csv')
  
  cancer_best <- cancer_nn %>% filter(param_NN__alpha == cancer_best_alpha, 
                                      param_NN__hidden_layer_sizes == cancer_best_hl,
                                      #param_rp__n_components == cancer_rp_best,
                                      param_km__n_components == 4) %>%
    select(mean_test_score) %>% as.numeric()
  wine_best <- wine_nn %>% filter(param_NN__alpha == wine_best_alpha, 
                                      param_NN__hidden_layer_sizes == wine_best_hl,
                                      #param_rp__n_components == wine_rp_best,
                                      param_km__n_components == 6) %>%
    select(mean_test_score) %>% as.numeric()
  
  cancer_bests <- c(cancer_bests, cancer_best)
  wine_bests <- c(wine_bests, wine_best)
  cancer_overall_bests <- c(cancer_overall_bests, max(cancer_nn$mean_test_score))
  wine_overall_bests <- c(wine_overall_bests, max(wine_nn$mean_test_score))
  
  
  cancer_nn <- read.csv('./output/RF/cancer cluster KM CLUSTER AND DATA.csv')
  wine_nn <- read.csv('./output/RF/wine cluster KM CLUSTERS AND DATA.csv')
  
  cancer_best <- cancer_nn %>% filter(param_NN__alpha == cancer_best_alpha, 
                                      param_NN__hidden_layer_sizes == cancer_best_hl,
                                      #param_filter__n == cancer_rf_best,
                                      param_km__n_components == 4) %>%
    select(mean_test_score) %>% as.numeric()
  wine_best <- wine_nn %>% filter(param_NN__alpha == wine_best_alpha, 
                                      param_NN__hidden_layer_sizes == wine_best_hl,
                                      #param_filter__n == wine_rf_best,
                                      param_km__n_components == 5) %>%
    select(mean_test_score) %>% as.numeric()
  
  cancer_bests <- c(cancer_bests, cancer_best)
  wine_bests <- c(wine_bests, wine_best)
  cancer_overall_bests <- c(cancer_overall_bests, max(cancer_nn$mean_test_score))
  wine_overall_bests <- c(wine_overall_bests, max(wine_nn$mean_test_score))
  
  
  
  ## GMM
  
  cancer_nn <- read.csv('./output/BASE/cancer cluster GMM CLUSTER AND DATA.csv')
  wine_nn <- read.csv('./output/BASE/wine cluster GMM CLUSTERS WITH DATA.csv')
  
  cancer_best <- cancer_nn %>% filter(param_NN__alpha == cancer_best_alpha, 
                                      param_NN__hidden_layer_sizes == cancer_best_hl, 
                                      param_gmm__n_components == 5) %>%
    select(mean_test_score) %>% as.numeric()
  wine_best <- wine_nn %>% filter(param_NN__alpha == wine_best_alpha, 
                                      param_NN__hidden_layer_sizes == wine_best_hl,
                                      param_gmm__n_components == 7) %>%
    select(mean_test_score) %>% as.numeric()
  
  cancer_bests <- c(cancer_bests, cancer_best)
  wine_bests <- c(wine_bests, wine_best)
  cancer_overall_bests <- c(cancer_overall_bests, max(cancer_nn$mean_test_score))
  wine_overall_bests <- c(wine_overall_bests, max(wine_nn$mean_test_score))
  
  
  cancer_nn <- read.csv('./output/PCA/cancer cluster GMM CLUSTER AND DATA.csv')
  wine_nn <- read.csv('./output/PCA/wine cluster GMM CLUSTERS WITH DATA.csv')
  
  cancer_best <- cancer_nn %>% filter(param_NN__alpha == cancer_best_alpha, 
                                      param_NN__hidden_layer_sizes == cancer_best_hl,
                                      #param_pca__n_components == cancer_pca_best,
                                      param_gmm__n_components == 8) %>%
    select(mean_test_score) %>% as.numeric()
  wine_best <- wine_nn %>% filter(param_NN__alpha == wine_best_alpha, 
                                      param_NN__hidden_layer_sizes == wine_best_hl,
                                      #param_pca__n_components == wine_pca_best,
                                      param_gmm__n_components == 8) %>%
    select(mean_test_score) %>% as.numeric()
  
  cancer_bests <- c(cancer_bests, cancer_best)
  wine_bests <- c(wine_bests, wine_best)
  cancer_overall_bests <- c(cancer_overall_bests, max(cancer_nn$mean_test_score))
  wine_overall_bests <- c(wine_overall_bests, max(wine_nn$mean_test_score))
  
  
  cancer_nn <- read.csv('./output/ICA/cancer cluster GMM CLUSTER AND DATA.csv')
  wine_nn <- read.csv('./output/ICA/wine cluster GMM CLUSTERS WITH DATA.csv')
  
  cancer_best <- cancer_nn %>% filter(param_NN__alpha == cancer_best_alpha, 
                                      param_NN__hidden_layer_sizes == cancer_best_hl,
                                      #param_ica__n_components == cancer_ica_best,
                                      param_gmm__n_components == 3) %>%
    select(mean_test_score) %>% as.numeric()
  wine_best <- wine_nn %>% filter(param_NN__alpha == wine_best_alpha, 
                                      param_NN__hidden_layer_sizes == wine_best_hl,
                                      #param_ica__n_components == wine_ica_best,
                                      param_gmm__n_components == 9) %>%
    select(mean_test_score) %>% as.numeric()
  
  cancer_bests <- c(cancer_bests, cancer_best)
  wine_bests <- c(wine_bests, wine_best)
  cancer_overall_bests <- c(cancer_overall_bests, max(cancer_nn$mean_test_score))
  wine_overall_bests <- c(wine_overall_bests, max(wine_nn$mean_test_score))
  
  
  cancer_nn <- read.csv('./output/RP/cancer cluster GMM CLUSTER AND DATA.csv')
  wine_nn <- read.csv('./output/RP/wine cluster GMM CLUSTERS WITH DATA.csv')
  
  cancer_best <- cancer_nn %>% filter(param_NN__alpha == cancer_best_alpha, 
                                      param_NN__hidden_layer_sizes == cancer_best_hl,
                                      #param_rp__n_components == cancer_rp_best,
                                      param_gmm__n_components == 6) %>%
    select(mean_test_score) %>% as.numeric()
  wine_best <- wine_nn %>% filter(param_NN__alpha == wine_best_alpha, 
                                      param_NN__hidden_layer_sizes == wine_best_hl,
                                      #param_rp__n_components == wine_rp_best,
                                      param_gmm__n_components == 9) %>%
    select(mean_test_score) %>% as.numeric()
  
  cancer_bests <- c(cancer_bests, cancer_best)
  wine_bests <- c(wine_bests, wine_best)
  cancer_overall_bests <- c(cancer_overall_bests, max(cancer_nn$mean_test_score))
  wine_overall_bests <- c(wine_overall_bests, max(wine_nn$mean_test_score))
  
  
  cancer_nn <- read.csv('./output/RF/cancer cluster GMM CLUSTER AND DATA.csv')
  wine_nn <- read.csv('./output/RF/wine cluster GMM CLUSTERS WITH DATA.csv')
  
  cancer_best <- cancer_nn %>% filter(param_NN__alpha == cancer_best_alpha, 
                                      param_NN__hidden_layer_sizes == cancer_best_hl,
                                      #param_filter__n == cancer_rf_best,
                                      param_gmm__n_components == 7) %>%
    select(mean_test_score) %>% as.numeric()
  wine_best <- wine_nn %>% filter(param_NN__alpha == wine_best_alpha, 
                                      param_NN__hidden_layer_sizes == wine_best_hl,
                                      #param_filter__n == wine_rf_best,
                                      param_gmm__n_components == 7) %>%
    select(mean_test_score) %>% as.numeric()
  
  cancer_bests <- c(cancer_bests, cancer_best)
  wine_bests <- c(wine_bests, wine_best)
  cancer_overall_bests <- c(cancer_overall_bests, max(cancer_nn$mean_test_score))
  wine_overall_bests <- c(wine_overall_bests, max(wine_nn$mean_test_score))
  
}


{
  axis_names <- merge(datasets, c('KM', 'GMM')) %$% paste0(x, '\n', y)
  jpeg('./output/plots/wine NN data and clust and base - baseline.jpg', height = 240)  
  
  barplot(wine_bests, names.arg = axis_names, 
          cex.names = .8,
          col = c('gray', rep('darkgreen', 4)),
          ylab = 'Average Validation Accuracy',
          main = 'Wine Quality - Data + Clusters\nNeural Network Performance with Baseline Parameters')
  
  dev.off()
  
  jpeg('./output/plots/cancer NN data and clust and base - baseline.jpg', height = 240)  
  
  barplot(cancer_bests, names.arg = axis_names, 
          cex.names = .8,
          col = c('gray', rep('darkgreen', 4)),
          ylab = 'Average Validation Accuracy',
          main = 'Breast Cancer - Data + Clusters\nNeural Network Performance with Baseline Parameters')
  
  dev.off()
  
  jpeg('./output/plots/wine NN data and clust and base - optimized.jpg', height = 240)  
  
  barplot(wine_overall_bests, names.arg = axis_names, 
          cex.names = .8,
          col = c('gray', rep('darkgreen', 4)),
          ylab = 'Average Validation Accuracy',
          main = 'Wine Quality - Data + Clusters\nNeural Network Performance with Optimized Parameters')
  
  dev.off()
  
  jpeg('./output/plots/cancer NN data and clust and base - optimized.jpg', height = 240)  
  
  barplot(cancer_overall_bests, names.arg = axis_names,
          cex.names = .8,
          col = c('gray', rep('darkgreen', 4)),
          ylab = 'Average Validation Accuracy',
          main = 'Breast Cancer - Data + Clusters\nNeural Network Performance with Optimized Parameters')
  
  dev.off()
}

cancer_bests
wine_bests
cancer_overall_bests
wine_overall_bests