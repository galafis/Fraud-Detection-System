# Advanced Statistical Analysis in R
# Customer Behavior Analytics - R Implementation
# Author: Gabriel Demetrios Lafis

# Load required libraries
library(ggplot2)
library(dplyr)
library(cluster)
library(factoextra)
library(corrplot)
library(plotly)

# Generate sample customer data
set.seed(123)
customers <- data.frame(
  customer_id = 1:1000,
  age = sample(18:80, 1000, replace = TRUE),
  income = rnorm(1000, 50000, 15000),
  spending_score = sample(1:100, 1000, replace = TRUE),
  frequency = sample(1:50, 1000, replace = TRUE),
  recency = sample(1:365, 1000, replace = TRUE),
  monetary = rnorm(1000, 500, 200)
)

# RFM Analysis
rfm_analysis <- function(data) {
  # Calculate RFM scores
  data$R_score <- ntile(desc(data$recency), 5)
  data$F_score <- ntile(data$frequency, 5)
  data$M_score <- ntile(data$monetary, 5)
  
  # Create RFM segments
  data$RFM_score <- paste0(data$R_score, data$F_score, data$M_score)
  
  return(data)
}

# Perform RFM analysis
customers_rfm <- rfm_analysis(customers)

# Customer Segmentation using K-means
perform_clustering <- function(data) {
  # Prepare data for clustering
  cluster_data <- data[, c("age", "income", "spending_score", "frequency")]
  cluster_data <- scale(cluster_data)
  
  # Determine optimal number of clusters
  wss <- sapply(1:10, function(k) {
    kmeans(cluster_data, k, nstart = 10)$tot.withinss
  })
  
  # Perform K-means clustering
  k <- 4  # Optimal clusters
  clusters <- kmeans(cluster_data, k, nstart = 25)
  
  data$cluster <- clusters$cluster
  return(list(data = data, clusters = clusters))
}

# Perform clustering
clustering_result <- perform_clustering(customers_rfm)
customers_final <- clustering_result$data

# Advanced Visualizations
create_visualizations <- function(data) {
  # Age vs Income by Cluster
  p1 <- ggplot(data, aes(x = age, y = income, color = factor(cluster))) +
    geom_point(alpha = 0.7) +
    labs(title = "Customer Segmentation: Age vs Income",
         x = "Age", y = "Income", color = "Cluster") +
    theme_minimal()
  
  # Spending Score Distribution
  p2 <- ggplot(data, aes(x = spending_score, fill = factor(cluster))) +
    geom_histogram(bins = 20, alpha = 0.7) +
    facet_wrap(~cluster) +
    labs(title = "Spending Score Distribution by Cluster",
         x = "Spending Score", y = "Frequency") +
    theme_minimal()
  
  # RFM Heatmap
  rfm_summary <- data %>%
    group_by(cluster) %>%
    summarise(
      avg_recency = mean(recency),
      avg_frequency = mean(frequency),
      avg_monetary = mean(monetary)
    )
  
  return(list(p1 = p1, p2 = p2, rfm_summary = rfm_summary))
}

# Generate visualizations
plots <- create_visualizations(customers_final)

# Statistical Analysis
statistical_analysis <- function(data) {
  # Correlation analysis
  cor_matrix <- cor(data[, c("age", "income", "spending_score", "frequency", "monetary")])
  
  # ANOVA for clusters
  anova_results <- list(
    age = aov(age ~ factor(cluster), data = data),
    income = aov(income ~ factor(cluster), data = data),
    spending = aov(spending_score ~ factor(cluster), data = data)
  )
  
  return(list(correlation = cor_matrix, anova = anova_results))
}

# Perform statistical analysis
stats <- statistical_analysis(customers_final)

# Predictive Modeling
library(randomForest)
library(caret)

build_predictive_model <- function(data) {
  # Prepare data for modeling
  model_data <- data[, c("age", "income", "frequency", "recency", "monetary", "spending_score")]
  
  # Split data
  set.seed(123)
  train_index <- createDataPartition(model_data$spending_score, p = 0.8, list = FALSE)
  train_data <- model_data[train_index, ]
  test_data <- model_data[-train_index, ]
  
  # Build Random Forest model
  rf_model <- randomForest(spending_score ~ ., data = train_data, ntree = 100)
  
  # Make predictions
  predictions <- predict(rf_model, test_data)
  
  # Calculate performance metrics
  rmse <- sqrt(mean((test_data$spending_score - predictions)^2))
  mae <- mean(abs(test_data$spending_score - predictions))
  
  return(list(model = rf_model, rmse = rmse, mae = mae, predictions = predictions))
}

# Build predictive model
model_results <- build_predictive_model(customers_final)

# Export results
write.csv(customers_final, "customer_segments.csv", row.names = FALSE)

# Print summary
cat("Customer Behavior Analytics - R Analysis Complete\n")
cat("Total customers analyzed:", nrow(customers_final), "\n")
cat("Number of clusters:", length(unique(customers_final$cluster)), "\n")
cat("Model RMSE:", round(model_results$rmse, 2), "\n")
cat("Model MAE:", round(model_results$mae, 2), "\n")

# Save plots
ggsave("age_income_clusters.png", plots$p1, width = 10, height = 6)
ggsave("spending_distribution.png", plots$p2, width = 12, height = 8)

print("Analysis complete! Check generated files for detailed results.")

