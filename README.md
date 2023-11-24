# Classification-Model-Comparison-and-Improvement
This project is about credit risk measurement for a bank. The project entailed a comprehensive analysis of client default tendencies relative to their backgrounds using advanced classification models, providing actionable insights by model comparison and refinement.

## Project Structure
The project is organized into the following components:

 - <strong>'credit_risk_dataset.csv': </strong> The dataset overall contains 12 columns with information about clients' backgrounds and some traits of their loans. There are 32573 data points in total. <br>
 - <strong> 'Data Analysis.R': </strong> The R code contains the entire steps taken in the data analysis process.<br>
_Data Preprocessing_  - First, drop the rows with null values. Second, use synthetic data generation to deal with the unbalanced data. Third, apply correlation tests and principal component analysis. <br>
_Model Analysis_ - Conduct a suite of 5 classification models (Logistic Regression, KNN, SVM, Decision Tree, Random Forest) to the data with/without PCA, respectively. <br>
_Model Comparison_ - Compare the model performance based on various metrics (accuracy, sensitivity, specificity, ROC, AUC, F1-score). <br>
_Model Combination_ - Combine and improve the model by integrating the top 3 performing models using Boosting techniques.

 - <strong> 'Paper.pdf': </strong> The final paper "Classification Model Comparison and Improvement Regarding Credit Risk". <br>
 
 - <strong> 'Slide.pdf': </strong> The slide presented at the Annual Conference of Financial and Banking Perspectives. 

## Conclusions
According to the model comparison, KNN has the highest sensitivity and Random Forest has the highest of both specificity and accuracy. By integrating the top 3 performing models using Boosting techniques, we finally construct a model with an accuracy rate of 93%.

## Contact Information
If you have any questions, please contact [simin.yu@columbia.edu].
