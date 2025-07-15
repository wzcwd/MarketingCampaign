# Maximizing Profit for Marketing Campaigns

## 1. Business Problem & Objective

The primary objective of this project is to build a predictive model to maximize the profit of an upcoming direct 
marketing campaign. This campaign aims to sell a new gadget to its customer base.

To achieve this, we leverage the pilot campaign data involving 2240 customers to build a model that can predict customer responses. By identifying and targeting only the customers most likely to purchase the gadget, the company can "cherry-pick" the customers that are most likely to purchase the offer while leaving out the non-respondents, making the incoming campaign highly profitable.

Beyond profit maximization, the Chief Marketing Officer (CMO) is also interested in understanding the key 
characteristics of customers who are willing to buy the new product.

## 2. Pilot Campaign Analysis

A pilot campaign was conducted to gather data for building the predictive model.

- **Sample Size**: 2,240 customers were selected randomly and contacted.
- **Colum**: 29 columns.
- **Total Cost**: 6,720 MU.
- **Total Revenue**: 3,674 MU.
- **Total Profit/Loss**: -3,046 MU
- **Success Rate**: 15% of customers accepted the offer.

The low success rate highlights the need for a data-driven approach to select the target audience for the next campaign.

## 3. Project Workflow

This project follows a standard machine learning pipeline to build the model

1.  **Data Preprocessing**: Cleans and transforms the raw data from the pilot campaign, handling missing values, encoding categorical features, and preparing it for modeling.
2.  **Model Training**: Trains 4 out-of-the-box classification models (Decision Tree, Random Forest, AdaBoost, XGBoost) to predict target customers.
3.  **Hyperparameter Tuning**: Utilizes `GridSearchCV` to find the optimal hyperparameters for each model, aiming to improve their predictive power.
4.  **Performance Evaluation**: Compares the tuned models against the baseline models on a  testing set using various metrics (Accuracy, Precision, Recall, F1-Score, ROC AUC).
5.  **Feature Importance for Interpretability**: Extracts and visualizes the most influential features from the best-performing models. This provides insights into the characteristics of target customers.

## 4. Run the project

1.  **Set up the environment and install dependencies.**
2.  **Run the main.py**
    ```bash
    python src/main.py
    ```
3.  **Check the results:**
    -   Performance metrics will be printed to the console.
    -   Feature importance plots for baseline and tuned models will be saved in the `results/baseline` and `results/tuned` directories, respectively.

## 5. Dataset

The dataset contains information from the pilot marketing campaign.
- **Source**: [Kaggle Marketing Campaign Dataset](https://www.kaggle.com/datasets/rodsaldanha/arketing-campaign/data)

### Feature Description

-   **`AcceptedCmp1`**: 1 if customer accepted the offer in the 1st campaign, 0 otherwise
-   **`AcceptedCmp2`**: 1 if customer accepted the offer in the 2nd campaign, 0 otherwise
-   **`AcceptedCmp3`**: 1 if customer accepted the offer in the 3rd campaign, 0 otherwise
-   **`AcceptedCmp4`**: 1 if customer accepted the offer in the 4th campaign, 0 otherwise
-   **`AcceptedCmp5`**: 1 if customer accepted the offer in the 5th campaign, 0 otherwise
-   **`Response` (Target)**: 1 if customer accepted the offer in the last campaign, 0 otherwise
-   **`Complain`**: 1 if customer complained in the last 2 years
-   **`DtCustomer`**: Date of customer’s enrolment with the company
-   **`Education`**: Customer’s level of education
-   **`Marital`**: Customer’s marital status
-   **`Kidhome`**: Number of small children in customer’s household
-   **`Teenhome`**: Number of teenagers in customer’s household
-   **`Income`**: Customer’s yearly household income
-   **`MntFishProducts`**: Amount spent on fish products in the last 2 years
-   **`MntMeatProducts`**: Amount spent on meat products in the last 2 years
-   **`MntFruits`**: Amount spent on fruit products in the last 2 years
-   **`MntSweetProducts`**: Amount spent on sweet products in the last 2 years
-   **`MntWines`**: Amount spent on wine products in the last 2 years
-   **`MntGoldProds`**: Amount spent on gold products in the last 2 years
-   **`NumDealsPurchases`**: Number of purchases made with discount
-   **`NumCatalogPurchases`**: Number of purchases made using catalogue
-   **`NumStorePurchases`**: Number of purchases made directly in stores
-   **`NumWebPurchases`**: Number of purchases made through company’s web site
-   **`NumWebVisitsMonth`**: Number of visits to company’s web site in the last month
-   **`Recency`**: Number of days since the last purchase
