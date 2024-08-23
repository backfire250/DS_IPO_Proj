# Predicting listing gains in the Indian IPO market using Tensorflow
Repo for my data science study for predicting IPO listing gains


**Project Goal**:  Build a deep learning classification model to determine if there will be listing gains for the IPO (% increase in the share price of a company from its IPO issue price on the day of listing).
* Aim to determine how specific characteristics in our dataset can be used to accurately predict whether or not an IPO will list a profit
* Use multiple machine learning techniques to predict outcomes based on provided data
* Build a sequential API using tensorflow as part of our predictive models

## Project Steps
1. Data Collection
2. Data Cleaning
3. Outlier Detection
4. Data Standardization
5. Exploratory Data Analysis
6. Feature Selection
7. Model Building
8. Model Evaluation

## Code and Resources Used  
**Python Version:** 3.7   
**Packages:** pandas, numpy, sklearn, matplotlib, seaborn, tensorflow
**Jupyter Notebooks**  

## Data Collection
I collected the data from the website Moneycontrol. The dataset we're using contains information on Indian IPOs from the year 2010 and includes information such as issue size, issue price, listing gains, and subscription totals.

## Data Cleaning
After taking an initial look at the data, I needed to clean it up so that it was usable for our model. I made the following changes after some initial evaluation:

*	Converted string values to numerical values for use in our models 
*	Dropped columns that did not seem relevant to predicting the amount of area damaged by forest fires
*	Found that our target column was skewed towards 0, so we applied a log transform to that particular column
*	There were 8 columns in our dataset that were missing data, so we used KNN imputation to fill them in

## Outlier Detection
Using boxplots, I was able to identify the number of outliers in each of our predictor columns. All 4 of our predictor columns had a decent amount of outliers. I decided to deal with them by setting the outliers to their upper or lower boundaries for each predictor variable.

## Data Standardization
I wanted to standardize our predictor variables so that they would all have a value between 0 and 1. I took our data, created an array for both the target variable and the predictor variables, then scaled our predictor variables so that they would all appear between 0 and 1. I then verified what I had done to make sure the scaling worked properly.

## EDA (Exploratory Data Analysis)
I looked at the correlation between my continuous variables and my target variable. Below are a couple highlights from my EDA.

![]([https://github.com/backfire250/Alex_Portfolio/blob/main/images/ipo_ss_bp.png)
![]([https://github.com/backfire250/Alex_Portfolio/blob/main/images/ipo_ss_hm.png)
![]([https://github.com/backfire250/Alex_Portfolio/blob/main/images/ipo_ss_main.png)

## Feature Selection
After cleaning our dataset and doing some initial data analysis, we could move on to selecting features for our model. There weren't too many variables to choose from in this data set, but looking at the correlations between each of the features and the target variable, I was able to conclude that about 4 variables had a decent correlation with our target. I decided to do my exploratory data analysis around these 4 predictors, and ignore the variables for date, issue size, and issue price.

## Model Building
I split the data into train and test sets with a test size of 30%.

I tried many different models using a sequential API and evaluated them using Mean Squared Error (the average variance between actual values and projected values in the dataset).

I tried different activation functions:
*    **ELU** - Used as the baseline for my model.
*    **ReLU** - I wanted to see how much of an impact a change in activation function could have.
*    **Mix** - I wanted to try a mix of ELU and ReLU activation functions to see if I could further optimize my model.
*    **Sigmoid** - After thinking more about the problem, I decided to try a sigmoid activation function for my output layer of my model.

I tried different amounts of nodes for my hidden layers:
*    **500/300/100/20** - Used as the baseline for my model.
*    **32/16/8/4** - I wanted to try my same models but with far fewer nodes to measure the impact it would have on my results.

I tried 3 different learning rates:
*    **0.1** - Used as the baseline for my model.
*    **0.01** - I wanted to experiment with a different array of values to see if I could optimize my accuracy.
*    **0.001** - Further experimentation

I tried different optimizers:
*    **Adam optimizer** - Used as the baseline for my model.
*    **SGD optimizer** - I wanted to try SGD here since SGD moves through the weights and updates them based on how well they are performing.
*    **RMSprop** - Root Mean Square Propagation keeps track of the moving average of the squares of the gradients and dividing this by the square root of the sum of all previous squared gradients. This would allow for faster learning for models with many training examples. Although my data did not have a ton of training examples, I wanted to experiment to see what the data would show me.

## Model Evaluation
After trying many different combinations of optimizers, learning rates, activation functions, and number of nodes, I needed to take a break and come back to the project. I wasn't satisfied with the scores I was seeing and needed to think about the data some more. After giving it some thought, I realized I should consider trying a different loss function when compiling my models - one for binary classifications. This makes sense after all - the value we're trying to predict is a binary value. Because of this, I switched to using a sigmoid activation function for the output layer in my model. I also switched my loss function in my compiler to BinaryCrossEntropy. This gave me much better results with my model.

My 500 node model ended up producing better results than my 32 node model.
*    **Training accuracy** : 83%
*    **Test accuracy**: 64%

## Conclusion
Interestingly, as I decreased the learning rate on this model, my training accuracy went up while my test accuracy went slightly down:

Learning rate 0.1:
  Train accuracy: 0.67, test accuracy: 0.65
Learning rate 0.01:
  Train accuracy: 0.79, test accuracy: 0.67
Learning rate 0.001:
  Train accuracy: 0.83, test accuracy: 0.64
  
I'm slightly concerned about overfitting with this model since the train and test accuracies got further apart as I decreased the learning rate. I think that I could definitely optimize this model more if I play around with it.
