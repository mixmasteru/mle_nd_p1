# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary

### Data
This dataset contains data about a marketing campaign of a bank. Its contains information about persons such as 
job, marital, education, housing, loan and some more, 21 in total. The data is not complete and contains
many "unknown" fields.  It was released by [kaggle](https://www.kaggle.com/henriqueyamahata/bank-marketing)

We seek to predict "y" which answers "Has the client subscribed a term deposit?" (binary: 'yes', 'no')

### Best model
The best performing model was the "VotingEnsemble" model created by automl with accuracy of 0.9176357880688271

![hyper](img/automl3.png)

## Scikit-learn Pipeline

### Explanation of the pipeline architecture, including data, hyperparameter tuning, and classification algorithm.
 
* load data into TabularDatasetFactory from provided URL
 * cleanup data e.g. drop job 
 * replace strings of month/weekday by int values
 * replace yes/no by 1/0 in data
 * split data in train and validation
 * use LogisticRegression as classification algorithm
 * use C (inverse of regularization strength) and max_iter (max iteration number) as hyperparameters'
 * run training
 * dump model

**What are the benefits of the parameter sampler you chose?**
I used RandomParameterSampling because I needed a way to put the hyperparameters c and max_iter to
the train scrip. It's also easy to use as you can specify the hyperparameter range very easily.

**What are the benefits of the early stopping policy you chose?**
I used the BanditPolicy as I can provide a threshold (slack_factor) for the allowable slack as a ration 
but also can use the delay_evaluation to avoid premature termination of training runs.

![hyper](img/hyper.png)

## AutoML
The best model find by AutoML was the "VotingEnsemble" model with an accuracy of 
0.9176357880688271 with 
 * training_type: "MeanCrossValidation"
 * primary_metric: "accuracy"


![hyper](img/automl2.png)

## Pipeline comparison
* booth approaches need a cleanup of the data beforehand
* the setup for automl is much shorter than for HyperDrive
* AutoML needed longer to find the model
* accuracy of both is nearly the same
* AutoML shows thar duration is an important feature

![hyper](img/automl1.png)

## Future work
* finetune the found parameters by HyperDrive more
* use other parameters for HyperDrive like learning_rate or batch_size
