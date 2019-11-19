# Corn futures price prediction model (“Model”)

## Technical parameters

### Assumptions in the Model
1. The Model is pre-trained with 10years worth of data and additional noise added to help counter biases. There is no need to fit the Model to the Data;
2. The Model currently makes predictions 10 days ahead of the actual historical data; and
3. The Model is designed to be flexible in terms of prediction days (see item 2) and feature count. The Model can use more features than listed above. current version has no option for the user to change the prediction days value from 10.

### Technical architecture

The Model is an LSTM/feed forward neural network combination. Hyper-parameters are tuned with bayesian algorithm to optimise the model architecture and run-time.
The training also took into consideration a noisy data as well as other regularisation techniques to minimise the effect of biases.
Model has function built-in to format the input Data to a relevant tensor for predictions.

## Input

### Input information (“Data”)

https://www.quandl.com/data/CHRIS/CME_C5

### Details of the Data

The ZC1! Continuous corn futures data is related to CBOT (Chicago commodity stock exchange) tradings based on the North American production and demand.
The Data is a daily break down of the continuous futures prices with additional features which are helpful for prediction purposes.

<# align="center">
  <img width="460" height="300" src=pictures/Corn.jpg">
</#>


#### Data structure and acceptable format

|  Column    |     Decription            |         Acceptable Format  |    Comment|
|------------|---------------------------|----------------------------|-----------|
|Open        |    Opening price of the day |       Float or Int       |  
|High        |    Highest price of the day |       Float or Int       |
|Low         |    Lowest price of the day  |       Float or Int       |
|Last        |    Closing price of the day |       Float or Int       |     Target value to predict
|Volume      |    Trading volume in ‘000   |       Float or Int       |     Needs to be formatted before loading into the model.

Rest of the Data is not essential for this model, but it can dynamically handle it.
Data frequency is daily.

## Output
Output is a python array of predictions (10 data points) of the Last price for the next 10 trading days.

## API call
TO BE UPDATED
