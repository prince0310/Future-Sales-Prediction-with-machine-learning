# Future-Sales-Prediction-with-machine-learning

Predicting the future sales of a product helps a business manage the manufacturing and advertising cost of the product. There are many more benefits of predicting the future sales of a product. So if you want to learn to predict the future sales of a product with machine learning, this article is for you. In this article, I will take you through the task of future sales prediction with machine learning using.

The dataset given here contains the data about the sales of the product. The dataset is about the advertising cost incurred by the business on various advertising platforms. Below is the description of all the columns in the dataset:

* Tv: Advertising cost spent in dollars for advertising on TV;
* Radio: Advertising cost spent in dollars for advertising on Radio;
* Newspaper: Advertising cost spent in dollars for advertising on Newspaper;
* Sales: Number of units sold;

So, in the above dataset, the sales of the product depend on the advertisement cost of the product. I hope you now have understood everything about this dataset. Now in the section below, I will take you through the task of future sales prediction with machine learning using Python.

Let’s start the task of future sales prediction with machine learning by importing the necessary Python libraries and the dataset:

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```
![data](https://user-images.githubusercontent.com/85225054/233142259-795d9d36-275e-4606-9e99-a9eed81f6b8d.png)

Let’s have a look at whether this dataset contains any null values or not:
```
df.info()
```
![info](https://user-images.githubusercontent.com/85225054/233142747-e92d7800-7817-446f-8f82-fc720d922094.png)

```
df.describe()
```
![describe](https://user-images.githubusercontent.com/85225054/233142890-da19ca73-26bd-4807-847e-ca27f22d3885.png)

So this dataset doesn’t have any null values. Now let’s visualize the relationship between the amount spent on advertising on TV and units sold:
```
import plotly.express as px
import plotly.graph_objects as go
figure = px.scatter(data_frame=df, x="Sales",
                    y="TV", size="TV", trendline="ols")
figure.show()
```
![newplot](https://user-images.githubusercontent.com/85225054/233143564-fadd3029-1d4d-4719-b19c-2da6769df427.png)

```
import plotly.express as px
import plotly.graph_objects as go
figure = px.scatter(data_frame=df, x="Sales",
                    y="Radio", size="Radio", trendline="ols")
figure.show()
```

![newplot1](https://user-images.githubusercontent.com/85225054/233143662-166439a8-986f-4aae-904c-a820fa6cf617.png)


```
import plotly.express as px
import plotly.graph_objects as go
figure = px.scatter(data_frame=df, x="Sales",
                    y="Newspaper", size="Newspaper", trendline="ols")
figure.show()
```

![newplot2](https://user-images.githubusercontent.com/85225054/233143732-e77b46cc-af0d-4538-9ae1-277b1dce4864.png)

Out of all the amount spent on advertising on various platforms, I can see that the amount spent on advertising the product on TV results in more sales of the product. Now let’s have a look at the correlation of all the columns with the sales column:

```
correlation = df.corr()
print(correlation["Sales"].sort_values(ascending=False))
sns.heatmap(correlation, annot=True)

```

![corr](https://user-images.githubusercontent.com/85225054/233144172-1ffcacf2-fc47-40a6-9a8f-409656db75fa.png)

import required libraries

```
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
```
Now in this section, I will train a machine learning model to predict the future sales of a product. But before I train the model, let’s split the data into training and test sets:
```
x = np.array(data.drop(["Sales"], 1))
y = np.array(data["Sales"])
xtrain, xtest, ytrain, ytest = train_test_split(x, y, 
                                                test_size=0.2, 
                                                random_state=42)
```
Now let’s train the model to predict future sales:

```
model = LinearRegression()
model.fit(xtrain, ytrain)
print(model.score(xtest, ytest))

```
```
16.95475874
```

Save the model by dumping using pickle 

```
# save model
import pickle
pickle.dump(model, open('model.pkl', 'wb'))
```

Before Prediction the web page looks 

![before_prediction](https://user-images.githubusercontent.com/85225054/234588531-a9687540-2aab-4246-91ec-b9c2ad62c92e.png)

After Prediction web page shows the prediction 

![after_predictiuoin](https://user-images.githubusercontent.com/85225054/234588744-875ac51c-d040-4c65-969e-77ea7fd165a3.png)



