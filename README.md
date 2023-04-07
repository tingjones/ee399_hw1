
<h1><br>Introduction to Curve Fitting<br></h1>
<h2>• EE 399 • SP 23 • Ting Jones</h2>
<h3>Abstract</h3>
<p>The first assignment for EE 399, Introduction to Machine Learning for Science and Engineering, involves fitting various models onto the given raw data. Models are optimized through minimizing the least-squares error:</p>

<img src="https://media.discordapp.net/attachments/847715688273281034/1093792980327989268/image.png" alt="Least-Squares Error Equation">

>Fig. 1. Least-Squares Error Equation
<p>Functions include the cosine wave, a line, a parabola, and a 19th degree polynomial, which include parameters that are minimized on the training data to attempt to model the test data. See <a href="#Computational Results">Computational Results</a> for the outcome.</p>

<h3>Table of Contents</h3>
<p> •&emsp;<a href="#Introduction and Overview">Introduction and Overview</a></p>
<p> •&emsp;<a href="#Theoretical Background">Theoretical Background</a></p>
<p> •&emsp;<a href="#Algorithm Implementation and Development">Algorithm Implementation and Development</a></p>
<p> •&emsp;<a href="#Computational Results">Computational Results</a></p>
<p> •&emsp;<a href="#Summary and Conclusions">Summary and Conclusions</a></p>

## Introduction and Overview
To attempt describing any given data, a model is built on trained data by minimizing the difference, or error, between the model and the training data. After training, the model is implemented on the test dataset to see if it can closely predict the values for the given data. Results are dependent on the size of the dataset, the initial prediction of the value of the optimal parameters for the model, and other factors.
In this assignment, the models were optimized through minimizing the least-squares error (Fig. 1).</p>
<p>Models were made from the functions for the cosine wave, a line, a parabola, and a 19th degree polynomial. Each term had a coefficient, or parameter, that was calculated to give the least-squares error for the training data.

## Theoretical Background


## Algorithm Implementation and Development
The procedure is discussed in this section. For the results, see <a href="#Computational Results">Computational Results</a>

The given data for this assignment is below, which is used in all four problems.
```
x = np.arange(0, 31)
y = np.array([30, 35, 33, 32, 34, 37, 39, 38, 36, 36, 37, 39, 42, 45, 45, 41,
              40, 39, 42, 44, 47, 49, 50, 49, 46, 48, 50, 53, 55, 54, 53])
```

### Problem 1
The minimum error for the cosine function (Fig. 2) was found by defining a function that evaluates the error of the function at each point of given data.

![image](https://user-images.githubusercontent.com/114551272/230567103-a3a254bb-7c5f-477e-9e71-94e4189d37ce.png)
> Fig. 2. Cosine Function

This function was sent through the scipy minimize function, where parameters A, B, C, and D are returned for a minimum error value.
```
def fit_er(c, x, y):
    model = c[0]*np.cos(c[1]*x)+c[2]*x + c[3] # define and evaluate cosine function at current point
    er = np.sqrt(np.sum((model-y)**2)/len(x)) # evaluate error
    return er

# prediction of parameter values to initiate optimiziation
min0 = np.array([3, np.pi/4, 2/3, 32])
res = opt.minimize(fit_er, min0, args=(x, y), method='Nelder-Mead')
```

After optimization, the found parameters were substituted in the cosine function, and the overall equation was graphed against the raw data.

### Problem 2
For this problem, two parameters from the optimized parameters from problem 1 are to be fixed, with the other two parameters being swept through various values. Every combination of two fixed and two swept parameters are used to find the error of the model with these modified parameters against the raw data.

To implement this, a function was made to cycle through each index of the swept parameters and return the error. The bulk of this function is given below.
```
# since checking in order of a, b, c, d, only one of sweep parameters
    # needs to be known to control index (selected with i instead of j)
# returns 2d array of calculated errors
for i in range(0, L):
    for j in range(0, L):
        c_fix[3] = d[j]
        if (sweep == 0): # sweep a
            c_fix[0], c_fix[1], c_fix[2] = a[i], b[j], c[j]
        elif (sweep == 1): # sweep b
            c_fix[0], c_fix[1], c_fix[2] = a[j], b[i], c[j]
        elif (sweep == 2): # sweep c
            c_fix[0], c_fix[1], c_fix[2] = a[j], b[j], c[i]
        err_fix[i][j] = fit_er(c_fix, x, y)
return err_fix
```
The 2D array of errors resulting from this function were then plotted using the matplot pcolor module, which can visualize the minima of the errors for the range of swept parameter values.

### Problem 3
This problem involved fitting a line, a parabola, and a 19th degree polynomial to the first 20 data points of the given raw data. These data points are therefore the training data for each of the three models.

For the line and parabola, a similar implementation was used as the cosine function in problem 1. A function was made to represent the objective function and the least-squares error evaluated with that function is returned.
```
# define the objective functions for line (Ax + B) and parabola (Ax^2 + Bx + C)
def fit_line(c, x, y):
    model = c[0] * x + c[1]
    er = np.sqrt(np.sum((model-y)**2)/len(x))
    return er

def fit_parab(c, x, y):
    model = c[0] * x ** 2 + c[1] * x + c[2]
    er = np.sqrt(np.sum((model-y)**2)/len(x))
    return er
```
These functions were then sent through the scipy minimize function to find the parameters for the functions that would give the least error. Results for substituting the parameters back into the function and evaluating at each training data point were then graphed.

For the polynomial, numpy already includes methods to handle polynomials, especially to easily manage and evaluate one to the 19th degree. Therefore, these methods were used to fit the polynomial to the raw data through
```
res_poly = np.polyfit(x_train, y_train, 19, full=True)
```
where (x_train, y_train) is the training data, 19 is the degree, and full=True means that the 0th index of res_poly will contain our optimized parameters.

After retrieving the optimized parameters in ```mins_poly```, the 19th degree polynomial can be evaluated at each of the training data points through
```
np.polyval(mins_poly, x_train)
```
and then graphed on the training data.

Following training, the models are then used to find the error from the testing dataset, which are the last 10 data points. This is done by maintaining the parameters found by the optimized models fitted to the training datset and evaluating each of the functions at each test data point. Then, the result is substituted into the least-squares error equation. For the line and parabola, the error was evaluated by returning the value from the ```fit_line``` and ```fit_parabola``` functions, while the polynomial was reevaluated with ```np.polyval``` but with ```x_test``` instead of ```x_train```.

### Problem 4
Here, the process from problem 3 is repeated, but with the first and last 10 data points used as training data, with the remaining middle as our testing dataset.

Again, the functions for the line and parabola were used to optimize the parameters based on the training data, returning the error. The polynomial was fitted using the ```polyfit``` function, and evaluated at each data poitn with ```polyval```. Parameters found by optimization are used to evaluate the model on the test dataset.

## Computational Results
### Problem 1
The cosine model roughly appears to follow the given data as the returned optimized parameters were substituted into the cosine function and a graph is generated (Fig. 3)
<p><img src="https://media.discordapp.net/attachments/847715688273281034/1093804058076270592/zyCXKlaN169YEBgayePFiq0MTERdT15KIeJSEhAQmTpzIzJkzCQgIwMvLi5kzZ7J69WomT55sdXgi4mJqkRERERGPpRYZERER8VhKZERERMRjKZERERERj6VERkRERDyWEhkRERHxWEpkRERExGMpkRERERGPpURGREREPJYSGREREfFYSmRERETEYymREREREYlREZEREQ81v8DGuq7Z2zupEAAAAAASUVORK5CYII.png" width=400></p>

> Fig. 3. Least-Squares Fit with Cosine Function

For this model, the parameters were found to be:
```
A: 2.1716818723637914
B: 0.9093249029166655
C: 0.7324784894461773
D: 31.45291849616531
```
with an error of: ```1.592725853040056```

### Problem 2
There are six possible combinations (without repeats) for fixing two and sweeping two parameters and calculating the resulting error, being AB, AC, AD, BC, BD, CD. The 2D error landscape for these combinations are illustrated in Fig. 4.
![v5Q9UBwNYi8xAsNHAIqHPnzmlpaUkvXrxY9W99fX3KycnR8PCw0tLSDFQHAFuLzEOw0MABAADYDHvgAAAAbIYGDgAAwGZo4AAAAGyGBg4AAMBmaOAAAABshgYOAADAZmjgAAAAbIYGDgAAwGZo4AAAAGyGBg4AAMBmaOAAAABshgYOAADAZv4LozTcM8lbT6oAAAAASUVORK5CYII](https://user-images.githubusercontent.com/114551272/230568691-3ad852b1-0d53-4986-a690-21b72795a070.png)
> Fig. 4. 2D Loss Landscape for Combinations of Fixed and Swept Parameters

The minimum of the error for each combination is found along the dark blue region, therefore related to specific regions of the swept parameters for each of the fixed parameters.

### Problem 3
Using the first 20 data points used as training data, the line, parabola, and 19th degree polynomial were fitted to this data. The results of the training are illlustrated on the left of Fig. 5, with the same model used on all data points, including the test dataset, on the right.

<p><img src="https://media.discordapp.net/attachments/847715688273281034/1093809017899450449/sXQAAAABJRU5ErkJggg.png" width=400</img><img src="https://media.discordapp.net/attachments/847715688273281034/1093809617651380244/TxEulk3tnEnwAAAABJRU5ErkJggg.png" width=400</img></p>

> Fig. 5. Model fits on training dataset and then demonstrated on test dataset

The least square error of these models on the training data and test data are given below:
```
LSE for Line:
Training: 2.2427493869452952
Test: 3.3636611831800165 

LSE for Parabola:
Training: 2.1255393484769733
Test: 8.713692925760776 

LSE for 19th degree Polynomial:
Training: 0.02835144302630829
Test: 28626352734.190914
```

### Problem 4
The process for problem 3 is repeated. Fig. 6 illustrates the function fitted to the training data points on the left, and then the trained model on the right with test data points.
<p><img src="https://media.discordapp.net/attachments/847715688273281034/1093809847406956635/t73ucAgG2Cv3giBQX09fUVDw8P0el0otlsxmg0it1uF7e3t3mfBkBJ2SY4J6SggAaDQUyn09hut1Gr1aLf70ej0YjX19e8TwOgpGwTnPPXPiiY5XIZo9EoJpNJ1Ov1qFQqMZlMYrVaxXg8zvs8AErINsFvnkgBAAAk8kQKAAAgkZACAABIJKQAAAASCSkAAIBEQgoAACCRkAIAAEgkpAAAABIJKQAAgERCCgAAIJGQAgAASCSkAAAAEn0DLiAINDiuCucAAAAASUVORK5CYII.png" width=800</img></p>

> Fig. 6. Model Fits on Training and Test Data

Following the graph, the least-square error for the training and test fitted curves were calculated.
```
LSE for Line:
Training: 1.8516699043464049
Test: 2.940305079714321 

LSE for Parabola:
Training: 1.8508364118270655
Test: 2.9058298821411337 

LSE for 19th degree Polynomial:
Training: 0.16381508563760222
Test: 507.47660243516526
```

## Summary and Conclusions
The cosine function, a line, parabola, and 19th degree polynomial were fitted to the data given. Fitting was done by minimizing the least-squares error when changing the parameters for each term of the function. Training was done on various sections of the data given, and the rest of the data was used for testing.
For the 2D loss landscape, as mentioned above, the minima can be observed by the dark blue region. As there are 31x31 errors calculated since there were 31 different values for the each of the two swept parameters, with one absolute minimum, but definitely a range of values that are within the dark blue region. While fixing most variables gives one distinct region being the minimum, fixing C and D gives a larger range of dark blue, with some lighter shades in between, meaning there are more minimums. Based on my swept values for the parameters, there seem to be three. Otherwise, there is one distinct region for each combination that is a minimum
For applying the model on the test data in the last two problems, using the first 10 and last 10 data points as training data with the middle 10 used as testing, the minimized error for each function has reduced. For the polynomial, by a lot. Each model has paramaters that are optimized closer to the raw data as the error returned a lesser value than before, when only the first 20 data points made up the training dataset.
