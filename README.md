
<h1><br>Homework 1: Model Fitting<br></h1>
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


```code sample
```

> **Note**

## Algorithm Implementation and Development
The procedure is discussed in this section. For the results, see <a href="#Computational Results">Computational Results</a>
### Problem 1
The minimum error for the cosine function (Fig. 2) was found by defining a function that evaluates the error of the function at each point of given data.

![image](https://user-images.githubusercontent.com/114551272/230567103-a3a254bb-7c5f-477e-9e71-94e4189d37ce.png)
> Fig. 2. Cosine Function

This function was sent through the python optimization module, where parameters A, B, C, and D are returned for a minimum error value.
```
def fit_er(c, x, y):
    model = c[0]*np.cos(c[1]*x)+c[2]*x + c[3] # define and evaluate cosine function at current point
    er = np.sqrt(np.sum((model-y)**2)/len(x)) # evaluate error
    return er
```
After optimization, the found parameters were substituted in the cosine function, and the overall equation was graphed against the raw data.

### Problem 2



```
# given data
x = np.arange(0, 31)
y = np.array([30, 35, 33, 32, 34, 37, 39, 38, 36, 36, 37, 39, 42, 45, 45, 41, 40, 39, 42, 44, 47, 49, 50, 49, 46, 48, 50, 53, 55, 54, 53])
```
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

![sXQAAAABJRU5ErkJggg](https://user-images.githubusercontent.com/114551272/230569544-d414d59f-feb7-4eee-8dd3-6dfe9f04d6ff.png)
> Fig. 5. Model Fits on Training Data

![TxEulk3tnEnwAAAABJRU5ErkJggg](https://user-images.githubusercontent.com/114551272/230569734-ec405486-f9a9-430e-a9c2-b131d176c7a3.png)
> Fig. 6. Model Fits on Training Data

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


<p><img src="https://media.discordapp.net/attachments/847715688273281034/1093809847406956635/t73ucAgG2Cv3giBQX09fUVDw8P0el0otlsxmg0it1uF7e3t3mfBkBJ2SY4J6SggAaDQUyn09hut1Gr1aLf70ej0YjX19e8TwOgpGwTnPPXPiiY5XIZo9EoJpNJ1Ov1qFQqMZlMYrVaxXg8zvs8AErINsFvnkgBAAAk8kQKAAAgkZACAABIJKQAAAASCSkAAIBEQgoAACCRkAIAAEgkpAAAABIJKQAAgERCCgAAIJGQAgAASCSkAAAAEn0DLiAINDiuCucAAAAASUVORK5CYII.png" </img></p>
> Fig. 7. Model Fits on Training and Test Data

## Summary and Conclusions
...
