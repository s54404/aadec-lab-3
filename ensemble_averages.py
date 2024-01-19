#!/usr/bin/env python
# coding: utf-8

# # Random Signals
# 
# *This jupyter notebook is part of a [collection of notebooks](../index.ipynb) on various topics of Digital Signal Processing. Please direct questions and suggestions to [Sascha.Spors@uni-rostock.de](mailto:Sascha.Spors@uni-rostock.de).*

# ## Ensemble Averages
# 
# Ensemble averages characterize the average properties of a random process across the population of all possible sample functions in the ensemble. We distinguish between first- and higher-order ensemble averages. The former consider the average properties of the sample functions of one random process for one particular time-instant $k$, while the latter take more than one random process at different time-instants into account.

# ### First-Order Ensemble Averages

# #### Definition
# 
# The first-order ensemble average of a continuous-amplitude real-valued random signal $x[k]$ is defined as
# 
# \begin{equation}
# E\{ f(x[k]) \} = \lim_{N \to \infty} \frac{1}{N} \sum_{n=0}^{N-1} f(x_n[k]),
# \end{equation}
# 
# where $E\{ \cdot \}$ denotes the expectation operator, $x_n[k]$ the $n$-th sample function and $f(\cdot)$ an arbitrary real-valued function. It is evident from the definition, that the ensemble average can only be given exactly for random processes where the internal structure is known. For practical random processes, like e.g. speech, the ensemble average can only be approximated by a finite but sufficiently large number $N$ of sample functions.
# 
# On the other hand, if the univariate probability density function (PDF) which characterizes the process is known, then the ensemble average may also be given as 
# 
# \begin{equation}
# E\{ f(x[k]) \} = \int\limits_{-\infty}^{\infty} f(\theta) \, p_x(\theta, k) \, \mathrm{d}\theta.
# \end{equation}

# #### Properties
# 
# The following properties can be concluded from the definition of the ensemble average:
# 
# 1. The ensemble averages of two different time-instants $k_1$ and $k_2$ generally differ
# 
#     \begin{equation}
#     E\{ f(x[k_1]) \} \neq E\{ f(x[k_2]) \}
#     \end{equation}
#     
# 2. For a linear mapping $f(x[k]) = C \cdot x[k]$ with $C \in \mathbb{R} \setminus \{0\}$, the ensemble average is a linear operation
#     
#     \begin{equation}
#     E\{ A \cdot x[k] + B \cdot y[k] \} = A \cdot E\{ x[k] \} + B \cdot E\{ y[k] \}
#     \end{equation}
# 
# 3. For a deterministic signal $x_n[k] = s[k]$ $\forall n$ the ensemble average is 
# 
#     \begin{equation}
#     E\{ f(s[k]) \} = f(s[k])
#     \end{equation}
# 
# The choice of the mapping function $f(\cdot)$ determines the particular property of the random process which is characterized by the ensemble average. Common and in practice most important choices are discussed in the following.

# #### Linear mean (1st raw moment)
# 
# The linear [mean](https://en.wikipedia.org/wiki/Mean), which is given for the linear mapping $f(x[k]) = x[k]$ is the arithmetic mean value across all sample functions $x_n[k]$ for a given time instant $k$. It is also known as the first raw [moment](https://en.wikipedia.org/wiki/Moment_%28mathematics%29).
# 
# Introducing $f(x[k]) = x[k]$ into the definition of the ensemble average yields
# 
# \begin{equation}
# \mu_x[k] = E\{ x[k] \} = \lim_{N \to \infty} \frac{1}{N} \sum_{n=0}^{N-1} x_n[k].
# \end{equation}
# 
# If the PDF of the process is known, then the linear mapping $f(\theta)=\theta$ yields 
# 
# \begin{equation}
# \mu_x[k] = E\{ x[k] \} = \int\limits_{-\infty}^{\infty} \theta \, p_x(\theta, k) \, \mathrm{d}\theta.
# \end{equation}
# 
# $\mu_x[k]$ is a widely accepted standard notation for the linear mean. A process with $\mu_x[k] = 0$ is termed as *zero-mean* or *mean-free*. Note that $\mu_x$ should not be confused with the discrete frequency index of the DFT.

# #### Quadratic mean (2nd raw moment)
# 
# The quadratic mean is given by the mappings $f(x[k]) = x^2[k]$ and $f(\theta)=\theta^2$. The expectation becomes
# 
# \begin{equation}
# E\{ x^2[k] \} = \lim_{N \to \infty} \frac{1}{N} \sum_{n=0}^{N-1} x_n^2[k]
# = \int\limits_{-\infty}^{\infty} \theta^2 \, p_x(\theta, k) \, \mathrm{d}\theta.
# \end{equation}
# 
# It quantifies the average instantaneous power of a sample function for a given time index $k$. It is also known as second raw moment.

# #### Variance (2nd central moment)
# 
# The [variance](https://en.wikipedia.org/wiki/Variance) is defined as the quadratic mean of a zero-mean random process. Note that the subtraction of the linear mean $\mu_x$ from the samples $x_n[k]$ makes a random signal mean-free. We are free to do this prior to calculation of the expectation. Then, for a general random process with linear mean $\mu_x$ this is given as
# 
# \begin{equation}
# \sigma_x^2[k] = E\{ (x[k] - \mu_x[k])^2 \} = E\{ \left(x[k] - E\{ x[k] \}\right)^2 \},
# \end{equation}
# 
# where $\sigma_x^2[k]$ commonly denotes the variance, $\sigma_x[k]$ is known as the [standard deviation](https://en.wikipedia.org/wiki/Standard_deviation). The variance characterizes how far the amplitude values of a random signal are spread out from its mean value. It is also known as second central moment.
# 
# The variance can be given in terms of the quadratic mean and the squared linear mean
# 
# \begin{equation}
# \sigma_x^2[k] = E\{ x^2[k] \} - \mu_x^2[k]
# \end{equation}
# 
# or vice versa, the quadratic mean is the sum of the squared linear mean and the variance 
# 
# \begin{equation}
# E\{ x^2[k] \} = \mu_x^2[k] + \sigma_x^2[k].
# \end{equation}
# 
# This dependence has a strong link to power calculation of pure resistor circuits in electrical engineering: The complete signal power is the sum of the powers of the direct component (DC) and the alternating component (AC).
# 

# **Excercise**
# 
# * Derive above relation from the definitions and properties of the first order ensemble average.
# 
# Solution: Taking the linearity of the expectation operator and the fact that the linear mean is a deterministic value into account yields
# 
# \begin{equation}
# \begin{split}
# \sigma_x^2[k] &= E\{ (x[k] - \mu_x[k])^2 \} \\
# &= E\{ x^2[k] - 2 \mu_x[k] x[k] + \mu_x^2[k]  \} \\
# &= E\{ x^2[k] \} - 2 \mu_x[k] E\{ x[k] \} + \mu_x^2[k] \\
# &= E\{ x^2[k] \} - 2 \mu_x[k] \mu_x[k] + \mu_x^2[k] \\
# & = E\{ x^2[k] \} - \mu_x^2[k]
# \end{split}
# \end{equation}

# #### Example - Linear and quadratic mean, and variance of a random process
# 
# The following example computes and plots the linear mean $\mu_x$ and quadratic mean, and variance $\sigma_x^2$ of a random process. Since in practice only a limited number $N$ of sample functions can be evaluated numerically the true values of these quantities are only approximated/estimated. Estimated quantities are denoted by a 'hat' over the respective quantity, e.g. $\hat{\mu}_x[k]$.

# In[7]:


import matplotlib.pyplot as plt
import numpy as np

K = 64  # number of random samples
N = 3000  # number of sample functions
f = 400
A = 400.25
B = 399.75

# generate the sample functions
np.random.seed(3)
W = np.random.normal(size=(N, K))  # Liczby losowe Wn(k) z rozkładu normalnego
x = A * np.cos(2 * f * np.pi / K) + B * W

# estimate the linear mean as ensemble average
mu = 1 / N * np.sum(x, 0)
# estimate the quadratic mean
qu = 1 / N * np.sum(x**2, 0)
# estimate the variance
sigma = 1 / N * np.sum((x - mu) ** 2, 0)


# plot results
plt.rc("figure", figsize=(10, 3))

plt.figure()
plt.stem(x[0, :], basefmt="C0:", linefmt="C0-", markerfmt="C0o", label=r"$x_0$")
plt.stem(x[1, :], basefmt="C1:", linefmt="C1--", markerfmt="C1o", label=r"$x_1$")
plt.stem(x[2, :], basefmt="C2:", linefmt="C2-.", markerfmt="C2o", label=r"$x_2$")
plt.title(r"Sample functions $x_0[k]$, $x_1[k]$, $x_2[k]$")
plt.xlabel(r"$k$")
plt.ylabel(r"$x[k]$")
plt.axis([0, K, -1500, 1500])
plt.legend()
plt.grid(True)

plt.figure()
plt.stem(mu, basefmt="C0:", linefmt="C0-", markerfmt="C0o", label=r"$\hat{\mu}_x[k]$")
plt.stem(
    mu**2, basefmt="C1:", linefmt="C1--", markerfmt="C1o", label=r"$\hat{\mu}^2_x[k]$"
)
plt.title(r"Estimate of linear mean and squared linear mean")
plt.xlabel(r"$k$")
plt.ylabel(r"$\hat{\mu}_x[k]$, $\hat{\mu}^2_x[k]$")
plt.axis([0, K, 0, 500])
plt.legend()

plt.figure()
plt.stem(qu, basefmt="C0:")
plt.title(r"Estimate of quadratic mean")
plt.xlabel(r"$k$")
plt.ylabel(r"$\hat{E}\{x^2[k]\}$")
plt.axis([0, K, 0, 250000])

plt.figure()
plt.stem(sigma, basefmt="C0:")
plt.title(r"Estimate of variance")
plt.xlabel(r"$k$")
plt.ylabel(r"$\hat{\sigma}^2_x[k]$")
plt.axis([0, K, 0, 250000])


# **Exercise**
# 
# * What does the linear and quadratic mean, and the variance tell you about the average behavior of the sample functions?
# * Change the number `N` of sample functions and rerun the example. What influence has a decrease/increase of the sample functions on the estimated ensemble averages?
# 
# Solution: Inspection of the estimated linear mean reveals that in average the sample functions follow a cosine function with respect to the sample index $k$. The variance shows that the amount of which the samples of the sample functions for one particular time-instant $k$ are spread around the linear mean is constant. The estimate of the quadratic mean is given as $\hat{E}\{ x^2[k] \} = \hat{\mu}_x^2[k] + \hat{\sigma}_x^2[k]$. The higher the number $N$ of sample functions used for the estimate of the ensemble averages, the lower the uncertainty in comparison to the true values becomes.

# ### Second-Order Ensemble Averages

# #### Definition
# 
# The second-order ensemble average of two continuous-amplitude, real-valued random signals $x[k]$ and $y[k]$ is defined as
# 
# \begin{equation}
# E\{ f(x[k_x], y[k_y]) \} := \lim_{N \to \infty} \frac{1}{N} \sum_{n=0}^{N-1} f(x_n[k_x], y_n[k_y]).
# \end{equation}
# 
# It is given in terms of the bivariate PDF as
# 
# \begin{equation}
# E\{ f(x[k_x], y[k_y]) \} = \iint\limits_{-\infty}^{\infty} f(\theta_x, \theta_y) \, p_{xy}(\theta_x, \theta_y, k_x, k_y) \, \mathrm{d}\theta_x\, \mathrm{d}\theta_y.
# \end{equation}
# 
# The definition of the second-order ensemble average can be extended straightforward to the case of more than two random variables. The resulting ensemble average is then termed as higher-order ensemble average. 
# 
# By setting $y = x$, the second-order ensemble average can also be used to characterize the average properties of the sample functions between two different time-instants $k_1$ and $k_2$ of one random process
# 
# \begin{equation}
# \begin{split}
# E\{ f(x[k_1], x[k_2]) \} &= \lim_{N \to \infty} \frac{1}{N} \sum_{n=0}^{N-1} f(x_n[k_1], x_n[k_2]) \\
# &= \iint\limits_{-\infty}^{\infty} f(\theta_1, \theta_2) \, p_{xx}(\theta_1, \theta_2, k_1, k_2) \, \mathrm{d}\theta_1\, \mathrm{d}\theta_2.
# \end{split}
# \end{equation}
# 
# It is worth to note that for $y = x$ and equal time-instant $k$, the first-order ensemble average $E\{ f(x[k]) \} = \int\limits_{-\infty}^{\infty} f(\theta) \, p_x(\theta, k) \, \mathrm{d}\theta$ is obtained. Thus, the definition of the higher-order ensemble average constitutes a general formulation of the expectation $E\{\cdot\}$.
# 
# The choice of the mapping function $f(\cdot)$ determines the particular property of the random process which is characterized by the ensemble average. The important case of a linear mapping is discussed in the following.

# #### Cross-correlation function
# 
# The [cross-correlation function](https://en.wikipedia.org/wiki/Cross-correlation) (CCF) of two random signals $x[k]$ and $y[k]$ is defined as the second-order ensemble average for a linear mapping $f(x[k_x], y[k_y]) = x[k_x] \cdot y[k_y]$
# 
# \begin{equation}
# \varphi_{xy}[k_x, k_y] = E\{ x[k_x] \cdot y[k_y] \}.
# \end{equation}
# 
# It characterizes the statistical dependencies of two random signals $x[k]$ and $y[k]$ at two different time instants $k_x$ and $k_y$.

# #### Auto-correlation function
# 
# The [auto-correlation function](https://en.wikipedia.org/wiki/Autocorrelation) (ACF) of a random signal $x[k]$ is defined as the second-order ensemble average for a linear mapping $f(x[k_1], x[k_2]) = x[k_1] \cdot x[k_2]$
# 
# \begin{equation}
# \varphi_{xx}[k_1, k_2] = E\{ x[k_1] \cdot x[k_2] \}.
# \end{equation}
# 
# It characterizes the average statistical dependencies between the samples of a random signal $x[k]$ at two different time instants $k_1$ and $k_2$.

# #### Example - Auto-correlation function of a random signal
# 
# The following example estimates the ACF of a random signal $x[k]$ as ensemble average of a finite number $N$ of sample functions. The ACF is plotted as image where the colors denote the level of the ACF $\varphi_{xx}[k_1, k_2]$ for given time-instants $k_1$ and $k_2$.

# In[3]:


# Określ wartości parametrów
N = 3000  # Liczba funkcji próbkowych
f = 400
A = 400.25
B = 399.75
L = 64  # Liczba losowych próbek

# Generowanie funkcji próbkowych
np.random.seed(1)
W = np.random.normal(size=(N, L))  # Liczby losowe Wn(k) z rozkładu normalnego
x = A * np.cos(2 * f * np.pi / np.arange(1, L + 1)) + B * W

# Estymacja funkcji autokorelacji (ACF)
acf = np.zeros((L, L))
for n in range(L):
    for m in range(L):
        acf[n, m] = 1 / N * np.sum(x[:, n] * x[:, m], axis=0)

# Wykres ACF
plt.figure(figsize=(7, 5))
plt.pcolormesh(np.arange(L + 1), np.arange(L + 1), acf)
plt.title(r"Estimate of ACF $\hat{\varphi}_{xx}[k_1, k_2]$")
plt.xlabel(r"$k_1$")
plt.ylabel(r"$k_2$")
plt.colorbar()
plt.axis("tight")
plt.show()


# **Exercise**
# 
# * Can you explain the specific symmetry of the ACF?
# 
# Solution: Inspection of the definition of the ACF $\varphi_{xx}[k_1, k_2] = \lim_{N \to \infty} \frac{1}{N} \sum_{n=0}^{N-1} x_n[k_1] \cdot x_n[k_2]$ reveals that the sample indexes $k_1$ and $k_2$ can be interchanged without changing the value of the ACF. Hence, $\varphi_{xx}[k_1, k_2] = \varphi_{xx}[k_2, k_1]$. This symmetry can be observed in above plot.

# **Copyright**
# 
# This notebook is provided as [Open Educational Resource](https://en.wikipedia.org/wiki/Open_educational_resources). Feel free to use the notebook for your own purposes. The text is licensed under [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/), the code of the IPython examples under the [MIT license](https://opensource.org/licenses/MIT). Please attribute the work as follows: *Sascha Spors, Digital Signal Processing - Lecture notes featuring computational examples*.
