import numpy as np
from scipy.integrate import quad
from scipy.special import gamma, binom
import matplotlib.pyplot as plt
import random
import math

def make_posterior(prior, likelihood):
    #define the unnormalized posterior
    def unnormalized(theta):
        return prior(theta) * likelihood(theta)

    Z, _ = quad(unnormalized, 0.0, 1.0)   # Get normalization

    def posterior(theta):
        return unnormalized(theta) / Z

    return posterior

def beta_dist(alpha, beta, theta):
    return (gamma(alpha+beta)/(gamma(alpha)*gamma(beta))*theta**(alpha-1)*(1-theta)**(beta-1))

def likelihood(n, h, theta):
    return binom(n, h) * theta**h * (1 - theta)**(n - h)

# starting problem 1

def read_coin_data(filename, max_lines=None):
    """
    Read up to max_lines from a file containing one value per line.
    
    Assumes:
        1 = heads
        0 = tails

    Returns:
        heads_count, total_measurements
    """
    heads_count = 0
    total_measurements = 0

    with open(filename, "r") as f:
        for i, line in enumerate(f):
            if max_lines is not None and i >= max_lines:
                break

            value = line.strip()

            if value == "":
                continue  # skip blank lines

            if value not in {"0", "1"}:
                raise ValueError(f"Invalid entry on line {i+1}: {value}")

            if value == "1":
                heads_count += 1

            total_measurements += 1

    return heads_count, total_measurements


#Plot the entire dataset
n, h = read_coin_data('HW06_data.txt')


def beta_prob1(theta):
    return beta_dist(1, 1, theta)

def likelihood_prob1(theta):
    return likelihood(h, n, theta)

x = np.linspace(0, 1, 1000)
posterior = make_posterior(beta_prob1, likelihood_prob1)
posterior_vals = [posterior(val) for val in x]
plt.plot(x, posterior_vals)
plt.xlabel("theta")
plt.ylabel("probability")
plt.show()

#Plot only the first 50 values
n, h = read_coin_data('HW06_data.txt', max_lines=50)

def beta_prob1(theta):
    return beta_dist(1, 1, theta)

def likelihood_prob1(theta):
    return likelihood(h, n, theta)

posterior = make_posterior(beta_prob1, likelihood_prob1)
posterior_vals = [posterior(val) for val in x]
plt.plot(x, posterior_vals)
plt.title("Probaility distrubution for 50 datapoints")
plt.xlabel("theta")
plt.ylabel("probability")
plt.show()

#Plot only the first 100 values
n, h = read_coin_data('HW06_data.txt', max_lines=100)


def beta_prob1(theta):
    return beta_dist(1, 1, theta)

def likelihood_prob1(theta):
    return likelihood(h, n, theta)

posterior = make_posterior(beta_prob1, likelihood_prob1)
posterior_vals = [posterior(val) for val in x]
plt.plot(x, posterior_vals)
plt.title("Probaility distrubution for 100 datapoints")
plt.xlabel("theta")
plt.ylabel("probability")
plt.show()

#Plot only the first 500 values
n, h = read_coin_data('HW06_data.txt', max_lines=500)


def beta_prob1(theta):
    return beta_dist(1, 1, theta)

def likelihood_prob1(theta):
    return likelihood(h, n, theta)

posterior = make_posterior(beta_prob1, likelihood_prob1)
posterior_vals = [posterior(val) for val in x]
plt.plot(x, posterior_vals)
plt.title("Probaility distrubution for 500 datapoints")
plt.xlabel("theta")
plt.ylabel("probability")
plt.show()

#try different distributionss
n, h = read_coin_data('HW06_data.txt', max_lines=500)


def beta_prob1(theta):
    return beta_dist(4, 2, theta)

def likelihood_prob1(theta):
    return likelihood(h, n, theta)

posterior = make_posterior(beta_prob1, likelihood_prob1)
posterior_vals_d = [posterior(val) for val in x]
plt.plot(x, posterior_vals_d, color="red", label="beta(2, 4)")
plt.plot(x, posterior_vals, color="blue", label="flat dist.")
plt.title("Probaility distrubution overlayed for flat and non-flat posterior")
plt.xlabel("theta")
plt.ylabel("probability")
plt.show()

def beta_prob1(theta):
    return beta_dist(20,40, theta)


posterior = make_posterior(beta_prob1, likelihood_prob1)
posterior_vals_d = [posterior(val) for val in x]
plt.plot(x, posterior_vals_d, color="red", label="beta(20, 40)")
plt.plot(x, posterior_vals, color="blue", label="flat dist.")
plt.title("Probaility distrubution overlayed for flat and non-flat posterior")
plt.xlabel("theta")
plt.ylabel("probability")
plt.show()


#problem 2 starting

def get_point():
    x = random.uniform(0.0, 1.0)
    y = random.uniform(0.0, 1.0)
    return x, y

n = 1000
n_index = 1
num_inside = 0
while n_index <= n:
    x, y = get_point()
    if math.sqrt(x**2 + y**2) < 1:
        num_inside += 1
    n_index += 1

print(num_inside)

def beta_prob2(theta):
    return beta_dist(1, 1, theta)

def likelihood_prob2(theta):
    return likelihood(n, num_inside, theta)

x = np.linspace(0, 1, 1000)
posterior_2 = make_posterior(beta_prob2, likelihood_prob2)
posterior_vals_2 = [posterior_2(val) for val in x]
plt.plot(x, posterior_vals_2)
plt.title("Probaility distrubution of random point in circle")
plt.xlabel("theta")
plt.ylabel("probability")
plt.show()

idx = np.argmax(posterior_vals_2)
print(x[idx])