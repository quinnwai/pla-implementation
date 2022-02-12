#!/usr/bin/python2.7
# Homework 0 Code
import numpy as np
import matplotlib.pyplot as plt


def perceptron_learn(data_in):
    # Run PLA on the input data
    #
    # Inputs: data_in: Assumed to be a matrix with each row representing an
    #                (x,y) pair, with the x vector augmented with an
    #                initial 1 (i.e., x_0), and the label (y) in the last column
    # Outputs: w: A weight vector (should linearly separate the data if it is linearly separable)
    #        iterations: The number of iterations the algorithm ran for

    # Your code here, assign the proper values to w and iterations:
    w = np.zeros(len(data_in[0][0]))

    iterations = 0
    i = 0
    while i < len(data_in):
        # print(f'iter: {i}')
        (x,y) = data_in[i]
        y_pred = -1
        if np.dot(w,x) >= 0:
            y_pred = 1
        if y == y_pred:
            i+=1
        else:
            w = w + y*x
            iterations += 1
            i = 0
    
    return w, iterations


def perceptron_experiment(N, d, num_exp):
    # Code for running the perceptron experiment in HW0
    # Implement the dataset construction and call perceptron_learn; repeat num_exp times
    #
    # Inputs: N is the number of training data points
    #         d is the dimensionality of each data point (before adding x_0)
    #         num_exp is the number of times to repeat the experiment
    # Outputs: num_iters is the # of iterations PLA takes for each experiment
    #          bounds_minus_ni is the difference between the theoretical bound and the actual number of iterations
    # (both the outputs should be num_exp long)

    # Initialize the return variables
    num_iters = np.zeros((num_exp,))
    bounds_minus_ni = np.zeros((num_exp,))

    # Your code here, assign the values to num_iters and bounds_minus_ni:
    for i in range(num_exp):
    # for i in range(1):
        w_star = np.append([0], np.random.random_sample(d))
        # print(f"w_star: {w_star}")

        data_in = [0]*N
        x_vals = [0]*N
        y_vals = [0]*N
        for j in range(N):
            x = np.append([1], 2*np.random.random_sample(d)-1)
            y = np.sign(np.dot(w_star,x))
            # print(f"x,y: {x}, {y}")
            x_vals[j] = x
            y_vals[j] = y
            data_in[j] = (x,y)
        

        data_in = np.array(data_in, dtype=tuple)
        w, num_iters[i] = perceptron_learn(data_in)
        
        
        rhos = np.array([np.dot(x,w) for x in x_vals])

        y_vals = np.array(y_vals)
        rho = (rhos*y_vals).min()
        r = np.array([np.linalg.norm(x) for x in x_vals]).max()
        w_norm = np.linalg.norm(w)
    
        # store bounds_min_ni w calculation
        bounds_minus_ni[i] = ((r**2)*w_norm)/rho**2

    return num_iters, bounds_minus_ni


def main():
    print("Running the experiment...")
    num_iters, bounds_minus_ni = perceptron_experiment(100, 10, 1000)

    print("Printing histogram...")
    plt.hist(num_iters)
    plt.title("Histogram of Number of Iterations")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Count")
    plt.show()

    print("Printing second histogram")
    plt.hist(np.log(bounds_minus_ni))
    plt.title("Bounds Minus Iterations")
    plt.xlabel("Log Difference of Theoretical Bounds and Actual # Iterations")
    plt.ylabel("Count")
    plt.show()

if __name__ == "__main__":
    main()
