from numpy import *
import pandas as pd

b_final = None
m_final = None


def compute_cost_function(m_gradient, b_gradient, points, learningRate):
    error = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        error += (y - (m_gradient * x + b_gradient)) ** 2
    return error / float(len(points))


def gradient_descent(b_current, m_current, points, learningRate):
    b_gradient = 0
    m_gradient = 0
    n = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        m_gradient += -(2 / n) * x * (y - (m_current * x + b_current))
        b_gradient += -(2 / n) * (y - (m_current * x + b_current))
    new_b = b_current - (learningRate * b_gradient)
    new_m = m_current - (learningRate * m_gradient)
    return [new_b, new_m]


def gradient_descent_main(m_initial, b_initial, learningRate, points, num):
    b = b_initial
    m = m_initial
    for i in range(num):
        b, m = gradient_descent(b, m, points, learningRate)
    return [b, m]


def run():
    points = genfromtxt("Data.csv", delimiter=",")
    # points = pd.read_csv("Data.csv")
    # pq = pd.read_csv('Data.csv')
    # print(pq)
    m_initial = 0
    b_initial = 0
    learningRate = 0.0001
    num = 1000

    print("Initial value of b = {0} and m = {1} and error = {1}".format(b_initial, m_initial,
                                                                        compute_cost_function(b_initial, m_initial,
                                                                                              points, learningRate)))
    [b, m] = gradient_descent_main(m_initial, b_initial, learningRate, points, num)
    print("Final value of b = {0} and m = {1} and error = {1}".format(b, m,
                                                                      compute_cost_function(b_initial, m_initial,
                                                                                            points, learningRate)))

    global b_final
    b_final = b
    global m_final
    m_final = m


def check_new_input(x):
    return m_final * x + b_final


run()
# c = check_new_input(61.5304)
# print("output = ", c)
