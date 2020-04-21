#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

# your code here
print(fruit)
fruits = ("apples", "bananas", "oranges", "peaches")
Farrah = fruit[0:][0:1]
Fred = fruit[0:][1:2]
Felicia = fruit[0:][2:3]
people = "Farrah", "Fred", "Felicia"
print("+++++++++++++++++")
print(Farrah)
print(Fred)
print(Felicia)
my_dict = {key: value[0:] for key, value in zip([v[0:] for v in fruits], [v[0:] for v in fruit])}
print(my_dict)
print(my_dict["apples"])
print([key for key in fruits])


for key in fruits:
    fruit_to_add = 0
    for person in people:
        if key == "apples":
            color = "red"
        elif key == "bananas":
            color = "yellow"
        print(my_dict.get(key))
        
        plt.bar(person, key, color=color)
        fruit_to_add += my_dict.get(key)

plt.show()