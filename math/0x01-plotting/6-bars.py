#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

# your code here
fruits = ("apples", "bananas", "oranges", "peaches")
Farrah = fruit[0:][0:1]
Fred = fruit[0:][1:2]
Felicia = fruit[0:][2:3]
people = "Farrah", "Fred", "Felicia"
my_dict = {key: value[0:] for key, value in zip([v[0:] for v in fruits],
                                                [v[0:] for v in fruit])}
i = 0
for person in people:
    fruit_to_add = 0
    for key in fruits:
        if key == "apples":
            color = "red"
        elif key == "bananas":
            color = "yellow"
        elif key == "oranges":
            color = "#ff8000"
        elif key == "peaches":
            color = "#ffe5b4"
        plt.bar(person, my_dict.get(key)[i], color=color, bottom=fruit_to_add,
                width=0.5)
        fruit_to_add += my_dict.get(key)[i]
    i += 1
plt.legend(fruits)
plt.ylim(0, 80, 10)
plt.ylabel("Quantity of Fruit")
plt.xticks(people)
plt.title("Number of Fruit per person")
plt.show()
