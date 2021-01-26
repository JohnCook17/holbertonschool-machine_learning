#!/usr/bin/env python3
"""A Simple input loop"""

while True:
    print("Q: ", end="")
    question = input()

    if ((question == "exit"
         or question == "quit"
         or question == "goodbye"
         or question == "bye")):
        print("A: Goodbye")
        exit()
