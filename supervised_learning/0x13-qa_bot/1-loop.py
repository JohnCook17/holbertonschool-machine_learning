#!/usr/bin/env python3
""""""

while True:
    print("Q: ", end="")
    question = input().lower()

    if ((question == "exit"
         or question == "quit"
         or question == "goodbye"
         or question == "bye")):
        print("A: Goodbye")
        exit()
