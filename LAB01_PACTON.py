# -*- coding: utf-8 -*-
"""
Charles-Arthur Pacton

"""
################# Exercise 1.4.1 ######################
print("Exercise 1.4.1")
print("")


for i in range(1, 6):
    for j in range(i):
        print("X", end="")
    print()
        
# This first loop will browse the numbers from 1 to 5 and at each iteration of the latter it will browse the numbers from 0 to i.
# Then we define the loop variable and we print 'X' using the "end" function to prevent the newline
# After this second loop, an empty line is printed to create a new line. 
#So this process will repeat 5 times

for i in range(4, 0, -1):
    for j in range(i):
        print("X", end="")
    print()    

# # The second part of the code is similar, but the first for loop iterates through numbers from 4 to 1 using a step size of -1. 
#(we make it only 4 times because there is one line with 5 "X")
# This way, the lines are printed in reverse order to give the same end result.

################# Exercise 1.4.2 ######################
print("")
print("Exercise 1.4.2")
print("")
input_str = "n45as29@#8ss6" #we define the character chain we have in the exercise
def function_2(input_str):
    number = 0 #initialization of a variable that will stock the number
    for x in input_str: 
        if x.isdigit() == True: #check if the character is a number or not
            z = int(x) 
            number = number + z #addition of the number that are detected one by one

    return number

print(function_2(input_str))

################# Exercise 1.4.3 ######################
print("")
print("Exercise 1.4.3")
print("")
def conversion(n):
    if n > 1:
        conversion(n // 2)
    print(n % 2, end='')

nbr = int(input("enter the number to convert: ")) #ask to the user to enter a number
conversion(nbr)
print("")

################# Exercise 1.5.1 ######################
print("")
print("Exercise 1.5.1")
print("")

list = [0,1]
def Fibonacci(num) :
    for i in range(1000):
        if list[-1] < num :
            list.append(list[-2]+list[-1])
    
    return list

nbr = int(input("enter the number you want : "))
Fibonacci(nbr)















