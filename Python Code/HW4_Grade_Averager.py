total = 0
x = True
i = 0
while x is True:  # while loop for getting user input until user inputs '-1'
    user_input = int(input("Enter a test score, -1 to get the average: "))
    if user_input is -1: # if user inputs '-1' will break out of the while loop
        break
    total = total + user_input
    i = i + 1  # adds number of times user inputs number in order to average

average = total/i
print("The average for all the grades is:", round(average, 1))
# prints average rounded to one decimal
