user_input_celsius = float(input("Enter a Celsius temperature to convert to Fahrenheit: "))   #taking user input in

F = ((9/5) * user_input_celsius) + 32   #Converting Celsius input to fahrenheit

F = round(F, 1)  #Rounding to one decimal place

print("The Fahrenheit equivalent is: " + str(F))   #printing desired output
