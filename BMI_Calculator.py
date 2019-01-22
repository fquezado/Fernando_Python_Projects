user_height = float(input("Enter your height in inches: "))  # takes in user height
user_weight = float(input("Enter your weight in pounds: "))  # takes in user weight

user_BMI = user_weight * 703 / user_height**2  # calcs user BMI, with user input


# output different outputs depending on which if conditions are meet
if(user_BMI >= 18.5 and user_BMI <= 25):
    BMI_indicator = "Your BMI indicates that you are optimal weight."

if(user_BMI < 18.5):
    BMI_indicator = "Your BMI indicates that you are underweight."

if(user_BMI > 25):
    BMI_indicator = "Your BMI indicates that you are overweight."


print("Your BMI is:", round(user_BMI, 1), "\n" + BMI_indicator)
# prints user BMI and if overweight, underweight or optimal weight
