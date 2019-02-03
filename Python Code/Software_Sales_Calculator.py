def rounding(x):
    return "{:.2f}".format(x)   # Rounding function allows me to simplify rounding the variables to 2 decimal places


def software_sales_calc(n):  # sales_calc function allows me to calculate and print total cost and discount
    purchase_price = n * 100
    package_type = bool

    # each if is for certain condition depending on what the user inputs
    if(packages_ordered >= 0 and packages_ordered <= 9):
        discount = purchase_price * 0
        purchase_price = purchase_price - discount
        package_type = True

    if(packages_ordered >= 10 and packages_ordered <= 19):
        discount = purchase_price * .1
        purchase_price = purchase_price - discount
        package_type = True

    if(packages_ordered >= 20 and packages_ordered <= 49):
        discount = purchase_price * .2
        purchase_price = purchase_price - discount
        package_type = True

    if(packages_ordered >= 50 and packages_ordered <= 99):
        discount = purchase_price * .3
        purchase_price = purchase_price - discount
        package_type = True

    if(packages_ordered >= 100):
        discount = purchase_price * .4
        purchase_price = purchase_price - discount
        package_type = True

    if(package_type == True):  # checks if one of the if's conditions were passed and prints the values calculated
        print("The total cost of your purchase was $" + rounding(purchase_price), "with a discount of $" + rounding(discount))


packages_ordered = float(input("Enter the number of packages ordered: "))  # asking user for input
software_sales_calc(packages_ordered)  # calling my function and passing user input into it, for n

