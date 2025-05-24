numbers = [15,8,22,7,31,4,17]
print("The even numbers in the list are: ")
for number in numbers:
    if number % 2 == 0:
        print(number)

square_of_odd_numbers = [number * number for number in numbers if number % 2 == 1]
print(f"The square of odd numbers in the list are: {square_of_odd_numbers}")
