def is_prime(x):
    check = 1
    for i in range(2, x):
        if x % i == 0:
            check = 0
            break
    if check == 1:
        return True
    else:
        return False

nums = [2, 4, 5, 10, 13, 17, 20, 23]    #Defining the list
primes = [num for num in nums if is_prime(num)==True]
print(f"The list of primes in the given list is : {primes}")
