#Given: numbers = [1, 2, 3, 4, 5, 6, 7]
#Expected Output: [1, 4, 9, 16, 25, 36, 49]
#Solution :Create an empty result listIterate a numbers list using a loopIn each iteration, calculate the square of a current number and
#add it to the result

num_list = [1, 2, 3, 4, 5, 6, 7]
squared_list = [num**2 for num in num_list]
print(f'Original List = {num_list}')
print(f'Squares = {squared_list}')
