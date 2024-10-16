#Given two lists of integers, Find the common elements between them and return them in a sorted list without duplicates.
#Given: list1 = [1, 2, 3, 4, 5, 6, 7, 8, 9] and list2 = [5, 6, 7, 8, 9, 10, 11, 12, 13]
#Expected Output:   Common elements: [5, 6, 7, 8, 9]

list1 = [1, 2, 3, 4, 5, 6, 7, 8, 9,]
list2 = [5, 6, 7, 8, 9, 10, 11, 12, 13]
common_element = sorted(list(set(list1).intersection(set(list2))))
print("list1: ",list1,"\nlist2: ",list2,"\n")
print(f"Common list from list1 & list2: {common_element}")
