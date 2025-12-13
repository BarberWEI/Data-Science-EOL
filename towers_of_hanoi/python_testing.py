
#initial_tower = [[i for i in range(2, 0, -1)], [], []] 
initial_tower = [[1],[],[3,2]]
target_tower = [[], [], [i for i in range(3, 0, -1)]]

def completed(one, two):
    return one == two 

print(completed(initial_tower, target_tower))