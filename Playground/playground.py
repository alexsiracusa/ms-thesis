

total = 0
sizes = [2500, 500, 200, 100, 10]

for i in range(len(sizes)):
    for j in range(len(sizes)):
        if i != 0:
            total += (sizes[j] * sizes[i]) + sizes[i]

print(total)