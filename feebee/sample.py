
def new_breaks(breaks, group_breaks):
    index = 0
    result = []
    n = len(breaks)
    for b0 in group_breaks:
        if index >= n:
            break
        if b0 < breaks[index]:
            continue
        while breaks[index] <= b0:
            index += 1
            if index >= n:
                break
        result.append(b0) 
    return result


breaks = [2, 7, 8, 9, 10, 11, 12, 18]
group_breaks = [0.5, 1, 8, 10, 12, 12.5, 13, 17, 20]

print(new_breaks(breaks, group_breaks))
