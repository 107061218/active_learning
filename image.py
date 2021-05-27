processed = []

fh = open('user02_FNbbox.txt', 'r')
fh2 = open('processed.txt', 'w')
L = fh.readlines()
d = dict()
areathreshold = 2800
# 5000: 77
# total = 0
for i in L:
    name = i.split(':')[0]
    box = eval(i.split(':')[1])
    d[name] = box
    x_range = abs(box[2] - box[0])
    y_range = abs(box[3] - box[1])
    area = x_range * y_range
    print(f'{name}: {x_range:3d} * {y_range:3d} = {area:5d}')
    if area >= areathreshold:
        processed.append(name[:6] + '\n')
        # total += 1

# print(d)
print(len(L))
print(len(processed))

fh2.writelines(processed)
fh.close()
fh2.close()
