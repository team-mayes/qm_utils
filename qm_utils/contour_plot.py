import numpy as np
import matplotlib.pyplot as plt
import os
import csv
from qm_utils.qm_common import read_csv_to_dict, get_csv_fieldnames
import numpy as np

file_read = os.path.join('/Users/vicchio/Box Sync/Michigan/lab-TEAM/projects/puckering/csv-files/','contour_sample_pathways_bxyl.csv')

#
# hartree_dict = read_csv_to_dict(file_read, mode='rU')
# hartree_headers = get_csv_fieldnames(file_read, mode='rU')
# print(hartree_dict, hartree_headers)


with open(file_read, 'rU') as read:
    headers = next(read)
    min_reader = csv.reader(read)
    x = []
    y = []
    z = []
    for row in min_reader:
        x.append(float(row[7]))
        y.append(float(row[8]))
        z.append(float(row[11]))

    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

f, ax = plt.subplots(1,2, sharex=True, sharey=True)
ax[0].tripcolor(x,y,z)
ax[1].tricontourf(x,y,z, 1000) # choose 20 contour levels, just to show how good its interpolation is
ax[1].plot(x,y, 'ko ')
ax[0].plot(x,y, 'ko ')
plt.savefig('test.png')

# plt.figure()
# CS = plt.contour(x, y, z)
# plt.title('labels at selected locations')


# with open(file_read, 'rU') as read:
#     min_reader = csv.reader(read, skipinitialspace=True, delimiter=',')
#     row_count = sum(1 for row in read)
#     x = np.array([0]*len(row_count),dtype=np.float)
#     y = np.array([0]*len(row_count),dtype=np.float)
#     z = np.array([0]*len(row_count),dtype=np.float)
#     for row in min_reader:
#
#
#
# x = np.random.rand(100)
# y = np.random.rand(100)
# z = np.sin(x)+np.cos(y)
# f, ax = plt.subplots(1,2, sharex=True, sharey=True)
# ax[0].tripcolor(x,y,z)
# ax[1].tricontourf(x,y,z, 20) # choose 20 contour levels, just to show how good its interpolation is
# ax[1].plot(x,y, 'ko ')
# ax[0].plot(x,y, 'ko ')
# plt.savefig('test.png')
