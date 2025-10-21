#!/export/apps/anaconda3/2021.05/bin/python

import os
import numpy as     np

T     = 300
#E     = [0.025, 0.05, 0.1, 0.3]
#Trays = [[6, 31, 32, 50, 51, 61, 62, 63, 64, 66, 74, 81, 82, 112, 121, 126, 140, 151, 157, 163, 174, 210, 213, 227, 228, 237, 256, 266, 297],
#         [4, 8, 31, 36, 77, 79, 88, 90, 93, 99, 100, 131, 138, 158, 164, 188, 227, 232, 266, 268, 278],
#         [11, 14, 40, 41, 44, 65, 67, 88, 130, 145, 156, 184, 218, 224, 236, 238, 253, 254, 265, 271, 277],
#         [49, 108, 141, 148, 166, 213, 234, 253, 270]]

E = [0.025, 0.05]
Trays = [[64, 65], [73, 74]]

for i, row  in enumerate(Trays):
    folder     = "vaspdata.Ei." + str(E[i]) + ".Ts." + str(T) + ".NO.rand.zpe"
    for elem in row:
        if os.path.exists(folder + "/aimd_resume-" + str(elem) + ".dat") == False or os.path.getsize(folder + "/aimd_resume-" + str(elem) + ".dat") == 0:      
            print("AIMD resume file for trayectory " + str(elem) + " in " + str(E[i]) + " eV present error")
            continue
        else:
            file = np.loadtxt(folder + "/aimd_resume-" + str(elem) + ".dat")
        
        for j in range(1, int(len(file)) + 1):
            file[j-1][0] = j

        with open(folder + "/aimd_resume-" + str(elem) + ".dat", 'w') as f:
            for row in file:
                for elem in row:
                    f.write(str(elem) + ' ')
                f.write("\n")
        f.close()

print('Done !!!')
