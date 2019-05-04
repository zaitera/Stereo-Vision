import numpy as np
from MouseClick import *

def calcWorldCoordinates(world_coordinates,focal_length, baseline, disp):
    print("calculating world coordenates...\n")
    for i in range(world_coordinates.shape[1]):
        for j in range(world_coordinates.shape[0]):
            #X and Y of left and right image considering the disparity on the shifted
            xL = i
            yL = j
            xR = i + disp[i][j]
            yR = j
            #Calcula as coordenadas do mundo
            if (xL- xR) != 0:
                X = (baseline * (xL + xR)) / (2 * (xL- xR))
                Y = (baseline * (yL + yR)) / (2 * (xL- xR))
                Z = (baseline * focal_length) / (xL-xR)
            else:
                X = Y = Z = 0
            
            world_coordinates[j][i][0] = X
            world_coordinates[j][i][1] = Y
            world_coordinates[j][i][2] = Z
    print("world coordenates calculated...\n")

def calculateDisparity(imgL,imgR,mindisp,maxdisp):
     # SGBM Parameters -----------------
    window_size = 15                     # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
    
    left_matcher = cv2.StereoSGBM_create(
        minDisparity=mindisp,
        numDisparities=maxdisp,             # max_disp has to be dividable by 16 f. E. HH 192, 256
        blockSize=5,
        P1=8 * 3 * window_size ** 2,    # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
        P2=32 * 3 * window_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=0,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    
    # FILTER Parameters
    lmbda = 80000
    sigma = 1.2
    visual_multiplier = 1.0
    
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)
    
    print('computing disparity...')
    displ = left_matcher.compute(imgL, imgR)
    dispr = right_matcher.compute(imgR, imgL) 
    displ = np.int16(displ)
    dispr = np.int16(dispr)

    filteredImg = wls_filter.filter(displ, imgL, None, dispr)  # important to put "imgL" here!!!
    cv2.filterSpeckles(filteredImg, 0, 4000, maxdisp) 

    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
    filteredImg = np.uint8(filteredImg)
    return filteredImg
