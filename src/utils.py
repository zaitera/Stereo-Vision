import numpy as np
from MouseClick import *

def calcWorldCoordinates(height, width, focal_length, baseline, disp):
    print("calculating real world coordenates...")
    xL = np.arange(np.float32(width))
    xL = np.tile(xL,(height,1))
    yL = np.arange(np.float32(height))
    yL = np.tile(yL,(width,1))
    yR = yL
    xR = xL + disp
    const = baseline/2
    deltaX = xL-xR
    deltaX[deltaX == 0.0] = np.inf
    X = const*((xL + xR) / deltaX)
    Y = const*((yL + yR) / np.transpose(deltaX))
    const = baseline * focal_length
    Z = const / deltaX
    world_coordinates = cv2.merge((X,np.transpose(Y),Z))
    return world_coordinates
    

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
