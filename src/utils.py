import numpy as np
import matplotlib.pyplot as plt
from MouseClick import *

MAX_VIS_DISTANCE_M = 19700

MAX_VIS_DISTANCE_P = 85000

def circles(img1, img2, pts1, pts2, i):
    color = tuple(np.random.randint(0,255,3).tolist())
    img1 = cv2.circle(img1,tuple(pts1[i]),15,color,-1)
    img2 = cv2.circle(img2,tuple(pts2[i]),15,color,-1)
    return img1, img2

def matching(img1, img2, number):
    orb = cv2.ORB_create()

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
    matches = bf.match(des1,des2)
    matches = sorted(matches, key = lambda x:x.distance)

    pts1 = []
    pts2 = []


    #reference: https://stackoverflow.com/questions/30716610/how-to-get-pixel-coordinates-from-feature-matching-in-opencv-python
    for mat in matches:
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        pts1.append((kp1[img1_idx].pt))
        pts2.append((kp2[img2_idx].pt)) 

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)
    print("Fundamental Matrix: \n", F)

    pts1 = pts1[mask.ravel()==1]
    pts2 = pts2[mask.ravel()==1]


    for i in range (number):
        circles(img1, img2, pts1, pts2, i)
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:number], None, flags = 2)

    img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
    plt.imshow(img3)
    plt.show()
    return pts1[0:5], pts2[0:5]

def reshape(img1, img2):
    height1, width1, _ = img1.shape
    height2, width2, _ = img2.shape

    height = min(height1, height2)
    width = min(width1, width2)

    img1 = img1[0:height, 0:width]
    img2 = img2[0:height, 0:width]

    return img1, img2


def normalize_depth(world_coordinates,image_option):
    print("Normalizing depth...")
    X,Y,Z = cv2.split(world_coordinates)
    if image_option ==  'm':
        Z[Z > MAX_VIS_DISTANCE_M] = MAX_VIS_DISTANCE_M
    else:
        Z[Z > MAX_VIS_DISTANCE_P] = MAX_VIS_DISTANCE_P
    
    Z = cv2.normalize(src=Z, dst=Z, beta=0, alpha=254, norm_type=cv2.NORM_MINMAX)
    Z[Z == 0] = 255

    world_coordinates= np.uint8(cv2.merge((X,Y,Z)))
    return world_coordinates

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
    X = -const*((xL + xR) / deltaX)
    Y = -const*(np.transpose(yL + yR) / deltaX)
    const = baseline * focal_length
    Z = -const / deltaX
    world_coordinates = cv2.merge((X,Y,Z))
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
