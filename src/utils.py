import numpy as np
import matplotlib.pyplot as plt
from MouseClick import *

MAX_VIS_DISTANCE_M = 19700

MAX_VIS_DISTANCE_P = 85000

fc2= [6682.125964, 6681.475962]
cc2= [875.207200, 357.700292]
alpha_c2 = 0.000101
R2 = np.asarray([[0.48946344,  0.87099159, -0.04241701], 
                [0.33782142, -0.23423702, -0.91159734], 
                [-0.80392924,  0.43186419, -0.40889007]])
Tc2= np.array([-614.549000, 193.240700, 3242.754000])

def realDistanceCalculator(camera_matrix,extrinsics,x,y):
    pseudo_inv_extrinsics = np.linalg.pinv(extrinsics)
    intrinsics_inv = np.linalg.inv(camera_matrix)
    pixels_matrix = np.array((x,y,1))
    ans = np.matmul(intrinsics_inv,pixels_matrix)
    ans = np.matmul(pseudo_inv_extrinsics,ans)
    ans /= ans[-1] 
    return ans

def distanceBetweenTwoPixels(pixel1,pixel2, intrinsics, extrinsics):
    p1 = realDistanceCalculator(intrinsics, extrinsics, pixel1[0], pixel2[1])
    p2 = realDistanceCalculator(intrinsics, extrinsics, pixel2[0], pixel2[1])
    aux = p2 - p1
    pixel1.clear()
    pixel2.clear()
    return aux

def retify(Il, Ir, R1, Tc1, R2, Tc2, h, w):
    El = calculateExtrinsicMatrix(R1, Tc1)
    cameraMatrix1 = np.dot(Il,El)
    Er = calculateExtrinsicMatrix(R2, Tc2)
    cameraMatrix2 = np.dot(Ir,Er)

    c1 = np.dot(np.linalg.inv(cameraMatrix1[:, 0:3]),cameraMatrix1[:,3])
    c2 = np.dot(np.linalg.inv(cameraMatrix2[:, 0:3]),cameraMatrix2[:,3])

    v1 = (c1-c2)
    v2 = np.cross(R1[2],v1)
    v3 = np.cross(v1,v2)

    R = np.array([np.transpose(v1)/np.linalg.norm(v1), np.transpose(v2)/np.linalg.norm(v2), 
                  np.transpose(v3)/np.linalg.norm(v3)])

    A = Il+Ir
    A = A/2
    A[0,1] = 0   

    aux1 = np.hstack((R, (np.dot(-R, c1)).reshape(3,1)))
    aux2 = np.hstack((R, (np.dot(-R, c1)).reshape(3,1)))
    Pn1 = np.dot(A, aux1)
    Pn2 = np.dot(A, aux2)
    T1 = np.dot(Pn1[:,0:3], np.linalg.inv(cameraMatrix1[:, 0:3]))
    T2 = np.dot(Pn2[:,0:3], np.linalg.inv(cameraMatrix2[:, 0:3]))
    
    return T1, T2, cameraMatrix1, cameraMatrix2
    #R = np.array([[],[],[]])
    #H1, H2 = cv.StereoRectify(cameraMatrix1, cameraMatrix2, (0,0,0,0,0), (0,0,0,0,0), (h, w), 
    #                 R, T, R1, R2, P1, P2, Q=None, flags=CV_CALIB_ZERO_DISPARITY, alpha=-1, newImageSize=(0, 0))
  
def calculateIntrinsicMatrix(focal_length, princPoint, skew):
    A = np.zeros((3,3))
    A[0,0] = focal_length[0]
    A[0,1] = skew
    A[0,2] = princPoint[0]
    A[1,1] = focal_length[1]
    A[1,2] = princPoint[1]
    A[2,2] = 1
    return A

def calculateExtrinsicMatrix(R, Tc):
    B = np.zeros((3,4))
    B = np.array([[R[0,0], R[0,1], R[0,2], Tc[0]], [R[1,0], R[1,1], R[1,2] , Tc[1]], [R[2,0], R[2,1], R[2,2], Tc[2]]])
    return B


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
