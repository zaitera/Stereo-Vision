import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def matching(img1, img2):
    orb = cv.ORB_create()

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck = True)

    matches = bf.match(des1,des2)
    matches = sorted(matches, key = lambda x:x.distance)


    img3 = cv.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags = 2)
    #cv.imshow('Final', img3)
    plt.imshow(img3)
    plt.show()
    #cv.waitKey(0)
    #cv.destroyAllWindows 


if __name__ == '__main__':

    while True:
        name = input("Select 'm' for the motorcycle or 'p' for the plant image:")
        if name == 'm' or name == 'p':
            break

    img1 = cv.imread('im0_' + name + '.png',0)
    img2 = cv.imread('im1_' + name + '.png',0)

    matching(img1, img2)


