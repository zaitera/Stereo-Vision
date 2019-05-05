import argparse
from common import *

def mainReq1(image_option):
    if image_option ==  'm':
        aux_directory = './data/Middlebury/Motorcycle-perfect/im'
        mindisp = 0
        maxdisp = 336
        intrinsics_L=np.asarray([[7315.238, 0, 997.555], [0, 7315.238, 980.754],[ 0, 0, 1]])
        intrinsics_R=np.asarray([[7315.238, 0, 1806.75], [ 0, 7315.238, 980.754], [0, 0, 1]])
        doffs=124.343
        baseline =  193.001
        focal_length = 3979.911
        pass
    else:
        aux_directory = './data/Middlebury/Jadeplant-perfect/im'
        mindisp = 16
        maxdisp = 496
        intrinsics_L=np.asarray([[3979.911, 0, 1244.772], [0, 3979.911, 1019.507], [ 0, 0, 1]])
        intrinsics_R=np.asarray([[3979.911, 0, 1369.115], [0, 3979.911, 1019.507], [0, 0, 1]])
        doffs=809.195
        baseline =  380.135
        focal_length = 7315.238
        pass
    print('loading images...')
    imgL = cv2.imread(aux_directory+'L.png')  # downscale images for faster processing
    imgR = cv2.imread(aux_directory+'R.png')


    #calculates disparity and filters the result
    disp = calculateDisparity(imgL,imgR,mindisp,maxdisp)
    cv2.namedWindow('disparity', cv2.WINDOW_NORMAL)
    cv2.imshow('disparity',disp)

    mouse_tracker = MouseClick('image left', True)
    cv2.imshow('image left', imgL)
    aux = 0
    while(True):
        if mouse_tracker.clicks_number > 0 and aux is not mouse_tracker.clicks_number:
            world_coordinates = calcWorldCoordinates(mouse_tracker.xi,mouse_tracker.yi, focal_length, baseline, disp)
            print("world coordenates [X, Y, Z] = ",world_coordinates)
            aux = mouse_tracker.clicks_number
            pass
        if(cv2.waitKey(10) & 0xFF == 27):
            break
        pass




   
def mainReq2():
    pass
def mainReq3():
    pass

if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("--r", "-requirement", default=1,
        help="Which requirement you would like to evaluate?")
    ap.add_argument("--i", "-images", default='m',
        help="Which images you would like to use? m for motorcycle and p for plant")

    args = vars(ap.parse_args())
    if args["i"] is not 'm' and args["i"] is not 'p' :
        raise NameError("for r1, parameter i can only be m or p")
    if (int(args["r"]) == 1):
        mainReq1(args["i"])
        pass
    elif (int(args["r"]) == 1):
        mainReq2()
        pass
    elif (int(args["r"]) == 1):
        mainReq3()
        pass
    else:
        raise NameError("Parameter r is only between 1 and 3")