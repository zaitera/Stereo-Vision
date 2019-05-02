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
    
    disp = calculateDisparity(imgL,imgR,mindisp,maxdisp)
    #disp = calculateDisparity2(imgL,imgR)
    print("HERE",imgL.shape[0:2])
    width = imgL.shape[1]
    height = imgL.shape[0]
    Q = np.array([
        [1, 0, 0, -width/2],
        [0, 1, 0, -height/2],
        [0, 0, 0, focal_length],
        [0, 0, -1/baseline, 0]
    ])
    print(Q)
    points,colors = calculatePointCloud(imgL,disp, Q)
    print(points)
    #identity matrix because theres no visible rotation between the two images
    r = np.eye(3)
    #1000 to consider milimeters, 100 for centimeters
    t = np.array([0, 0, -1000.0])
    k = np.array([[focal_length, 0, width/2],
        [0, focal_length, height/2],
        [0, 0, 1]])
    # source of images didnt inform any distortion, considering distortions to be zero
    dist_coeff = np.zeros((4, 1))

    def view(r, t):
        aux = calculateProjectedImage(
            points, colors, r, t, k, dist_coeff, width, height
        )
        cv2.namedWindow('projected',cv2.WINDOW_GUI_NORMAL )
        cv2.imshow('projected',aux )
    
    cv2.namedWindow("both",cv2.WINDOW_NORMAL )
    cv2.imshow("both", np.hstack((imgL, imgR)))

    view(r,t)
    while(True):
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