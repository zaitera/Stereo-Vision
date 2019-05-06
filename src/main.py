import argparse
from utils import *


def mainReq1(image_option):
    if image_option ==  'm':
        aux_directory = './data/Middlebury/Motorcycle-perfect/im'
        mindisp = 0
        maxdisp = 336
        intrinsics_L=np.asarray([[3979.911, 0, 1244.772], [0, 3979.911, 1019.507], [ 0, 0, 1]])
        intrinsics_R=np.asarray([[3979.911, 0, 1369.115], [0, 3979.911, 1019.507], [0, 0, 1]])
        doffs=124.343
        baseline =  193.001
        focal_length = 3979.911
        pass
    else:
        aux_directory = './data/Middlebury/Jadeplant-perfect/im'
        mindisp = 16
        maxdisp = 496
        intrinsics_L=np.asarray([[7315.238, 0, 997.555], [0, 7315.238, 980.754],[ 0, 0, 1]])
        intrinsics_R=np.asarray([[7315.238, 0, 1806.75], [ 0, 7315.238, 980.754], [0, 0, 1]])
        doffs=809.195
        baseline =  380.135
        focal_length = 7315.238
        pass
    print('loading images...')
    imgL = cv2.imread(aux_directory+'L.png')  # downscale images for faster processing
    imgR = cv2.imread(aux_directory+'R.png')

    #calculates disparity and filters the result
    disp = calculateDisparity(imgL,imgR,mindisp,maxdisp)
    
    width = imgL.shape[1]
    height = imgL.shape[0]
    
    world_coordenates = calcWorldCoordinates(height, width, focal_length,baseline,disp)
    world_coordenates = normalize_depth(world_coordenates,image_option)

    __, __, Z = cv2.split(world_coordenates)
    if image_option ==  'm':
        cv2.imwrite('./data/Middlebury/Motorcycle-perfect/disparity.pgm',disp)
        cv2.imwrite('./data/Middlebury/Motorcycle-perfect/depth.png',Z)
    else:
        cv2.imwrite('./data/Middlebury/Jadeplant-perfect/disparity.pgm',disp)
        cv2.imwrite('./data/Middlebury/Jadeplant-perfect/depth.png',Z)
    
    mouse_tracker = MouseClick('image left', True)
    cv2.imshow('image left', imgL)
    aux = 0
    while(True):
        if mouse_tracker.clicks_number > 0 and aux is not mouse_tracker.clicks_number:
            print("world coordenates [ X Y Z ] = ",world_coordenates[mouse_tracker.yi][mouse_tracker.xi])
            aux = mouse_tracker.clicks_number
            pass
        if(cv2.waitKey(10) & 0xFF == 27):
            break
        pass




   
def mainReq2():
    aux_directory = './data/FurukawaPonce/Morpheus'
    fc1 = [6704.926882, 6705.241311]
    cc1 = [738.251932, 457.560286]
    alpha_c1 = 0.000103
    R1 = np.asarray([[0.70717199,  0.70613396, -0.03581348],
                    [0.28815232, -0.33409066, -0.89741388], 
                    [-0.64565936,  0.62430623, -0.43973369]])
    Tc1 = np.array([-532.285900, 207.183600, 2977.408000])

    fc2 = [6682.125964, 6681.475962]
    cc2 = [875.207200, 357.700292]
    alpha_c2 = 0.000101
    R2 = np.asarray([[0.48946344,  0.87099159, -0.04241701], 
                    [0.33782142, -0.23423702, -0.91159734], 
                    [-0.80392924,  0.43186419, -0.40889007]])
    Tc2 = np.array([-614.549000, 193.240700, 3242.754000])

    print('loading images...')
    imgL = cv2.imread(aux_directory+'L.jpg')  # downscale images for faster processing
    imgR = cv2.imread(aux_directory+'R.jpg')

    imgL, imgR = reshape(imgL, imgR)

    p1, p2 = matching(imgL, imgR, 20)


    Il = calculateIntrinsicMatrix(fc1, cc1, alpha_c1)
    Ir = calculateIntrinsicMatrix(fc2, cc2, alpha_c2)

    h,w,_ = imgL.shape
    #Tc3 = Tc1 - Tc2
    #R3 = np.dot(R1, R2)

    #aux = np.cross(Tc3, R3)
    #F = np.dot(np.linalg.pinv(Il), aux)
    #F = np.dot(F, np.linalg.inv(Ir))
    #print("\n\n", F)
    H1, H2, camM1, camM2 = retify(Il, Ir, R1, Tc1, R2, Tc2, h, w)
    ones = np.ones((5,1), dtype=int)
    p2 = np.hstack((p2, ones))
    p2[:,0] = p2[:,0] - 1850
    p1 = np.hstack((p1, ones))
    PosReal = np.dot(np.linalg.pinv(H2), p2[4])
    EstimatedL = np.dot(H1, PosReal)

    print("\n\n\nPosição na imagem da esquerda em Pixel:", p1[4])
    print("Posição adquirida pela projeção: ",EstimatedL)


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
    elif (int(args["r"]) == 2):
        mainReq2()
        pass
    elif (int(args["r"]) == 3):
        mainReq3()
        pass
    else:
        raise NameError("Parameter r is only between 1 and 3")