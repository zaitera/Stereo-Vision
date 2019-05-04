import cv2
class MouseClick:
    xi = int
    xf = int
    yi = int
    yf = int
    clicks_number = int
    single_click = bool
    def __init__(self, name, single_click):
        cv2.namedWindow(name,cv2.WINDOW_GUI_NORMAL)
        cv2.setMouseCallback(name, self.click)
        self.clicks_number = 0
        self.single_click = single_click

    def click(self, event, x, y, flags, param):
        # if the left mouse button was clicked, record the starting
        # (x, y) coordinates and indicate that cropping is being
        # performed
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.clicks_number == 0 or self.single_click:
                self.xi = x
                self.yi = y
                self.clicks_number += 1
                print("First click captured ({},{})".format(x,y))

            elif self.clicks_number == 1 and not self.single_click:
                self.xf = x
                self.yf = y
                self.clicks_number += 1
                print("Second click captured ({},{})".format(x,y))
                #self.calc_euclidian_distance()

            else:
                print("Two clicks captured")
                pass 
