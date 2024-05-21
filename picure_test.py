import cv2
import numpy as np
import imutils
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import time 

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) / 2, (ptA[1] + ptB[1]) / 2)

# Đường dẫn của ảnh
image_path = "data\kenvang4.png"  
frame = cv2.imread(image_path)

ref_width_in_cm = 2  
pixelsPerMetric = None  

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
gray = clahe.apply(gray)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)  
edges = cv2.Canny(blurred, 30, 100) 
edged = cv2.dilate(edges, None, iterations=1) 
edged = cv2.erode(edged, None, iterations=1)  

cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

if cnts:
    (cnts, _) = contours.sort_contours(cnts)  

    for c in cnts:
        if cv2.contourArea(c) < 100:
            continue

        box = cv2.minAreaRect(c)
        box = cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        box = perspective.order_points(box)

        # Xác định các điểm giữa
        (tl, tr, br, bl) = box
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)
        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)

        d1 = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))  
        d2 = dist.euclidean((tlblX, tlblY), (trbrX, trbrY)) 

        if pixelsPerMetric is None:
            pixelsPerMetric = d2 / ref_width_in_cm  
            
        if d1 > d2:
            dimA = d1
            dimB = d2
        else:
            dimA = d2
            dimB = d1

        dimA_cm = dimA / pixelsPerMetric
        dimB_cm = dimB / pixelsPerMetric

        if dimA_cm > 6:
            continue

        cv2.drawContours(frame, [box.astype("int")], -1, (0, 255, 0), 2)
        cv2.putText(frame, "{:.1f} cm".format(dimA_cm), (int(tltrX - 15), int(tltrY - 10)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(frame, "{:.1f} cm".format(dimB_cm), (int(trbrX + 10), int(trbrY - 10)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

cv2.imshow("Image size", frame)
cv2.imshow("Image edged", edged)

cv2.waitKey(0)
cv2.destroyAllWindows()
