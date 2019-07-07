import numpy as np
import cv2 
import urllib.request
from collections import deque
from keras.models import load_model
cen_pts = deque(maxlen=300)
#Url should be written as provided by ip webcam.
#(You can use your inbuilt camera feature also) but few lines of code will change)
#Make sure you open the url in chrome first and do proper setting.
#url = "ipwebcam address/shot.jpg" --> syntax
url='url_address from ip web cam application/shot.jpg'#for e.g http://.....:8080/shot.jpg
#You can use any deep learning model trained on handwritten digits dataset.
model = load_model('CNN_adam_optimizer.h5')
cap = cv2.VideoCapture(0)
cap.release()
cv2.destroyAllWindows()
while True:
	#image decoding process captured from camera of mobile phone.
    imgResp = urllib.request.urlopen(url)
    imgNp = np.array(bytearray(imgResp.read()) ,dtype = np.uint8)
    img = cv2.imdecode(imgNp, -1)
    # draw rectangle on image
    cv2.rectangle(img,(200, 200), (475, 475), (255,0,0), 2) 
    #slice the rectangular box of image into img_rect
    img_rect = img[200:475,200:475]     			
    #output image
    img_out = img.copy()
    #creating a blackbacground image with img_rect  box shape
    black_pic = np.zeros(img_rect.shape,dtype=np.uint8)
    #converting the image into gray
    gray = cv2.cvtColor(img_rect,cv2.COLOR_BGR2GRAY) 
    #Applying Gaussian blurring to only img_rect for noise removal
    blurred = cv2.GaussianBlur(gray,(5,5),0)
    #Capturing circle with certain min parameter in img_rect
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT,1,10,param1=50,param2=30,minRadius=20,maxRadius=30)
    #TO check whether there is any circle detect or not with given min distance 
    if circles is not None:
					# convert the (x, y) coordinates and radius of the circles to integers
					circles = np.round(circles[0, :]).astype("int")
					# loop over the (x, y) coordinates and radius of the circles
					for (x, y, r) in circles:
						cen_pts.appendleft((x,y)) #append centers

						for i in range(1, len(cen_pts)):

									if cen_pts[i - 1] is None or cen_pts[i] is None:
										continue
									#draw circle with x,y cordinates and  r radius
									cv2.circle(img_rect, (x, y), r, (0, 255, 0), 4)
									#for center of circle draw a point circle with radius 3
									cv2.circle(img_rect, (x,y),3, (0, 128, 255), 4)
									#plotting line using two points
									cv2.line(img_rect, cen_pts[i - 1], cen_pts[i], (0, 0, 255), 3) 
									#plotting line using two points in blackbackground image
									cv2.line(black_pic, cen_pts[i - 1], cen_pts[i], (255, 255, 255), 5)
									#replacing the slice of output image with the img_rect  
									img_out[200:475,200:475] = img_rect
    #converting from rgb value (3 color plane to one plane graylevel image)
    black_pic2gray = cv2.cvtColor(black_pic, cv2.COLOR_BGR2GRAY)
    GB = cv2.GaussianBlur(black_pic2gray, (3, 3), 0)
    #cv2.imshow("GB",GB)
    __,threshold  = cv2.threshold(GB,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #cv2.imshow("threshold",threshold)
    blackpic_cnts = cv2.findContours(threshold.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
    #print([cv2.contourArea(i) for i in blackpic_cnts[1:]])
    if len(blackpic_cnts) >1:
                    cnt = max(blackpic_cnts, key=cv2.contourArea)
                    #print("area",cv2.contourArea(cnt))
                    if cv2.contourArea(cnt) >2000:
                        x, y, w, h = cv2.boundingRect(cnt)
                        cv2.rectangle(img_rect,(x,y),(x+w,y+h),(0,255,0),2)

                        digit_pic = black_pic2gray[y:y + h, x:x + w]
                        #cv2.imshow("digit_pic",digit_pic)
                        image = cv2.resize(digit_pic, (28, 28))
                        #cv2.imshow("image_re",image)
                        image = np.array(image)
                        image = image.flatten()
                        image = image.reshape(image.shape[0],1)
                        image = image.reshape(1,28,28,1)
                        
                        pred = model.predict(image)
                        pred_value = pred.argmax()
                        cv2.putText(img_out, "Prediction : " + str(pred_value), (200, 190),
                    				cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
     
    
    cv2.imshow("Frame", img_out)
    if ord('q')==cv2.waitKey(10): exit(0)
    
 


