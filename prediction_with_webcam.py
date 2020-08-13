
from keras.models import load_model
import cv2


model = load_model('model.h5')
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
video = cv2.VideoCapture(0)

import numpy as np
from keras.preprocessing import image

i=0
while True:
    ret , img = video.read()
    cv2.imwrite("test/test.jpg",img)
    test_image = image.load_img('test/test.jpg', target_size = (64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = model.predict(test_image)
    print(result)
    if result[0][0] == 1.0:
        out = "Ten"
        print(10)
    elif result[0][1] == 1.0:
        out = "Twenty"
        print(20)
    elif result[0][2] == 1.0:
        out = "Two Hundered"
        print(200)
    elif result[0][3] == 1.0:
        out = "Five Hundered"
        print(500)
    else:
        out = "No Note Found"
    
    top_left = [100, 110]
    bottom_right = [110, 120]
    color = [0,255,0]
    pos = (90,90)
    #cv2.rectangle(image,top_left,bottom_right,color,cv2.FILLED)
    cv2.putText(img,out, pos, cv2.FONT_HERSHEY_PLAIN, 2,(0,0,200),3)
    cv2.imshow("image",img)
    cv2.waitKey(1)
    
