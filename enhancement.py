import cv2 
import matplotlib.pyplot as plt 
import numpy as np 
  
# Load the image 
image = cv2.imread('/Users/yuanweiyun/Desktop/EE6222/valid_data/2/202_0.png') 
  
#Plot the original image 
plt.subplot(1, 2, 1) 
plt.title("Original") 
plt.imshow(image) 
  
# Adjust the brightness and contrast 
# Adjusts the brightness by adding 10 to each pixel value 
brightness = 20
# Adjusts the contrast by scaling the pixel values by 2.3 
contrast = 2.3  
image2 = cv2.addWeighted(image, contrast, np.zeros(image.shape, image.dtype), 0, brightness) 
  
#Save the image 
cv2.imwrite('modified_image.jpg', image2) 
#Plot the contrast image 
plt.subplot(1, 2, 2) 
plt.title("Brightness & contrast") 
plt.imshow(image2) 
plt.show()



