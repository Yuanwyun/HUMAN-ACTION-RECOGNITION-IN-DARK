#Import the necessary libraries 
import cv2 
import matplotlib.pyplot as plt 
import numpy as np 
  
# Load the image 
image = cv2.imread('PATH') 
  
#Plot the original image 
plt.subplot(1, 2, 1) 
plt.title("Original") 
plt.imshow(image) 
  
# Adjust the brightness and contrast  
# g(i,j)=α⋅f(i,j)+β 
# control Contrast by 1.5 
alpha = 1.5  
# control brightness by 50 
beta = 30  
image2 = cv2.convertScaleAbs(image, alpha=alpha, beta=beta) 
  
#Save the image 
cv2.imwrite('Brightness & contrast.jpg', image2) 
#Plot the contrast image 
plt.subplot(1, 2, 2) 
plt.title("Brightness & contrast") 
plt.imshow(image2) 
plt.show() 
