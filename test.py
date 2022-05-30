import numpy as np
from PIL import Image
# Initializing value of x-axis and y-axis
# in the range -1 to 1
im = Image.open("tree-736885__480.jpg")
image =  np.asarray(im)
print(image[0][0])
# x, y = np.meshgrid(np.linspace(-1,1,10), np.linspace(-1,1,10))
# dst = np.sqrt(x*x+y*y)
 
# # Initializing sigma and muu
# sigma = 1
# muu = 0.000
 
# # Calculating Gaussian array
# gauss = np.exp(-( (dst-muu)**2 / ( 2.0 * sigma**2 ) ) )
 
# print("2D Gaussian array :\n")
# print(gauss)image