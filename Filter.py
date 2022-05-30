import math
import numpy as np
import cv2
import random
class Filter():
	def __init__(self):
		self.hSobelMask = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
		self.vSobelMask = np.array([[-1,-2,-1], [0,0,0], [1,2,1]])
		self.laplaceMask = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
		self.laplaceMask2 = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
	def gausseMask(self,a):
		matrix = []
		for u in range(-1,2):
			matrix.append([])
			for v in range(-1,2):
				matrix[u+1].append((1/(2*a*a*math.pi))*math.exp(-((u*u+v*v)/a*a)))
		return np.array(matrix)

	def doGMask(self,a):
		matrix = []
		for u in range(-1,2):
			matrix.append([])
			for v in range(-1,2):
				r = u*u + v*v
				p = a*a
				e = math.exp(-(r/p))
				matrix[u+1].append((-u*e/(p*p*math.pi)))
		return np.array(matrix)

	def loGMask(self,a):
		matrix = []
		for u in range(-1,2):
			matrix.append([])
			for v in range(-1,2):
				r = u*u + v*v
				p = a*a
				matrix[u+1].append(-((r-p)/(p*p*p*math.pi))*math.exp(-(r/p)))
		print(matrix)
		return np.array(matrix)

	def median_filter(self,img):
		img_new = np.zeros([len(img), len(img[0])])
		for i in range(1,len(img)-1):
			for j in range(1,len(img[0])-1):
				temp = [img[i-1][j-1],
				img[i-1][j],
				img[i-1][j+1],
				img[i][j-1],
				img[i][j],
				img[i][j+1],
				img[i+1][j],
				img[i+1][j-1],
				img[i+1][j+1]]
				a = sorted(temp)
				img_new[i][j] = a[4]
		return img_new.astype(np.uint8)
	def max_filter(self,img):
		img_new = np.zeros([len(img), len(img[0])])
		print(len(img))
		for i in range(1,len(img)-1):
			for j in range(1,len(img[0])-1):
				temp = [img[i-1][j-1],
				img[i-1][j],
				img[i-1][j+1],
				img[i][j-1],
				img[i][j],
				img[i][j+1],
				img[i+1][j],
				img[i+1][j-1],
				img[i+1][j+1]]
				img_new[i][j] = max(temp)
		return img_new.astype(np.uint8)

	def min_filter(self,img):
		img_new = np.zeros([len(img), len(img[0])])
		for i in range(1,len(img)-1):
			for j in range(1,len(img[0])-1):
				temp = [img[i-1][j-1],
				img[i-1][j],
				img[i-1][j+1],
				img[i][j-1],
				img[i][j],
				img[i][j+1],
				img[i+1][j],
				img[i+1][j-1],
				img[i+1][j+1]]
				img_new[i][j] = min(temp)
		return img_new.astype(np.uint8)



	def convol(self,img,mask):
		img_new = np.zeros([len(img), len(img[0])])
		for x in range(1,len(img)-1):
			for y in range(1,len(img[0])-1):
				img_new[x][y] =	img[x-1][y-1] * mask[0][0]\
							+ img[x-1][y] * mask[0][1]\
							+ img[x-1][y+1] * mask[0][2]\
							+ img[x][y-1] * mask[1][0]\
							+ img[x][y] * mask[1][1]\
							+ img[x][y+1] * mask[1][2]\
							+ img[x+1][y-1] * mask[2][0]\
							+ img[x+1][y] * mask[2][1]\
							+ img[x+1][y+1] * mask[2][2]
		return img_new.astype(np.uint8)
	def noise(self, n, img, a):
		img_new = np.copy(img)
		for x in range(n):
			x = random.randint(0,len(img)-1)
			y = random.randint(0,len(img[0])-1)
			img_new[x][y] = a
		return img_new

	def gausse_filter(self,img):
		# return self.convol(img,self.gausseMask(1))
		return cv2.GaussianBlur(img,(5,5),-2)
	def laplace_filter(self,img,a):
		# return self.convol(img,self.laplaceMask2) if (a==1) else self.convol(img,self.laplaceMask)
		return cv2.filter2D(img,-1,self.laplaceMask) if (a!=1) else cv2.filter2D(img,-1,self.laplaceMask2)
	def sobel_filter(self,img,a):
		# return img + (self.convol(img,self.vSobelMask) + self.convol(img,self.hSobelMask)) if a==1 else (self.convol(img,self.vSobelMask) + self.convol(img,self.hSobelMask))
		return (cv2.filter2D(img,-1,self.vSobelMask) + cv2.filter2D(img,-1,self.hSobelMask)) if (a!=1) else img + (cv2.filter2D(img,-1,self.vSobelMask) + cv2.filter2D(img,-1,self.hSobelMask))
	def LoG_filter(self,img):
		# return self.convol(img,self.loGMask(1))
		return (cv2.filter2D(img,-1,self.loGMask(1)))
	def pepper(self,img):
		# img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

		# equalize the histogram of the Y channel
		# img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

		# convert the YUV image back to RGB format
		# return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
		return self.noise(100,img,255)


	def thresholding(self,img):
		max = 126
		img_new = np.zeros([len(img), len(img[0])])
		for x in range(len(img)):
			for y in range(len(img[0])):
				if img[x][y] >= max:
					img_new[x][y] = 0
				else: img_new[x][y] = 255
		return img_new.astype(np.uint8)
	# math_on_pixel

	# def pixel_Reverse(self,img):
	# 	img_new = np.zeros([len(img), len(img[0])])
	# 	for x in range(len(img)):
	# 		for y in range(len(img[0])):
	# 			img_new[x][y] = 255 - img[x][y]
	# 	return img_new.astype(np.uint8)

	def pixel_Reverse(self,img):
		img_new = 255 - img
		return img_new.astype(np.uint8)

	def log_Transforms(self,img):
		img_new = np.zeros([len(img), len(img[0])])
		c = 255/math.log(1+255)-0.2
		for x in range(len(img)):
			for y in range(len(img[0])):
				img_new[x][y] = c*math.log(1 + img[x][y])
		return img_new.astype(np.uint8)

	def gamma_Transforms(self,img,γ):
		img_new = np.zeros([len(img), len(img[0])])
		c = 1
		for x in range(len(img)):
			for y in range(len(img[0])):
				img_new[x][y] = c*math.pow(img[x][y],γ)
				
		return img_new.astype(np.uint8)

	def bit_Plane_Slicing(self,img):
		import matplotlib.pyplot as plt
		fig = plt.figure(figsize=(16, 9))
		(ax1, ax2, ax3), (ax4, ax5, ax6), (ax7,ax8,ax9) = fig.subplots(3, 3)
		temp = []
		for i in range(len(img)):
			for j in range(len(img[0])):
				temp.append(np.binary_repr(img[i][j],width=8)) 
		bit = [[], [], [], [], [], [], [], []]
		for z in temp:
			bit[0].append(int(z[0])) #bit 8
			bit[1].append(int(z[1])) #bit 7
			bit[2].append(int(z[2]))#bit 6
			bit[3].append(int(z[3]))#bit 5
			bit[4].append(int(z[4]))#bit 4
			bit[5].append(int(z[5]))#bit 3
			bit[6].append(int(z[6]))#bit 2
			bit[7].append(int(z[7]))#bit 1
		ax = [ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9]
		for x in range(len(bit)):
			bit[x] = (np.array(bit[x],dtype='uint8')*pow(2,8-x-1)).reshape(len(img),len(img[0]))
			ax[x].imshow(bit[x], cmap='gray')
			ax[x].set_title("bit "+str(8-x))
		ax[8].imshow(bit[0] + bit[1] + bit[2] + bit[3], cmap='gray')
		ax[8].set_title("bit 7 + 8")
		plt.show()	
			

	def equal_histogram(self,img):
		import collections
		img_new = np.zeros([len(img), len(img[0])])
		temp = {}
		sum = len(img)*len(img[0])
		for x in img:
			for y in x:
				temp[y] = temp.get(y, 0) +1 #get h(rk)
		
		temp = collections.OrderedDict(sorted(temp.items()))
		print(sum)
		prk = {}
		for x,y in temp.items():
			prk[x] = y/sum #get p(rk)
		prk=collections.OrderedDict(sorted(prk.items()))
		s = 0
		for x,y in prk.items():
			s = s+y
			prk[x] = round(255*s) #equalize histogram
		for x,y in prk.items():
			for i in range(len(img)):
				for j in range(len(img[0])):
					if img[i][j] == x:
						img_new[i][j] = y
		return img_new.astype(np.uint8)	
	