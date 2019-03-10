import numpy as np
from sklearn import linear_model
import cv2

img = cv2.imread("/Users/linweichen/Desktop/lena_noise.jpg",0).astype(np.float)
img_dct=cv2.dct(img)
cv2.namedWindow("Noise")
cv2.imshow("Noise",img.astype(np.uint8))
mask_mat=np.zeros(img.shape)
mask_mat[0:mask_mat.shape[0]/3, 0:mask_mat.shape[1]/3]=1
img=cv2.idct(img_dct*mask_mat)
cv2.namedWindow("DCT")
cv2.imshow("DCT",img.astype(np.uint8))

#https://blog.csdn.net/dugudaibo/article/details/78701487
MM = 8
NN = 16
print np.arange(0, MM).shape, type(np.arange(0, MM))
np.dot(np.arange(0, MM).T, np.arange(0,NN))
V = np.sqrt(2/MM) * np.cos(np.dot(np.arange(0, MM).T, np.arange(0,NN))*3.1415926/MM/2);
V[1,:] = V[1, :]/np.sqrt(2);
DCT=np.kron(V, V)
DCT=DCT-np.amin(DCT)
DCT=DCT/np.amax(DCT)
cv2.namedWindow("DCT_DIC")
cv2.imshow("DCT_DIC",DCT)

cv2.waitKey(0)


