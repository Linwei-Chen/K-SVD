#coding:UTF-8
import numpy as np
from sklearn import linear_model
import cv2


class KSVD(object):
    def __init__(self, n_components=256, max_iter=30, tol=1e-6,
                 n_nonzero_coefs=None):
        """
        稀疏模型Y = DX，Y为样本矩阵，使用KSVD动态更新字典矩阵D和稀疏矩阵X
        :param n_components: 字典所含原子个数（字典的列数）
        :param max_iter: 最大迭代次数
        :param tol: 稀疏表示结果的容差
        :param n_nonzero_coefs: 稀疏度
        """
        self.dictionary = None
        self.sparsecode = None
        self.max_iter = max_iter
        self.tol = tol
        self.n_components = n_components
        self.n_nonzero_coefs = n_nonzero_coefs

    def _initialize(self, y):
        """
        用随机二阶单位范数初始化字典矩阵
        """
        shape=[64,self.n_components]
        #对每一列归一化为L2-norm
        self.dictionary = np.random.random(shape)
        for i in range(shape[1]):
            self.dictionary[:, i]=self.dictionary[:, i]/np.linalg.norm(self.dictionary[:, i])

    def _update_dict(self, y, d, x):
        """
        使用KSVD更新字典的过程
        """
        for i in range(self.n_components):
            index = np.nonzero(x[i, :])[0]
            if len(index) == 0:
                continue
            d[:, i] = 0
            r = (y - np.dot(d, x))[:, index]
            u, s, v = np.linalg.svd(r, full_matrices=False)
            d[:, i] = u[:, 0].T
            x[i, index] = s[0] * v[0, :]
        return d, x


    def fit(self, img):
        """
        KSVD迭代过程
        """
        #以防图片不是256*256，先进行一reshape
        img=cv2.resize(img,(256,256),img)
        print img.shape, type(img)

        #将图像按8*8的块转化列向量，合起来成为64*1024的矩阵
        #img保存原始图像转化的矩阵，y用于保存img减去列均值后的矩阵
        y=np.zeros((8*8, 32*32))
        img_reshape=np.zeros((8*8, 32*32))

        patch_num=(256/8)**2
        for patch_index in range(patch_num):
            #按先行后列，将图片分解成32*32个8*8的小块并装换为列向量
            r=(patch_index/32)*8
            c=(patch_index%32)*8
            patch=img[r:r+8, c:c+8].flat
            normalize=np.linalg.norm(patch)
            mean=np.sum(patch)/64
            #print mean
            img_reshape[:, patch_index]=patch
            #y[:, patch_index]=(patch/mean)
            y[:, patch_index]=(patch-mean*np.ones(64))/normalize

        #字典初始化
        self._initialize(y)

        for i in range(self.max_iter):
            #linear_model.orthogonal_mp 用法详见：
            #http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.orthogonal_mp.html
            x = linear_model.orthogonal_mp(self.dictionary, y, n_nonzero_coefs=self.n_nonzero_coefs)#OMP
            e = np.linalg.norm(y- np.dot(self.dictionary, x))
            print '第%s次迭代，误差为：%s' %(i, e)
            if e < self.tol:
                break
            self._update_dict(y, self.dictionary, x)

        #
        src_rec=np.zeros(img.shape)
        for patch_index in range(patch_num):
            x = linear_model.orthogonal_mp(self.dictionary, y[:, patch_index], n_nonzero_coefs=self.n_nonzero_coefs)
            #x = linear_model.orthogonal_mp(self.dictionary, img_reshape[:, patch_index], n_nonzero_coefs=self.n_nonzero_coefs)
            nomalize=np.linalg.norm(img_reshape[:, patch_index])
            mean=np.sum(img_reshape[:, patch_index])/64
            #patch=np.dot(self.dictionary, x)+mean*np.ones(64)
            patch=np.dot(self.dictionary, x)*nomalize+mean*np.ones(64)
            r=(patch_index/32)*8
            c=(patch_index%32)*8
            src_rec[r:r+8, c:c+8]=patch.reshape((8,8))

        return self.dictionary, src_rec

    def missing_pixel_reconstruct(self, img):
        img_patchs=img_to_patch(img)
        patch_num=img_patchs.shape[1]
        #patch_dim=img_patchs.shape[0]
        for i in range(patch_num):
            img_col=img_patchs[:, i]
            index = np.nonzero(img_col)[0]
            #对每列去掉丢失的像素值后求平均、二阶范数，将其归一化
            l2norm=np.linalg.norm(img_col[index])
            mean=np.sum(img_col)/index.shape[0]
            img_col_norm=(img_col-mean)/l2norm
            x = linear_model.orthogonal_mp(self.dictionary[index, :], img_col_norm[index].T, n_nonzero_coefs=self.n_nonzero_coefs)
            img_patchs[:, i]=(self.dictionary.dot(x)*l2norm)+mean

        return patch_to_img(img_patchs)


def pixel_miss(ori,per=0.3):
    img=ori.copy()
    shape=img.shape
    #rand=np.random.random(shape)
    n=int(per*shape[0]*shape[1])
    for i in range(n):
        rand_r=int(np.random.random()*shape[0])
        rand_c=int(np.random.random()*shape[1])
        img[rand_r, rand_c]=0
    return img

#高斯噪点
def Gauss_noise(ori,sigma=20):
    #sigma=np.sqrt(sigma)
    img=ori.copy().astype(np.float64)
    shape=img.shape
    img=img+(np.matlib.randn(shape))*sigma
    return img.astype(np.uint8)

#计算PSNR值
def psnr(A, B):
    if (A==B).all(): return 0
    return 10*np.log10(255*255.0/(((A.astype(np.float)-B)**2).mean()))

#将8*8块为列向量的矩阵还原为原矩阵
def patch_to_img(patchs):
    patch_num=patchs.shape[1]
    size=np.sqrt(patch_num).astype(np.int)
    patch_size=np.sqrt(patchs.shape[0]).astype(np.int)
    img=np.zeros((patch_size*size, patch_size*size))
    for i in range(patch_num):
        r=(i/size)*8
        c=(i%size)*8
        img[r:r+8, c:c+8]=patchs[:, i].reshape((8, 8))
    return img

#将图像分割为8*8块作为列向量
def img_to_patch(img):
    patchs=np.zeros((8*8, 32*32))
    blocks_r=img.shape[0]/8
    blocks_c=img.shape[1]/8
    patch_num=blocks_r*blocks_c
    for i in range(patchs.shape[1]):
        #按先行后列，将图片分解成32*32个8*8的小块并装换为列向量
        r=(i/blocks_r)*8
        c=(i%blocks_c)*8
        patch=img[r:r+8, c:c+8].flat
        patchs[:, i]=patch
    return patchs


#读入原图
ori = cv2.imread("/Users/linweichen/Desktop/lena.jpg",0).astype(np.float)
#像素丢失后的图
img=pixel_miss(ori)
#训练字典所用的图
train = cv2.imread("/Users/linweichen/Desktop/house.png",0).astype(np.float)

#展示原图、破坏图、训练用图
cv2.namedWindow("Original")
cv2.imshow("Original",ori.astype(np.uint8))

cv2.namedWindow("Destory")
cv2.imshow("Destory",img.astype(np.uint8))
print '破坏像素点后图像PSNR值：', psnr(ori, img)

cv2.namedWindow("Train")
cv2.imshow("Train", train.astype(np.uint8))

#最大迭代次数设为80
ksvd = KSVD(max_iter=3)
dictionary, src_rec = ksvd.fit(train)

#按块展示字典
cv2.namedWindow("Dictionary")
dictionary=dictionary-np.amin(dictionary)
dictionary=dictionary/np.amax(dictionary)
cv2.imshow("Dictionary", patch_to_img(dictionary))#.astype(np.uint8))

#用训练得到的字典还原图像
cv2.namedWindow("K-SVD_Rec")
img=ksvd.missing_pixel_reconstruct(img)
cv2.imshow("K-SVD_Rec", img.astype(np.uint8))
print '利用训练获得的字典重构图像后PSNR值：', psnr(ori, img)

#用训练得到的字典还原训练用图
cv2.namedWindow("K-SVD")
cv2.imshow("K-SVD",src_rec.astype(np.uint8))
print '用字典重构训练集合信号PSNR：', psnr(train,src_rec)

cv2.waitKey(0)

