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
        初始化字典矩阵
        """

        '''
        u, s, v = np.linalg.svd(y)
        self.dictionary = u[:, :self.n_components]
        print type(u),u.shape

        shape=list(y.shape)
        shape[1]=self.n_components
        '''
        shape=[64,self.n_components]
        #对每一列归一化为L2-norm
        self.dictionary = np.random.random(shape)
        #self.dictionary=y[:, :256]
        #print y.shape
        for i in range(shape[1]):
        #for i in range(256):
            self.dictionary[:, i]=self.dictionary[:, i]/np.linalg.norm(self.dictionary[:, i])
            #print np.linalg.norm(self.dictionary[:, i])

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

        pitch_num=(256/8)**2
        for pitch_index in range(pitch_num):
            #按先行后列，将图片分解成32*32个8*8的小块并装换为列向量
            r=(pitch_index/32)*8
            c=(pitch_index%32)*8
            pitch=img[r:r+8, c:c+8].flat

            normalize=np.linalg.norm(pitch)
            mean=np.sum(pitch)/64
            #print mean
            img_reshape[:, pitch_index]=pitch
            #y[:, pitch_index]=(pitch/mean)
            y[:, pitch_index]=(pitch-mean*np.ones(64))/normalize

        #字典初始化
        self._initialize(y)

        #print self.dictionary.shape, type(self.dictionary)
        #print y.shape, type(y)

        for i in range(self.max_iter):
            #linear_model.orthogonal_mp 用法详见：
            #http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.orthogonal_mp.html
            x = linear_model.orthogonal_mp(self.dictionary, y, n_nonzero_coefs=self.n_nonzero_coefs)#OMP
            e = np.linalg.norm(y- np.dot(self.dictionary, x))
            print '第%s次迭代，误差为：%s' %(i, e)
            if e < self.tol:
                break
            self._update_dict(y, self.dictionary, x)
        '''
        dictionary=np.zeros((128,128))
        dict_cols=self.dictionary.shape[1]
        for i in range(dict_cols):
            #print i
            r=(i/16)*8
            c=(i%16)*8
            pitch=self.dictionary[:, i]
            dictionary[r:r+8, c:c+8]=pitch.reshape((8,8))
        '''
        denoise=np.zeros(img.shape)
        for pitch_index in range(pitch_num):
            x = linear_model.orthogonal_mp(self.dictionary, y[:, pitch_index], n_nonzero_coefs=self.n_nonzero_coefs)
            #x = linear_model.orthogonal_mp(self.dictionary, img_reshape[:, pitch_index], n_nonzero_coefs=self.n_nonzero_coefs)
            nomalize=np.linalg.norm(img_reshape[:, pitch_index])
            mean=np.sum(img_reshape[:, pitch_index])/64
            #pitch=np.dot(self.dictionary, x)+mean*np.ones(64)
            pitch=np.dot(self.dictionary, x)*nomalize+mean*np.ones(64)
            r=(pitch_index/32)*8
            c=(pitch_index%32)*8
            denoise[r:r+8, c:c+8]=pitch.reshape((8,8))

        return self.dictionary, denoise
        #self.sparsecode = linear_model.orthogonal_mp(self.dictionary, y, n_nonzero_coefs=self.n_nonzero_coefs)
        #return self.dictionary, self.sparsecode

def pixel_miss(ori,per=0.1):
    img=ori.copy()
    shape=img.shape
    #rand=np.random.random(shape)
    n=int(per*shape[0]*shape[1])
    for i in range(n):
        rand_r=int(np.random.random()*shape[0])
        rand_c=int(np.random.random()*shape[1])
        img[rand_r, rand_c]=0
    return img

def Gauss_noise(ori,sigma=20):
    #sigma=np.sqrt(sigma)
    img=ori.copy().astype(np.float64)
    shape=img.shape
    img=img+(np.matlib.randn(shape))*sigma
    return img.astype(np.uint8)

def psnr(A, B):
    if (A==B).all(): return 0
    mse=((A.astype(np.float)-B.astype(np.float))**2).mean()
    print mse
    return 10*np.log10((255.0**2)/mse)

def pitch_to_img(pitchs):
    pitch_num=pitchs.shape[1]
    size=np.sqrt(pitch_num).astype(np.int)
    pitch_size=np.sqrt(pitchs.shape[0]).astype(np.int)
    img=np.zeros((pitch_size*size, pitch_size*size))
    for i in range(pitch_num):
        r=(i/size)*8
        c=(i%size)*8
        img[r:r+8, c:c+8]=pitchs[:, i].reshape((8, 8))
    return img

def img_to_pitch(img):
    pitchs=np.zeros((8*8, 32*32))
    blocks_r=img.shape[0]/8
    blocks_c=img.shape[1]/8
    pitch_num=blocks_r*blocks_c
    for i in range(pitchs.shape[1]):
        #按先行后列，将图片分解成32*32个8*8的小块并装换为列向量
        r=(i/blocks_r)*8
        c=(i%blocks_c)*8
        pitch=img[r:r+8, c:c+8].flat
        pitchs[:, i]=pitch
    return pitchs

if __name__ == '__main__':

    #img = cv2.imread("/Users/linweichen/Desktop/lena_noise.jpg",0).astype(np.float)
    ori = cv2.imread("/Users/linweichen/Desktop/lena.jpg",0).astype(np.float)
    img=pixel_miss(ori)
    #img=Gauss_noise(ori)

    cv2.namedWindow("Original")
    cv2.imshow("Original",ori.astype(np.uint8))
    print psnr(ori,img)

    cv2.namedWindow("Noise")
    cv2.imshow("Noise",img.astype(np.uint8))

    ksvd = KSVD(max_iter=5)
    dictionary, denoise = ksvd.fit(img)
    #dictionary,denoise = ksvd.fit(ori)

    #cv2.namedWindow("Img_pitchs")
    #cv2.imshow("Img_pitchs", img_pitchs)
    #print dictionary,sparsecode

    #利用训练好的字典稀疏表示含噪图像
    img_pitchs=img_to_pitch(img)
    #print dictionary.shape, img_pitchs.shape
    cv2.namedWindow("Img_pitchs")
    cv2.imshow("Img_pitchs", img_pitchs.astype(np.uint8))

    #按块表示字典
    cv2.namedWindow("Dictionary")
    dictionary=dictionary-np.amin(dictionary)
    dictionary=dictionary/np.amax(dictionary)
    print dictionary
    cv2.imshow("Dictionary", pitch_to_img(dictionary))#.astype(np.uint8))


    cv2.namedWindow("Dictionary_denoise")
    x=linear_model.orthogonal_mp(dictionary, img_pitchs, n_nonzero_coefs=None)
    dictionary_denoise=pitch_to_img(dictionary.dot(x))
    cv2.imshow("Dictionary_denoise",((img+10*dictionary_denoise)/(11*np.ones(img.shape))).astype(np.uint8))
    print psnr(ori, dictionary_denoise)

    cv2.namedWindow("K-SVD")
    cv2.imshow("K-SVD",denoise.astype(np.uint8))
    print psnr(ori,denoise)
    #cv2.imshow("K-SVD",dictionary[:, :2048].dot(sparsecode[:2048, :]).astype(np.uint8))
    cv2.waitKey(0)

