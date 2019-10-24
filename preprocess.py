import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def extractUS(img):
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    nrow,ncol = gray.shape
    ret, bw = cv2.threshold(gray, 0.0, 1.0, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(10,10))
    bw = cv2.morphologyEx(bw,cv2.MORPH_OPEN,kernel)
    pcol = np.sum(bw,0)/ncol
    prow = np.sum(bw,1)/nrow
    row_min,row_max = findcorner(prow)
    col_min,col_max = findcorner(pcol)
    US = img[row_min:row_max,col_min:col_max]
    return US

def findcorner(prow):
    nrow = len(prow)
    n_2 = int(nrow/2)
    prow_diff = [prow[i+1]-prow[i] for i in range(nrow-1)]
    half0 = prow_diff[:n_2]
    pmax = max(half0)
    if pmax < 0.05:
        row_min = 0
    else:
        peaks_idx = [i for i in range(n_2) if half0[i] > 0.3*pmax]
        row_min = peaks_idx[-1]
        if n_2-row_min < 100 and len(peaks_idx)>=2:
            row_min = peaks_idx[-2]
    half1 = prow_diff[n_2:]
    pmin = min(half1)
    if pmin > -0.05:
        row_max = nrow-1
    else:
        valleys_idx = [i for i in range(len(half1)) if half1[i] < 0.2*pmin]
        row_max = valleys_idx[0]
        if row_max < 100 and len(valleys_idx)>=2:
            row_max = valleys_idx[1]
        tmps = [i for i in range(len(half1)) if half1[i] > 0.2*pmax]
        if tmps:
            tmp = tmps[0]
            if tmp < row_max:
                tmp_idx = [i for i in range(tmp) if half1[i] < 0]
                if tmp_idx:
                    row_max = tmp_idx[-1]
        row_max += n_2
    return row_min, row_max

def repair(US):
    nrow,ncol,_ = US.shape
    mask = np.zeros((nrow,ncol),dtype=np.uint8)
    for ii in range(nrow):
        for jj in range(ncol):
            p = [int(a) for a in US[ii,jj]]
            d = abs(p[1]-p[2])+abs(p[1]-p[0])+abs(p[2]-p[0])
            if d > 40:
                mask[ii,jj] = 1
    template = []
    thresholds = []
    ws = 13
    hs = np.int((ws-1)/2)
    cross = np.zeros((ws,ws),dtype=np.uint8)
    cross[hs-1:hs+2,:] = 1
    cross[:,hs-1:hs+2] = 1
    cross[hs-1:hs+2,hs-1:hs+2] = 0
    template.append(cross)
    thresholds.append(0.75)
    ws = 13
    hs = np.int((ws-1)/2)
    rot_cross = np.zeros((ws,ws),dtype=np.uint8)
    for i in range(ws):
        rot_cross[i,max(0,i-1):min(i+2,ws)] = 1
        rot_cross[i,max(0,ws-2-i):min(ws+1-i,ws)] = 1
        rot_cross[hs-1:hs+2,hs-1:hs+2] = 0
    template.append(rot_cross)
    thresholds.append(0.7)
    ws = 11
    hs = np.int((ws-1)/2)
    cen_cross = np.zeros((ws,ws),dtype=np.uint8)
    cen_cross[hs-1:hs+1,:] = 1
    cen_cross[:,hs-1:hs+1] = 1
    template.append(cen_cross)
    thresholds.append(0.8)
    cen_rot_cross = np.zeros((ws,ws),dtype=np.uint8)
    for i in range(ws):
        cen_rot_cross[i,max(0,i-1):min(i+1,ws)] = 1
        cen_rot_cross[i,max(0,ws-1-i):min(ws+1-i,ws)] = 1
    template.append(cen_rot_cross)
    thresholds.append(0.75)
    ws = 51
    hs = np.int((ws-1)/2)
    hline = np.zeros((ws,ws),dtype=np.uint8)
    hline[:,hs-1:hs] = 1
    template.append(hline)
    thresholds.append(0.2)
    gray = cv2.cvtColor(US,cv2.COLOR_RGB2GRAY)
    cc = 0
    for temp in template:
        pad = np.int(np.floor(temp.shape[0]/2))
        gray_padded = cv2.copyMakeBorder(gray,pad,pad,pad,pad,cv2.BORDER_CONSTANT,0)
        corr = cv2.matchTemplate(gray_padded,temp,cv2.TM_CCOEFF_NORMED)
        # print("max correlation: {}".format(np.max(corr)))
        thre = thresholds[cc]
        cc += 1
        ret, loc = cv2.threshold(corr, thre, 1.0, cv2.THRESH_BINARY)
        if np.sum(loc)>0:
            maskt = conv2(loc,temp)
            mask += maskt
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    mask = cv2.dilate(mask,kernel)
    pure_US = cv2.inpaint(US,mask,3,cv2.INPAINT_TELEA)
    return pure_US

def conv2(img,filt):
    conv = np.zeros(img.shape,dtype=np.uint8)
    pad = np.int(np.floor(filt.shape[0]/2))
    img = cv2.copyMakeBorder(img,pad,pad,pad,pad,cv2.BORDER_CONSTANT,0)
    mm,nn = img.shape
    for ii in range(pad,mm-pad):
        for jj in range(pad,nn-pad):
            mat = img[ii-pad:ii+pad+1,jj-pad:jj+pad+1]*filt
            conv[ii-pad,jj-pad] = np.sum(mat,dtype=np.uint8)
    return conv

def preprocess():
    file_dir = "../WZL病灶"
    save_dir = "../lesions"

    son_file = os.listdir(file_dir)
    for son_name in son_file:
        son_dir = os.path.join(file_dir,son_name,"Images")
        file_name = os.listdir(son_dir)
        for img_name in file_name:
            img_path = os.path.join(son_dir,img_name)
            image = cv2.imdecode(np.fromfile(img_path,dtype=np.uint8),-1)
            if (not image.shape == (1753,1240,3) and not image.shape==(950,1240,3)):
                print(img_path)
                US = extractUS(image)
                pure_US = repair(US)
                US_path = os.path.join(save_dir,img_name)
                # if not os.path.exists(US_path):
                cv2.imencode(".jpg",pure_US)[1].tofile(US_path)

if __name__ == "__main__":
    preprocess()
