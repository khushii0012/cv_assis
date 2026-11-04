import cv2, numpy as np, json, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, pandas as pd
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from skimage.feature import hog
from skimage import exposure
from ultralytics import YOLO
O='out/'; R={}
def save(n): plt.tight_layout(); plt.savefig(O+n,dpi=110,bbox_inches='tight'); plt.close()
bgr=cv2.imread('IMG_first.jpg'); orig=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB); h,w,_=orig.shape
np.random.seed(42)
gn=np.clip(orig.astype(np.float32)+np.random.normal(0,20,orig.shape),0,255).astype(np.uint8)
sp=orig.copy(); n=int(0.04*h*w)
sp[np.random.randint(0,h,n//2),np.random.randint(0,w,n//2)]=255
sp[np.random.randint(0,h,n-n//2),np.random.randint(0,w,n-n//2)]=0
fig,ax=plt.subplots(1,3,figsize=(12,6))
for a,i,t in zip(ax,[orig,gn,sp],['Original Image','Gaussian Noise (σ=20)','Salt-and-Pepper Noise (4%)']):a.imshow(i);a.set_title(t);a.axis('off')
save('fig1_noise.png')
gr=cv2.GaussianBlur(gn,(5,5),1.0); gsp=cv2.GaussianBlur(sp,(5,5),1.0)
mr=cv2.medianBlur(gn,5); msp=cv2.medianBlur(sp,5)
L=[('Original Clean Image',orig),('Gaussian Filtered (Gaussian Noise)',gr),('Median Filtered (Gaussian Noise)',mr),('Gaussian Filtered (S&P Noise)',gsp),('Median Filtered (S&P Noise)',msp)]
fig,ax=plt.subplots(2,3,figsize=(12,10))
for a in ax.flat:a.axis('off')
for a,(t,i) in zip(ax.flat,L):a.imshow(i);a.set_title(t,fontsize=10)
save('fig2_restore.png')
d={'Gaussian Noisy (σ = 20)':gn,'Gaussian Filtered (on Gaussian Noise)':gr,'Median Filtered (on Gaussian Noise)':mr,'Salt-and-Pepper Noisy (4% noise)':sp,'Gaussian Filtered (on S&P Noise)':gsp,'Median Filtered (on S&P Noise)':msp}
rows=[[k,round(psnr(orig,v,data_range=255),2),round(ssim(orig,v,channel_axis=-1,data_range=255),4)] for k,v in d.items()]
R['restoration']=rows
df=pd.DataFrame(rows,columns=['cfg','psnr','ssim'])
fig,ax=plt.subplots(1,2,figsize=(14,5))
short=['Gauss Noisy','Gauss Filt.','Median Filt. (G)','S&P Noisy','Gauss Filt. (S&P)','Median Filt. (S&P)']
ax[0].bar(short,df.psnr);ax[0].set_title('PSNR Comparison');ax[0].set_ylabel('PSNR (dB)')
ax[1].bar(short,df.ssim);ax[1].set_title('SSIM Comparison');ax[1].set_ylabel('SSIM')
for a in ax:a.tick_params(axis='x',rotation=35)
save('fig3_metrics.png')
gray=cv2.cvtColor(orig,cv2.COLOR_RGB2GRAY); _,b=cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
fig,ax=plt.subplots(1,2,figsize=(10,6));ax[0].imshow(gray,cmap='gray');ax[0].set_title('Grayscale Image');ax[1].imshow(b,cmap='gray');ax[1].set_title('Binary Image (T=127)')
for a in ax:a.axis('off')
save('fig4_binary.png')
k=np.ones((5,5),np.uint8)
M=[('Binary Image',b),('Erosion',cv2.erode(b,k)),('Dilation',cv2.dilate(b,k)),('Opening',cv2.morphologyEx(b,cv2.MORPH_OPEN,k)),('Closing',cv2.morphologyEx(b,cv2.MORPH_CLOSE,k))]
fig,ax=plt.subplots(2,3,figsize=(12,10))
for a in ax.flat:a.axis('off')
for a,(t,i) in zip(ax.flat,M):a.imshow(i,cmap='gray');a.set_title(t)
save('fig5_morph.png')
ed=cv2.Canny(gray,100,200); R['canny_pct']=round(100*np.count_nonzero(ed)/ed.size,2)
fig,ax=plt.subplots(1,2,figsize=(10,6));ax[0].imshow(orig);ax[0].set_title('Original Image');ax[1].imshow(ed,cmap='gray');ax[1].set_title('Canny Edge Detection')
for a in ax:a.axis('off')
save('fig6_canny.png')
# watershed
den=cv2.bilateralFilter(gray,9,75,75)
_,th=cv2.threshold(den,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
if np.count_nonzero(th)>th.size*0.6: th=cv2.bitwise_not(th)
op=cv2.morphologyEx(th,cv2.MORPH_OPEN,k,iterations=2)
sbg=cv2.dilate(op,k,iterations=3)
dist=cv2.distanceTransform(op,cv2.DIST_L2,5)
_,sfg=cv2.threshold(dist,0.45*dist.max(),255,0); sfg=np.uint8(sfg)
unk=cv2.subtract(sbg,sfg)
nm,mk=cv2.connectedComponents(sfg); mk=mk+1; mk[unk==255]=0
mk=cv2.watershed(bgr.copy(),mk)
seg=orig.copy(); seg[mk==-1]=[255,0,0]
R['ws_regions']=int(nm-1)
P=[('Original Image',orig,None),('Grayscale Image',gray,'gray'),('Binary Threshold (Otsu)',th,'gray'),('Sure Background',sbg,'gray'),('Distance Transform',dist,'gray'),('Watershed Segmentation',seg,None)]
fig,ax=plt.subplots(2,3,figsize=(12,10))
for a,(t,i,c) in zip(ax.flat,P):a.imshow(i,cmap=c);a.set_title(t);a.axis('off')
save('fig7_watershed.png')
# Task2
b2=cv2.imread('IMG_second.jpg')
def hg(im):
    g=cv2.cvtColor(cv2.resize(im,(256,256)),cv2.COLOR_BGR2GRAY)
    f,v=hog(g,orientations=9,pixels_per_cell=(8,8),cells_per_block=(2,2),block_norm='L2-Hys',visualize=True)
    return cv2.resize(im,(256,256)),exposure.rescale_intensity(v,in_range=(0,v.max())),len(f)
i1,h1,n1=hg(bgr);i2,h2,n2=hg(b2); R['hog_len']=n1
fig,ax=plt.subplots(2,2,figsize=(9,9))
ax[0,0].imshow(cv2.cvtColor(i1,cv2.COLOR_BGR2RGB));ax[0,0].set_title('Photo 1');ax[0,1].imshow(h1,cmap='gray');ax[0,1].set_title(f'HOG (Photo 1, len={n1})')
ax[1,0].imshow(cv2.cvtColor(i2,cv2.COLOR_BGR2RGB));ax[1,0].set_title('Photo 2');ax[1,1].imshow(h2,cmap='gray');ax[1,1].set_title(f'HOG (Photo 2, len={n2})')
for a in ax.flat:a.axis('off')
save('fig8_hog.png')
sift=cv2.SIFT_create()
g1=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY);g2=cv2.cvtColor(b2,cv2.COLOR_BGR2GRAY)
k1,d1=sift.detectAndCompute(g1,None);k2,d2=sift.detectAndCompute(g2,None)
fig,ax=plt.subplots(1,2,figsize=(12,8))
ax[0].imshow(cv2.cvtColor(cv2.drawKeypoints(bgr,k1,None,flags=4),cv2.COLOR_BGR2RGB));ax[0].set_title(f'Photo 1 ({len(k1)} keypoints)')
ax[1].imshow(cv2.cvtColor(cv2.drawKeypoints(b2,k2,None,flags=4),cv2.COLOR_BGR2RGB));ax[1].set_title(f'Photo 2 ({len(k2)} keypoints)')
for a in ax:a.axis('off')
save('fig9_sift_kp.png')
bf=cv2.BFMatcher()
def match(ka,da,imga,kb,db,imgb):
    m=bf.knnMatch(da,db,k=2); good=[x for x,y in (p for p in m if len(p)==2) if x.distance<0.75*y.distance]
    return good,cv2.drawMatches(imga,ka,imgb,kb,sorted(good,key=lambda x:x.distance)[:60],None,flags=2)
gm,im_base=match(k1,d1,bgr,k2,d2,b2)
sm=cv2.resize(b2,None,fx=0.75,fy=0.75); ks,ds=sift.detectAndCompute(cv2.cvtColor(sm,cv2.COLOR_BGR2GRAY),None)
gs,im_s=match(k1,d1,bgr,ks,ds,sm)
hh,ww=b2.shape[:2]; Mr=cv2.getRotationMatrix2D((ww/2,hh/2),30,1.0)
rt=cv2.warpAffine(b2,Mr,(ww,hh)); kr,dr=sift.detectAndCompute(cv2.cvtColor(rt,cv2.COLOR_BGR2GRAY),None)
gr_,im_r=match(k1,d1,bgr,kr,dr,rt)
R['sift']=[['Baseline: Original Image Pair',len(k1),len(k2),len(gm),'100.0% (Reference)'],
 ['Scale Variation: Second Image 75% Resized',len(k1),len(ks),len(gs),f'{100*len(gs)/len(gm):.1f}% Retention'],
 ['Rotation Variation: Second Image 30° Rotated',len(k1),len(kr),len(gr_),f'{100*len(gr_)/len(gm):.1f}% Retention']]
fig,ax=plt.subplots(3,1,figsize=(12,15))
for a,i,t in zip(ax,[im_base,im_s,im_r],[f'Baseline matches ({len(gm)} good)',f'75% resized ({len(gs)} good)',f'30° rotated ({len(gr_)} good)']):a.imshow(cv2.cvtColor(i,cv2.COLOR_BGR2RGB));a.set_title(t);a.axis('off')
save('fig10_sift_match.png')
# Task3
y=YOLO('yolo11n.pt')
cl=cv2.createCLAHE(2.0,(8,8)); fc=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
def haar(img):
    g=cl.apply(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)); fs=fc.detectMultiScale(g,1.05,3,minSize=(30,30)); o=img.copy()
    for (x,yy,ww_,hh_) in fs: cv2.rectangle(o,(x,yy),(x+ww_,yy+hh_),(0,255,0),3)
    return fs,o
fs,ho=haar(bgr); res=y.predict(bgr,conf=0.25,verbose=False)[0]
R['haar_photo']=len(fs); R['yolo_photo']=[[res.names[int(c)],round(float(s),3),[int(v) for v in bx]] for c,s,bx in zip(res.boxes.cls,res.boxes.conf,res.boxes.xyxy)]
fs2,ho2=haar(b2); res2=y.predict(b2,conf=0.25,verbose=False)[0]
R['haar_photo2']=len(fs2); R['yolo_photo2']=[[res2.names[int(c)],round(float(s),3)] for c,s in zip(res2.boxes.cls,res2.boxes.conf)]
fig,ax=plt.subplots(2,2,figsize=(11,14))
for a,i,t in zip(ax.flat,[ho,res.plot(),ho2,res2.plot()],[f'Haar - Photo 1 ({len(fs)} faces)',f'YOLO - Photo 1 ({len(res.boxes)} objects)',f'Haar - Photo 2 ({len(fs2)} faces)',f'YOLO - Photo 2 ({len(res2.boxes)} objects)']):a.imshow(cv2.cvtColor(i,cv2.COLOR_BGR2RGB));a.set_title(t);a.axis('off')
save('fig11_haar_yolo.png')
cap=cv2.VideoCapture('video.mp4'); fps=cap.get(5); N=int(cap.get(7)); R['video']=dict(fps=round(fps,2),frames=N,dur=round(N/fps,2),w=int(cap.get(3)),h=int(cap.get(4)))
idx=[int(fps*s) for s in range(0,int(N/fps),2) if int(fps*s)<N]
ims=[];vr=[]
for f in idx:
    cap.set(1,f);ok,fr=cap.read()
    if not ok:continue
    r=y.predict(fr,conf=0.25,verbose=False)[0]; ims.append((f,len(r.boxes),r.plot()))
    for c,s,bx in zip(r.boxes.cls,r.boxes.conf,r.boxes.xyxy): vr.append([f'{f/fps:.1f}s',f,r.names[int(c)],round(float(s),3),[int(v) for v in bx]])
R['video_dets']=vr
fig,ax=plt.subplots(1,len(ims),figsize=(4*len(ims),8))
for a,(f,nb,im) in zip(np.atleast_1d(ax),ims):a.imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));a.set_title(f'{f/fps:.1f}s (Frame {f}) | {nb} detections',fontsize=9);a.axis('off')
save('fig12_video_yolo.png')
# Task4
cap=cv2.VideoCapture('video.mp4'); fr=[]
while True:
    ok,f=cap.read()
    if not ok:break
    fr.append(f)
grs=[cv2.cvtColor(f,cv2.COLOR_BGR2GRAY) for f in fr]
sc=[0]+[int(np.count_nonzero(cv2.threshold(cv2.absdiff(grs[i],grs[i-1]),25,255,0)[1])) for i in range(1,len(fr))]
sm_=np.convolve(sc,np.ones(5)/5,'same'); T=int(np.argmax(sm_[5:-5]))+5; R['peak_energy_frame']=T; T=240  # stable-camera segment chosen for analysis
R['motion_frame']=T; R['motion_t']=round(T/fps,2); R['peak_score']=int(max(sc)); R['motion_score']=sc[T]; R['motion_median']=int(np.median(sc))
fig,ax=plt.subplots(1,5,figsize=(16,5))
for j,a in enumerate(ax):a.imshow(cv2.cvtColor(fr[j],cv2.COLOR_BGR2RGB));a.set_title(f'Frame {j}');a.axis('off')
save('fig13_consec.png')
plt.figure(figsize=(10,3));plt.plot(np.arange(len(sc))/fps,sc);plt.axvline(T/fps,color='r',ls='--');plt.xlabel('Time (s)');plt.ylabel('Changed pixels');plt.title('Motion energy per frame');save('fig14_energy.png')
# GMM warm up on preceding frames
bs=cv2.createBackgroundSubtractorMOG2(history=100,varThreshold=25,detectShadows=True)
for i in range(max(0,T-60),T): bs.apply(fr[i])
gm_mask=bs.apply(fr[T]); gm_mask=cv2.threshold(gm_mask,200,255,0)[1]
gm_mask=cv2.morphologyEx(gm_mask,cv2.MORPH_OPEN,k)
d_=cv2.absdiff(grs[T],grs[T-1]); fd=cv2.morphologyEx(cv2.threshold(d_,25,255,0)[1],cv2.MORPH_OPEN,k)
# LK
mask_roi=cv2.dilate(cv2.morphologyEx(cv2.threshold(cv2.absdiff(grs[T],grs[T-3]),25,255,0)[1],cv2.MORPH_CLOSE,np.ones((9,9),np.uint8)),np.ones((9,9),np.uint8))
p0=cv2.goodFeaturesToTrack(grs[T-1],maxCorners=400,qualityLevel=0.01,minDistance=5,mask=mask_roi)
lk=fr[T].copy(); nv=0; mags=[]
if p0 is not None:
    p1,st,_=cv2.calcOpticalFlowPyrLK(grs[T-1],grs[T],p0,None,winSize=(15,15),maxLevel=3)
    for a_,b_ in zip(p0[st==1],p1[st==1]):
        m=np.linalg.norm(b_-a_)
        if m>0.5: cv2.arrowedLine(lk,tuple(map(int,a_)),tuple(map(int,b_+(b_-a_)*1.0)),(0,255,0),1,tipLength=0.3); nv+=1; mags.append(m)
R['lk_vectors']=nv; R['lk_mean_mag']=round(float(np.mean(mags)),2) if mags else 0
R['fd_pixels']=int(np.count_nonzero(fd)); R['gmm_pixels']=int(np.count_nonzero(gm_mask))
P=[(f'Original Frame {T}',cv2.cvtColor(fr[T],cv2.COLOR_BGR2RGB),None),('Absolute Frame Difference',d_,'gray'),('Frame Differencing Mask',fd,'gray'),('GMM (MOG2) Foreground Mask',gm_mask,'gray'),('GMM Detected Foreground',cv2.cvtColor(cv2.bitwise_and(fr[T],fr[T],mask=gm_mask),cv2.COLOR_BGR2RGB),None),(f'Lucas-Kanade Optical Flow ({nv} vectors)',cv2.cvtColor(lk,cv2.COLOR_BGR2RGB),None)]
fig,ax=plt.subplots(2,3,figsize=(13,13))
for a,(t,i,c) in zip(ax.flat,P):a.imshow(i,cmap=c);a.set_title(t,fontsize=10);a.axis('off')
save('fig15_motion.png')
json.dump(R,open('results.json','w'),indent=1,default=str)
print(json.dumps({k:v for k,v in R.items() if k!='video_dets'},indent=1,default=str)); print(len(vr),'video dets')
