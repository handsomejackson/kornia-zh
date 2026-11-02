import kornia as K, torch, cv2, numpy as np
def load(p):
    return K.io.load_image(p, K.io.ImageLoadType.RGB32)[None]  # 1x3xHxW float 0-1
def save(t,p):
    a=(t.clamp(0,1)[0].permute(1,2,0).detach().numpy()*255).astype('uint8')
    cv2.imwrite(p,cv2.cvtColor(a,cv2.COLOR_RGB2BGR) if a.shape[2]==3 else a)
def grid(ts,labels,p,cols=4):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif']=['Noto Sans CJK SC'];
    rows=(len(ts)+cols-1)//cols
    f,ax=plt.subplots(rows,cols,figsize=(3.2*cols,3.4*rows))
    for a in np.array(ax).ravel(): a.axis('off')
    for a,t,l in zip(np.array(ax).ravel(),ts,labels):
        x=t.clamp(0,1)[0].permute(1,2,0).detach().numpy()
        a.imshow(x.squeeze(),cmap='gray' if x.shape[2]==1 else None); a.set_title(l,fontsize=12)
    plt.tight_layout(); plt.savefig(p,dpi=110); plt.close()
