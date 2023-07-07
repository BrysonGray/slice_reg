from glob import glob
from os.path import split,join,splitext,exists
from os import makedirs, listdir
from tifffile import tifffile
import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.stats import norm
import torch

import logging
logger = logging.getLogger()
old_logger_level = logger.level
logger.setLevel(100)

# now we need to take the pca and view it
def pca(I,w=None,output_v=False):
    if w is None:
        w = np.ones((I.shape[0],I.shape[1]))
    
    w = w.reshape(-1)[...,None]
    ws = np.sum(w)
    Iv = np.reshape(I,(I.shape[0]*I.shape[1],-1))
    Ivbar = np.sum(Iv*w,0,keepdims=True)/ws
    Iv0 = Iv - Ivbar
    S = (Iv0.T@(Iv0*w))/ws # divide by number of pixels
    w_,v = np.linalg.eigh(S)
    s = np.sqrt(w_)
    Ip = (Iv0@v).reshape(I.shape)
    
    if output_v:
        return Ip[...,::-1],s[::-1],v[:,::-1]
    # put in decreasing order and return
    return Ip[...,::-1],s[::-1]


def pca_scale(Ip,s):    
    return norm.cdf(Ip/s[None,None])


def draw(I,w=None):
    Ip,sI,vI = pca(I,w=w,output_v=True)
    Is = pca_scale(Ip[...,:7],sI[:7])
    # let's project L onto the first PC
    if w is not None:
        muI = np.sum(I*w[...,None],axis=(0,1),keepdims=True)/np.sum(w)
    else:
        muI = np.mean(I,axis=(0,1),keepdims=True)
    c = np.sum(vI[:,0]*(I - muI),-1)

    fav,ax = plt.subplots()
    ax.imshow(I[...,:3])
    ax.set_title('average')
    
    f1,ax = plt.subplots()
    ax.imshow(c[...,None]*vI[:3,0] + muI[...,:3])
    ax.set_title('PC 1 proj')

    fs1,ax = plt.subplots()
    ax.imshow(Is[...,1:4])
    ax.set_title('PC components 2 to 4')

    fs2,ax = plt.subplots()
    ax.imshow(Is[...,4:7])
    ax.set_title('PC components 5 to 7')
    
    return fav, f1, fs1, fs2


def downsample(nI,xI,I,labels,n_scatter,n_init=0,n_final=0,max_scatter=2):
    '''
    This downsampling multiplies by all wavelets at once
    This may be too much memory
    Below I am separating red and green and blue channels
    '''
    # filter
    r=1
    x = np.arange(-r,r+1)
    X = np.stack(np.meshgrid(x,x,indexing='ij'))
    k = np.exp(-np.sum(X**2,0)/2.0)
    k /= np.sum(k)

    # now we multiply by a complex exponential
    l=3.0
    ntheta = 4
    thetas = np.arange(ntheta)/ntheta*np.pi
    X_ = X[0][...,None]*np.cos(thetas) + X[1][...,None]*np.sin(thetas)
    wave = np.exp(2.0*np.pi*1j/l*X_)
    wavelet = k[...,None] * wave

    # now the imaginary part is 0 mean, but the real part is not
    # we want to add a constant to the wave
    # k * (wave + c)
    # what should c be?
    # sum (k*wave) + sum(k*c) = sum(k*wave) + sum(k)*c = sum(k*wave) + c = 0
    # so c = - sum(k*wave)
    wavelet = k[...,None] * (wave - np.sum(wavelet,(0,1)))
    wavelet = np.concatenate((k[...,None],wavelet),-1)
    
    #print(f'Initial I shape {I.shape}')
    # initial downsampling (e.g. for debugging)
    for i in range(n_init):
        nd = np.array(I.shape)//2
        I = I[0:nd[0]*2:2]*0.5 + I[1:nd[0]*2:2]*0.5
        I = I[:,0:nd[1]*2:2]*0.5 + I[:,1:nd[1]*2:2]*0.5     
        # update labels here
        labels = [ l + str(0) for l in labels] 
        #print(f'Initial downsampled I, size is {I.shape}')
    for _ in range(n_scatter):
        # do the convolution
        I_ = I[1:-1,1:-1,None,:]*wavelet[1,1,:,None]
        I_ += I[:-2,1:-1,None,:]*wavelet[0,1,:,None]
        I_ += I[2:,1:-1,None,:]*wavelet[2,1,:,None]

        I_ += I[1:-1,:-2,None,:]*wavelet[1,0,:,None]
        I_ += I[:-2,:-2,None,:]*wavelet[0,0,:,None]
        I_ += I[2:,:-2,None,:]*wavelet[2,0,:,None]

        I_ += I[1:-1,2:,None,:]*wavelet[1,2,:,None]
        I_ += I[:-2,2:,None,:]*wavelet[0,2,:,None]
        I_ += I[2:,2:,None,:]*wavelet[2,2,:,None]

        # the None's are as above so they will reshape properly
        # take absolute value and reshape it
        I_ = np.abs(I_.reshape(I_.shape[0],I_.shape[1],-1))

        # downsample
        nd = np.array(I_.shape)//2
        I_ = I_[0:nd[0]*2:2]*0.5 + I_[1:nd[0]*2:2]*0.5
        I_ = I_[:,0:nd[1]*2:2]*0.5 + I_[:,1:nd[1]*2:2]*0.5

        # labels
        labels_ = [ l + str(i) for i in range(ntheta+1) for l in labels] 
        
        # reset references to working variables
        I = I_
        labels = labels_

        # filter out paths that are too long (have too many nonlinearities)
        # note it would be better to just not calculate them, but this is a minor performance hit
        lengths = [len([t for t in l if t=='1' or t=='2' or t=='3' or t=='4' ]) for l in labels]
        inds = [l<=max_scatter for l in lengths]        
        I = I[...,inds]
        labels = [l for l,i in zip(labels,inds) if i]
        #print(f'Filtered I, shape is {I.shape}')
    # now final downsampling
    for i in range(n_final):
        nd = np.array(I.shape)//2
        I = I[0:nd[0]*2:2]*0.5 + I[1:nd[0]*2:2]*0.5
        I = I[:,0:nd[1]*2:2]*0.5 + I[:,1:nd[1]*2:2]*0.5     
        # update labels here
        labels = [ l + str(0) for l in labels] 
        #print(f'Final downsampled I, size is {I.shape}')
        
    # downsample coordinate space
    for i in range(n_init):
        if i==0:
            nd = nI//2
        else:
            nd = nd//2    
        xI = [x[0:n*2:2]*0.5 + x[1:n*2:2]*0.5 for n,x in zip(nd,xI)]
    for i in range(n_scatter):
        if n_init == 0 and i == 0:
            nd = (nI-2)//2
        else:
            nd = (nd-2)//2 
        xI = [x[1:-1] for x in xI]
        xI = [x[0:n*2:2]*0.5 + x[1:n*2:2]*0.5 for n,x in zip(nd,xI)]
    for i in range(n_final):
        if n_scatter==0 and n_init==0 and i==0:
            nd = nI//2
        else:
            nd = nd//2    
        xI = [x[0:n*2:2]*0.5 + x[1:n*2:2]*0.5 for n,x in zip(nd,xI)]
    
    return xI, I, labels


# we want to take a label and get a turn
# this will be a different sequence
# now there's either
# (l)eft
# (r)ight
# (n)inety
# or (p)arallel
def label_to_turn(label):
    label_turn = []
    first_direction = ''
    last_direction = ''
    for l in label:
        if l not in '1234':
            label_turn.append(l)
        else:
            if first_direction:
                # here we compare to last direction
                if last_direction == '1':
                    if l == '1':
                        label_turn.append('p')
                    elif l == '2':
                        label_turn.append('l')
                    elif l == '3':
                        label_turn.append('n')
                    elif l == '4':
                        label_turn.append('r')
                elif last_direction == '2':
                    if l == '1':
                        label_turn.append('r')
                    elif l == '2':
                        label_turn.append('p')
                    elif l == '3':
                        label_turn.append('l')
                    elif l == '4':
                        label_turn.append('n')
                elif last_direction == '3':
                    if l == '1':
                        label_turn.append('n')
                    elif l == '2':
                        label_turn.append('r')
                    elif l == '3':
                        label_turn.append('p')
                    elif l == '4':
                        label_turn.append('l')
                elif last_direction == '4':
                    if l == '1':
                        label_turn.append('l')
                    elif l == '2':
                        label_turn.append('n')
                    elif l == '3':
                        label_turn.append('r')
                    elif l == '4':
                        label_turn.append('p')

                last_direction = l

            else:
                first_direction = l
                last_direction = l
                label_turn.append(l)
    return ''.join(label_turn)
                

# now we need to go the other way
def turn_to_label(turn):

    last_direction = ''
    labels_ = []
    for t in turn:
        if t not in 'lrpn':
            labels_.append(t)
            if t in '1234':
                last_direction = t
        else:
            # compare to last direction
            if last_direction == '1':
                if t == 'p':
                    labels_.append('1')
                elif t == 'l':
                    labels_.append('2')
                elif t == 'n':
                    labels_.append('3')
                elif t == 'r':
                    labels_.append('4')
            elif last_direction == '2':
                if t == 'p':
                    labels_.append('2')
                elif t == 'l':
                    labels_.append('3')
                elif t == 'n':
                    labels_.append('4')
                elif t == 'r':
                    labels_.append('1')
            elif last_direction == '3':
                if t == 'p':
                    labels_.append('3')
                elif t == 'l':
                    labels_.append('4')
                elif t == 'n':
                    labels_.append('1')
                elif t == 'r':
                    labels_.append('2')
            elif last_direction == '4':
                if t == 'p':
                    labels_.append('4')
                elif t == 'l':
                    labels_.append('1')
                elif t == 'n':
                    labels_.append('2')
                elif t == 'r':
                    labels_.append('3')

            last_direction = labels_[-1]
    return ''.join(labels_)

            
def reflect(label):
    turn = label_to_turn(label)
    reflected_turn = []
    for t in turn:
        if t not in 'rl':
            reflected_turn.append(t)
        elif t == 'r':
            reflected_turn.append('l')
        elif t == 'l':
            reflected_turn.append('r')
    reflected_label = turn_to_label(reflected_turn)
    return ''.join(reflected_label)
            
            
# now we want to combine them by averaging, taking interference sums, over rotations
def rot(label):
    rotated_label = []
    for l in label:
        if l == '1':
            rotated_label.append('2')
        elif l == '2':
            rotated_label.append('3')
        elif l == '3':
            rotated_label.append('4')
        elif l == '4':
            rotated_label.append('1')
        else:
            rotated_label.append(l)
    return ''.join(rotated_label)


def rigid_alignment_block(xI, I, xJ, J, A=None, 
                    device='cuda:0', dtype=torch.float64, 
                    niter=4000, epL=1e-6, epT=1e-3, title='', ndraw=250):
    
    '''
    Rigid alignment with block matching.
    
    Every 100 iterations contrast coefficients are updated.
    
    Every 200 iterations, block size is reduced
    
    TODO
    ----
    Add voxel size.
    
    '''
    
    if A is None:
        A = torch.eye(3,requires_grad=True,device=device,dtype=dtype)
    else:
        if type(A) is torch.Tensor:
            A = A.clone().detach().requires_grad_(True)
        else:
            A = torch.tensor(A,device=device,dtype=dtype,requires_grad=True)
    if type(xI[0]) is torch.Tensor:
        xI = [x.detach().clone().requires_grad_(True) for x in xI]
    else:    
        xI = [torch.tensor(x,device=device,dtype=dtype) for x in xI]
    if type(xJ[0]) is torch.Tensor:
        xJ = [x.detach().clone().requires_grad_(True) for x in xJ]
    else:
        xJ = [torch.tensor(x,device=device,dtype=dtype) for x in xJ]
    if type(I) is torch.Tensor:
        I = I.detach().clone().requires_grad_(True)
    else:
        I = torch.tensor(I,device=device,dtype=dtype)
    if type(J) is torch.Tensor:
        J = J.detach().clone().requires_grad_(True)
    else:
        J = torch.tensor(J,device=device,dtype=dtype)
    XJ = torch.stack(torch.meshgrid(xJ,indexing='ij'),-1)
    
    Esave = []
    fig,ax = plt.subplots(2,3)
    ax = ax.ravel()
    Lsave = []
    Tsave = []
    reduce_factor = 0.5
    for it in range(niter):
        Ai = torch.linalg.inv(A)
        Xs = (Ai[:2,:2]@XJ[...,None])[...,0] + Ai[:2,-1]
        # scale 0 to 1
        Xs = Xs - torch.stack([xI[0][0],xI[1][0]])
        Xs = Xs / torch.stack([xI[0][-1] - xI[0][0],xI[1][-1] - xI[1][0]])
        # scale -1 to 1
        Xs = 2*Xs - 1
        # convert to xy
        Xs = Xs.flip(-1)
        # sample
        AI = torch.nn.functional.grid_sample(I[None],Xs[None],align_corners=True,padding_mode='border')[0]
        if False:
            # predict contrast
            B = torch.cat(   [torch.ones((1,J.shape[1]*J.shape[2]),device=device,dtype=dtype),AI.reshape(AI.shape[0],-1)] ,0)
            with torch.no_grad():    
                BB = B@B.T
                BJ = B@J.reshape(J.shape[0],-1).T
                coeffs = torch.linalg.solve(BB,BJ)
            fAI = (coeffs.T@B).reshape(J.shape)
        if True:
            # predict local contrast
            if it == 0:
                # start with the whole image
                M = torch.tensor(J.shape[1:],device=J.device)
            elif it == niter//4:
                # M = torch.tensor([64,64],device=J.device)
                M = torch.tensor([J.shape[1]//4,J.shape[1]//4],device=J.device)
            elif it == niter//2:
                M = torch.tensor([J.shape[1]//8,J.shape[1]//8],device=J.device)
            # elif it == 3*niter//4:
            #     M = torch.tensor([J.shape[1]//16,J.shape[1]//16],device=J.device)

            Jshape = torch.as_tensor(J.shape[1:],device=device)
            topad = Jshape%M
            topad = (M-topad)%M
            W = torch.ones_like(J[0][None])

            Jpad = torch.nn.functional.pad(J,(0,topad[1].item(),0,topad[0].item()))        
            AIpad = torch.nn.functional.pad(AI,(0,topad[1].item(),0,topad[0].item()))
            Wpad = torch.nn.functional.pad(W,(0,topad[1].item(),0,topad[0].item()))
            # now we will reshape it so that each block is a leading dimension
            #
            Jpad_ = Jpad.reshape( (Jpad.shape[0],Jpad.shape[1]//M[0].item(),M[0].item(),Jpad.shape[2]//M[1].item(),M[1].item()))
            Jpad__ = Jpad_.permute(1,3,2,4,0)
            Jpadv = Jpad__.reshape(Jpad__.shape[0],Jpad__.shape[1],(M[0]*M[1]).item(),Jpad__.shape[-1])

            AIpad_ = AIpad.reshape( (AIpad.shape[0],AIpad.shape[1]//M[0].item(),M[0].item(),AIpad.shape[2]//M[1].item(),M[1].item()))
            AIpad__ = AIpad_.permute(1,3,2,4,0)
            AIpadv = AIpad__.reshape(AIpad__.shape[0],AIpad__.shape[1],(M[0]*M[1]).item(),AIpad__.shape[-1],)

            Wpad_ = Wpad.reshape( (Wpad.shape[0],Wpad.shape[1]//M[0].item(),M[0].item(),Wpad.shape[2]//M[1].item(),M[1].item()))
            Wpad__ = Wpad_.permute(1,3,2,4,0)
            Wpadv = Wpad__.reshape(Wpad__.shape[0],Wpad__.shape[1],(M[0]*M[1]).item(),Wpad__.shape[-1],)

            # now basis function
            B = torch.cat((torch.ones_like(AIpadv[...,0])[...,None],AIpadv),-1)

            # coeffs
            ncoeffs = 10 # update every ncoeffs (was 100)
            if not it%ncoeffs:
                with torch.no_grad():            
                    BB = B.transpose(-1,-2)@(B*Wpadv)
                    BJ = B.transpose(-1,-2)@(Jpadv*Wpadv)
                    small = 1e-3
                    coeffs = torch.linalg.solve(BB + torch.eye(BB.shape[-1],device=BB.device)*small,BJ)
            fAIpadv = (B@coeffs).reshape(Jpadv.shape[0],Jpadv.shape[1],M[0].item(),M[1].item(),Jpadv.shape[-1])

            # reverse this permutation (1,3,2,4,0)
            fAIpad_ = fAIpadv.permute(4,0,2,1,3)        
            fAIpad = fAIpad_.reshape(Jpad.shape)

            fAI = fAIpad[:,:J.shape[1],:J.shape[2]]
        
        # note I changed sum to mean
        E = torch.mean((fAI - J)**2)/2.0
        Esave.append(E.item())
        E.backward()

        Tsave.append(A[:2,-1].detach().clone().squeeze().cpu().numpy())
        Lsave.append(A[:2,:2].detach().clone().squeeze().reshape(-1).cpu().numpy())

        if it > 10:
            checksign0 = np.sign(Tsave[-1] - Tsave[-2])
            checksign1 = np.sign(Tsave[-2] - Tsave[-3])
            checksign2 = np.sign(Tsave[-3] - Tsave[-4])
            reducedA = False
            if np.any((checksign0 != checksign1)*(checksign1 != checksign2)):
                epT *= reduce_factor
                print(f'Iteration {it}, translation oscilating, reducing epT to {epT}')
                reducedA = True
            checksign0 = np.sign(Lsave[-1] - Lsave[-2])
            checksign1 = np.sign(Lsave[-2] - Lsave[-3])
            checksign2 = np.sign(Lsave[-3] - Lsave[-4])
            if np.any( (checksign0 != checksign1)*(checksign1 != checksign2) ) and not reducedA:
                epL *= reduce_factor
                print(f'Iteration {it}, linear oscilating, reducing epL to {epL}')
        # update
        with torch.no_grad():
            A[:2,:2] -= A.grad[:2,:2]*epL
            u,s,vh = torch.linalg.svd(A[:2,:2])
#             A[:2,:2] = u@vh
            A[:2,:2] = u@vh * torch.exp(torch.mean(torch.log(s))) # try updating scale
            A[:2,-1] -= A.grad[:2,-1]*epT
            A.grad.zero_()

        # draw
        if not it%ndraw or it == niter-1:
            with torch.no_grad():
                ax[0].cla()
                Ishow = AI.clone().detach().permute(1,2,0).cpu().numpy()
                if Ishow.shape[-1] > 3:
                    Ip,Is = pca(Ishow)
                    Is = pca_scale(Ip,Is)[...,:3]
                elif Ishow.shape[-1] == 3:
                    Is = Ishow
                elif Ishow.shape[-1] == 2:
                    Is = torch.stack((Ishow[...,0],Ishow[...,1],Ishow[...,0]),-1)
                elif Ishow.shape[-1] == 1:
                    Is = torch.stack((Ishow[...,0],Ishow[...,0],Ishow[...,0]),-1)
                ax[0].imshow(Is)
                
                if it == 0:
                    Jshow = J.clone().detach().permute(1,2,0).cpu()
                    vmin = torch.quantile(Jshow,0.5)
                    vmax = torch.quantile(Jshow,0.999)
                    ax[2].cla()
                    if Jshow.shape[-1] >= 3:
                        ax[2].imshow(Jshow[...,:3], vmax=vmax, vmin=vmin)
                    elif Jshow.shape[-1] == 2:
                        Jshow = torch.stack((Jshow[...,0],Jshow[...,1],Jshow[...,0]),-1)
                        ax[2].imshow(Jshow, vmax=vmax, vmin=vmin)
                    elif Jshow.shape[-1] == 1:
                        Jshow = torch.stack((Jshow[...,0],Jshow[...,0],Jshow[...,0]),-1)
                        ax[2].imshow(Jshow, vmax=vmax, vmin=vmin)

                ax[1].cla()
                fAIshow = fAI.clone().detach().permute(1,2,0).cpu()
                if fAIshow.shape[-1] >= 3: 
                    ax[1].imshow(fAIshow[...,:3], vmax=vmax, vmin=vmin)
                elif fAIshow.shape[-1] == 2:
                    fAIshow = torch.stack((fAIshow[...,0],fAIshow[...,1],fAIshow[...,0]),-1)
                    ax[1].imshow(fAIshow, vmax=vmax, vmin=vmin)
                elif fAIshow.shape[-1] == 1:
                    fAIshow = torch.stack((fAIshow[...,0],fAIshow[...,0],fAIshow[...,0]),-1)
                    ax[1].imshow(fAIshow, vmax=vmax, vmin=vmin)

                ax[4].cla()
                errshow = ((fAI-J).clone().detach()*0.5*2+0.5).permute(1,2,0).cpu()
                if errshow.shape[-1] >= 3:
                    ax[4].imshow(errshow[...,:3])
                elif errshow.shape[-1] == 2:
                    errshow = torch.stack((errshow[...,0], errshow[...,1], errshow[...,0]),-1)
                    ax[4].imshow(errshow)
                elif errshow.shape[-1] == 1:
                    errshow = torch.stack((errshow[...,0], errshow[...,0], errshow[...,0]),-1)
                    ax[4].imshow(errshow)
                ax[3].cla()
                ax[3].plot(Esave)
                fig.suptitle(title)
                fig.canvas.draw()

    return A.clone().detach().cpu(), AI, fAI, fig


def register_slices(source_dir, target_dir, out_dir, ids=None, **kwargs):
    """
    Register slices from source_dir to target_dir and save them in out_dir

    Parameters
    ----------
    source_dir : str
        Directory containing source images
    target_dir : str
        Directory containing target images
    out_dir : str
        Directory to save registered images
    ids : list, optional
        List of indices of images to register. The default is None.
    **kwargs : dict
        Keyword arguments for rigid_alignment_block
    """

    # check all directories exist
    if not exists(out_dir):
        makedirs(out_dir)
        print(f'{out_dir} does not exist, creating it')
    if not exists(source_dir):
        raise ValueError(f'{source_dir} does not exist')
    if not exists(target_dir):
        raise ValueError(f'{target_dir} does not exist')

    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    dtype = torch.float

    src_files = listdir(source_dir)
    src_files.sort()
    target_files = listdir(target_dir)
    target_files.sort()
    if ids is not None:
        src_files = [src_files[i] for i in ids]
        target_files = [target_files[i] for i in ids]

    # now loop through the files
    fig0,ax0 = plt.subplots(1,3)
    fig1,ax1 = plt.subplots() # this will be my stripes figure
    start1 = time.time()
    # for fileN,fileM in list(zip(filesO[-3:],filesP[-3:])):
    for fileM,fileN in list(zip(src_files, target_files)):
        # fileN is a list of files, one for each fluorescent channel. FileM is the myelin image
        print(f'registering image {fileM} and {fileN}')

        with tifffile.TiffFile(join(source_dir, fileM)) as tif:
            try:
                dx = tif.shaped_metadata[0]['pixelsizex']
                dy = tif.shaped_metadata[0]['pixelsizey']
            except:
                dx = tif.pages[0].tags.values()[19].value['pixelsizex']
                dy = tif.pages[0].tags.values()[19].value['pixelsizey']
            I0 = tif.asarray()

        print('I0 shape: ', I0.shape)
        if I0.dtype == np.uint8:
            I0 = I0.astype(float)/255.0
        else:
            # I0 = I0.astype(float) / np.mean(I0.reshape(-1,I0.shape[-1], axis=0))
            I0 = I0.astype(float) / np.max(I0, axis=(0,1))
        dI = np.array([dy,dx])
        nI = np.array(I0.shape[:-1])
        xI = [np.arange(n)*d - (n-1)*d/2 for n,d in zip(nI,dI)]
        n_scatter = 3
        n_init= 0
        n_final=0
    #     start_downsample1 = time.time()
        # downsample them and scatter
        xI, I, labels = downsample(nI,xI, I0,['R','G','B'],n_scatter,n_init,n_final)
    #     end_downsample1 = time.time()
        print('I after downsample: ', I.shape)

        # for I we will rotate
        J = []
        label_J = []
        done = np.zeros(I.shape[-1])
        for i in range(I.shape[-1]):
            if done[i]:
                continue

            label = labels[i]    
            rotated_label_1 = rot(label)
            rotated_label_2 = rot(rotated_label_1)
            rotated_label_3 = rot(rotated_label_2)

            #print(label,rotated_label_1,rotated_label_2,rotated_label_3)    
            ind_1 = labels.index(rotated_label_1)
            ind_2 = labels.index(rotated_label_2)
            ind_3 = labels.index(rotated_label_3)
            #print(ind_1)
            if ind_1 == i:
                # this is the case where everything is zeros
                J.append(I[...,i])
                done[i] = 1
            else:        
                J.append(I[...,i]*0.25+I[...,ind_1]*0.25+I[...,ind_2]*0.25+I[...,ind_3]*0.25)    
                done[i] = 1
                done[ind_1] = 1
                done[ind_2] = 1
                done[ind_3] = 1        
            label_J.append(label)
        J = np.stack(J,-1)


        # now we are looking for reflections
        # it seems that a lot of the reflections are gone
        L = []
        label_L = []
        done = np.zeros(J.shape[-1])
        for i in range(J.shape[-1]):
            if done[i]:        
                continue

            label = label_J[i]    
            reflected_label = reflect(label)


            try:
                ind = label_J.index(reflected_label)
            except:
                try:
                    ind = label_J.index(rot(reflected_label))
                except:
                    try:
                        ind = label_J.index(rot(rot(reflected_label)))
                    except:
                        try:
                            ind = label_J.index(rot(rot(rot(reflected_label))))
                        except:
                            print('couldn\'t find one')
                            ind = i # don't do anything    
            if ind == i:
                # this is the case where everything is zeros
                L.append(J[...,i])
                done[i] = 1
                label_L.append(label)
            else:                
                L.append(J[...,i]*0.5+J[...,ind]*0.5)    

                done[i] = 1
                done[ind] = 1        
                label_L.append(label)

        L = np.stack(L,-1)
        Lp,s = pca(L)
        Ls = pca_scale(Lp,s)


        with tifffile.TiffFile(join(target_dir, fileN)) as tif:
            try:
                dx = tif.shaped_metadata[0]['pixelsizex']
                dy = tif.shaped_metadata[0]['pixelsizey']
            except:
                dx = tif.pages[0].tags.values()[19].value['pixelsizex']
                dy = tif.pages[0].tags.values()[19].value['pixelsizey']
            J = tif.asarray()

        print('J shape: ', J.shape)
        if J.dtype == np.uint8:
            J = J.astype(float)/255.0
        else:
            # J = J.astype(float) / np.mean(J.reshape((-1, J.shape[-1])), axis=0)
            J = J.astype(float) / np.max(J, axis=(0,1))
        nJ = np.array(J.shape[:-1])
        dJ = np.array([dy,dx])
        xJ = [np.arange(n)*d - (n-1)*d/2 for n,d in zip(nJ,dJ)]

        n_scatter = 0
        n_init=3
        n_final=0
    #     start_downsample2 = time.time()
        xJ, J,labels_ = downsample(nJ,xJ,J,['R','G','B'],n_scatter, n_init, n_final)
    #     end_downsample2 = time.time()
        print('J after downsample: ', J.shape)

        ax0[0].cla()
        ax0[0].imshow(I[...,:3])
        ax0[1].cla()
        ax0[1].imshow(Ls[...,:3])
        ax0[2].cla()
        try:
            ax0[2].imshow(J[...,:3])# / J.max())
        except:   
            toshow = np.concatenate((J,np.zeros_like(J[...,0])[...,None]),-1)
            ax0[2].imshow(toshow)
        fig0.canvas.draw()

        Inp = np.array(L)
        Jnp = np.array(J)
        xInp = [np.array(x) for x in xI]
        xJnp = [np.array(x) for x in xJ]
        dInp = np.array(dI)
        dJnp = np.array(dJ)

        I = torch.tensor(Inp.transpose(-1,0,1),dtype=dtype,device=device)
        J = torch.tensor(Jnp.transpose(-1,0,1),dtype=dtype,device=device)
        dI = torch.tensor(dInp,dtype=dtype,device=device)
        dJ = torch.tensor(dJnp,dtype=dtype,device=device)
        xI = [torch.tensor(x,dtype=dtype,device=device) for x in xInp]
        xJ = [torch.tensor(x,dtype=dtype,device=device) for x in xJnp]

        # need a quick COM for init (TODO)
        Imax = torch.quantile(I[:3],0.99)
        Jmax = torch.quantile(J[:3],0.99)
        COMI = torch.stack((torch.sum((Imax-I[:3])*xI[0][None,:,None])/torch.sum(Imax-I[:3]), torch.sum((Imax-I[:3])*xI[1][None,None,:])/torch.sum(Imax-I[:3])))
        COMJ = torch.stack((torch.sum((Jmax-J)*xJ[0][None,:,None])/torch.sum(Jmax-J), torch.sum((Jmax-J)*xJ[1][None,None,:])/torch.sum(Jmax-J)))


        A = torch.tensor([[1.0,0.0,COMJ[0]-COMI[0]],[0.0,1.0,COMJ[1]-COMI[1]],[0.0,0.0,1.0]],device=device,dtype=dtype)

    #     start2 = time.time()
        if 'niter' in kwargs:
            niter = kwargs['niter']
        else:
            niter = 4000
        if 'epL' in kwargs:
            epL = kwargs['epL']
        else:
            epL = 1e-3
        if 'epT' in kwargs:
            epT = kwargs['epT']
        else:
            epT = 1e-6
        Anew,AI,fAI,fig = rigid_alignment_block(xI,I,xJ,J,A=A, title='scattering',device=device,dtype=dtype,epL=epL,epT=epT,niter=niter)

        end = time.time()

        makedirs(out_dir,exist_ok=True)

        outname = splitext(split(fileN)[1])[0] + '_to_' + splitext(split(fileM)[1])[0] + '_registration'
        fig.savefig(join(out_dir,outname+'.jpg'))
        np.savez(join(out_dir,outname+'.npz'),A=np.array(Anew.cpu().numpy()), I=Inp, J=Jnp, xI=xInp, xJ=xJnp)

        print(f'time elapsed for image pair: {end-start1}')

    return


def transform_series(out_dir, src_dir, target_dir, transforms_dir, src_files, target_files, transform_list, device=None):
    """
    Transform a series of images using a list of transforms

    Parameters
    ----------
    out_dir : str
        output directory
    src_dir : str
        source image directory
    target_dir : str
        target image directory
    transforms_dir : str
        directory containing transforms
    src_files : list of str
        list of source image filenames
    target_files : list of str
        list of target image filenames
    transform_list : list of str
        list of transform filenames (npz files)
    device : torch.device, optional
    """

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float
    start = time.time()
    for fileM,fileN,npzfile in zip(src_files, target_files, transform_list):
        print("loading: ",fileM)
        with tifffile.TiffFile(join(src_dir, fileM)) as tif:
            try:
                dx = tif.shaped_metadata[0]['pixelsizex']
                dy = tif.shaped_metadata[0]['pixelsizey']
            except:
                dx = tif.pages[0].tags.values()[19].value['pixelsizex']
                dy = tif.pages[0].tags.values()[19].value['pixelsizey']
            I0 = tif.asarray()

        if I0.dtype == np.uint8:
            I0 = I0.astype(float)/255.0
        else:
            # I0 = I0.astype(float) / np.mean(I0.reshape((-1, I0.shape[-1])), axis=0)
            I0 = I0.astype(float) / np.max(I0, axis=(0,1))
        I = torch.tensor(I0,dtype=dtype,device=device)
        dI = np.array([dy,dx])
        nI = np.array(I0.shape)
        xI = [np.arange(n)*d - (n-1)*d/2 for n,d in zip(nI,dI)]

        print('I shape: ', I0.shape)

        xI = [torch.tensor(x,dtype=dtype,device=device) for x in xI]

        with tifffile.TiffFile(join(target_dir,fileN)) as tif:
            try:
                dx = tif.shaped_metadata[0]['pixelsizex']
                dy = tif.shaped_metadata[0]['pixelsizey']
            except:
                dx = tif.pages[0].tags.values()[19].value['pixelsizex']
                dy = tif.pages[0].tags.values()[19].value['pixelsizey']
            nJ = np.array([tif.pages[0].imagelength,
                        tif.pages[0].imagewidth])

        dJ = np.array([dy,dx])
        xJ = [np.arange(n)*d - (n-1)*d/2 for n,d in zip(nJ,dJ)]
        xJ = [torch.tensor(x, dtype=dtype, device=device) for x in xJ]
        XJ = torch.stack(torch.meshgrid(xJ,indexing='ij'),-1)

        npz = np.load(join(transforms_dir,npzfile), allow_pickle=True)
        A = npz['A']
        A = torch.tensor(A,dtype=dtype,device=device)
        Ai = torch.linalg.inv(A)
        Xs = (Ai[:2,:2]@XJ[...,None])[...,0] + Ai[:2,-1]
        # scale 0 to 1
        Xs = Xs - torch.stack([xI[0][0],xI[1][0]])
        Xs = Xs / torch.stack([xI[0][-1] - xI[0][0],xI[1][-1] - xI[1][0]])
        # scale -1 to 1
        Xs = 2*Xs - 1
        # convert to xy
        Xs = Xs.flip(-1)

        # sample
        AI = torch.nn.functional.grid_sample(I.permute(-1,0,1)[None],Xs[None],align_corners=True,padding_mode='border')[0]
        end = time.time()
        print(f'time to apply transform: {end-start}')
        print(AI.shape)
        outname = npzfile.split('registration')[0] + 'AI.tif'
        print("saved AI to ", join(out_dir,outname))
        tifffile.imwrite(join(out_dir,outname),AI[:3].detach().permute(1,2,0).cpu().numpy(),
                        metadata={'pixelsizex':dJ[1], 'pixelsizey':dJ[0]})

    end = time.time()

    print(f'time to load images: {end-start}')

    return