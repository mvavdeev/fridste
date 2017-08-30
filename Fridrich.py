from PIL import Image
from scipy import fftpack
import numpy as np
from math import copysign
from random import normalvariate


def form_data(data,N1,N2,height,width):
    return np.array([data[i:i+N1, j:j+N2] for i in range(0,height//N1*N1, N1) for j in range(0,width//N2*N2, N2)])

def deform_data(vix,count1,count2):
    return np.row_stack([np.column_stack([vix[i+count2*j] for i in range(count2)]) for j in range(count1)]).ravel()

def conv_data(form_data):
    G=[]
    N_del=len(form_data)
    for i in range(N_del):
        N1=len(form_data[i])
        N2=len(form_data[i][0])
        form_data_mean=np.mean(form_data[i])
        form_data_std=np.std(form_data[i])
        G.append((1024/np.sqrt(N1*N2))*((form_data[i]-form_data_mean)/form_data_std))
    return np.array(G,dtype=np.float64)

def iconv_data(G_vix):
    N_del=len(G_vix)
    C_vix=[]
    for i in range(N_del):
        C_vix.append((G_vix[i]+np.abs(np.min(G_vix[i])))/(np.max(G_vix[i]+np.abs(np.min(G_vix[i]))))*255)
    C_vix_float=np.array(C_vix,dtype=np.float64)+0.5
    return np.array(C_vix_float,dtype=int)

def DKP(G):
    DKP_G=[]
    N_del=len(G)
    for i in range(N_del):
        DKP_G.append(fftpack.dct(fftpack.dct(G[i], axis=0, norm='ortho'),axis=1, norm='ortho'))
    return np.array(DKP_G,dtype=np.float64)

def IDKP(DKP_G):
    N_del=len(DKP_G)
    G_vix=[]
    for i in range(N_del):
        G_vix.append(fftpack.idct(fftpack.idct(DKP_G[i], axis=0, norm='ortho'), axis=1, norm='ortho'))
    return np.array(G_vix,dtype=np.float64)

def gen_posl(N_soob,start_gen,power_gen):
    d=int(np.log2(power_gen*N_soob)+0.5)
    RS = np.random.mtrand.RandomState(start_gen) 
    posl=[RS.uniform(0,1,2**d) for i in range(N_soob)]
    return np.array(posl)



def vstraivanie(cont_file,soob_file,skrit_file,L_low,L_mid,alpha,start_gen,gamma=1,count1=1,count2=1,
                low_mode=True,
                mid_mode=True, power_posl=16,power_gen=4,
                clone_mode=True,min_DKP=0,qual=100,
                R_mode=False,G_mode=False,B_mode=True,clone_color_mode=True,light=True):
    
    cont=Image.open(cont_file)
    
    width_cont = cont.size[0] 
    height_cont = cont.size[1]
    N1=height_cont//count1
    N2=width_cont//count2
    N_del=count1*count2
    
    if cont.getbands()==('R','G','B'):
        RGB_mode=True

        count_color=R_mode+G_mode+B_mode
    
        R_cont_data=np.array(cont.getdata())[:,0]
        R_cont_data.shape=(height_cont,width_cont)    
    
        G_cont_data=np.array(cont.getdata())[:,1]
        G_cont_data.shape=(height_cont,width_cont)
    
        B_cont_data=np.array(cont.getdata())[:,2]
        B_cont_data.shape=(height_cont,width_cont)
        if light:
            Y=0.299*R_cont_data+0.587*G_cont_data+0.114*B_cont_data
            Cb=(-0.1687*R_cont_data-0.3313*G_cont_data+0.5*B_cont_data).ravel()
            Cr=(0.5*R_cont_data-0.4187*G_cont_data-0.0813*B_cont_data).ravel()
    else:
        RGB_mode=False
        cont_data = np.array(cont,dtype=np.float64)
        
    if RGB_mode:
        if light:
            Y_form_data=form_data(Y,N1,N2,height_cont,width_cont)
        else:
            if R_mode:
                R_form_cont_data=form_data(R_cont_data,N1,N2,height_cont,width_cont)
            if G_mode:
                G_form_cont_data=form_data(G_cont_data,N1,N2,height_cont,width_cont)
            if B_mode:
                B_form_cont_data=form_data(B_cont_data,N1,N2,height_cont,width_cont)
    else:
        form_cont_data=form_data(cont_data,N1,N2,height_cont,width_cont)
    
    soob=Image.open(soob_file)
    width_soob = soob.size[0]
    height_soob = soob.size[1]
    soob_data=np.array(soob.getdata())//255
    N_soob=len(soob_data)
    if RGB_mode:
        if light:
            form_soob_data=soob_data
        else:
            if not clone_color_mode:
                form_soob_data=[soob_data[i:i+N_soob//count_color] for i in range(0,N_soob//count_color*count_color,N_soob//count_color)]
                form_soob_data[-1]=soob_data[-N_soob//count_color:]
                N_soob=[len(i) for i in form_soob_data]
            else:
                form_soob_data=soob_data
    else:
        form_soob_data=soob_data
    
    if RGB_mode:
        if light:
            Y_G=conv_data(Y_form_data)
        else:
            if R_mode:
                R_G=conv_data(R_form_cont_data)
            if G_mode:
                G_G=conv_data(G_form_cont_data)
            if B_mode:
                B_G=conv_data(B_form_cont_data)
    else:
        G=conv_data(form_cont_data)
    
    def fun_tau(a):
        ans=[]
        i=1
        t=1
        while t<5096:
            t=((1+a)/(1-a))**(i-1)
            ans.append(t)
            i+=1
        return ans
    
    tau=fun_tau(alpha)
    
    def ind(t):
        if t<1:
            return 1
        i=0
        while t>=tau[i]:
            i+=1
        return 1-1*(i%2)
    
    if RGB_mode:
        if light:
            DKP_Y_G=DKP(Y_G)
        else:
            if R_mode:
                DKP_R_G=DKP(R_G)
            if G_mode:
                DKP_G_G=DKP(G_G)
            if B_mode:
                DKP_B_G=DKP(B_G)
    else:
        DKP_G=DKP(G)
        
    def low(DKP_G,soob_data,L_low,clone_mode,min_DKP):
        N_soob=len(soob_data)
        N_del=len(DKP_G)
        N1=len(DKP_G[0])
        N2=len(DKP_G[0][0])
        if not clone_mode:
            form_soob=[soob_data[i:i+N_soob//N_del] for i in range(0,N_soob//N_del*N_del,N_soob//N_del)]
            form_soob[-1]=soob_data[-N_soob//N_del:]
        else:
            form_soob=[soob_data]*N_del
            
        for b in range(N_del):
            H_low=L_low
            d=abs(np.fliplr(DKP_G[b]))>=min_DKP
            l_d=len(d[0])
            if np.sum([np.sum(d.diagonal(l_d-1-i)) for i in range(L_low,2*l_d)])<N_soob:
                raise AssertionError ('Не хватает коэффициентов ДКП')
            while np.sum([np.sum(d.diagonal(l_d-1-i)) for i in range(L_low,H_low+1)])<N_soob:
                H_low+=1
            j=0
            for u in range(N1):
                if j>=len(form_soob[b]):
                    break
                for v in range(N2):
                    if L_low<=u+v<=H_low:
                        d=abs(DKP_G[b][u][v])
                        if d>=min_DKP:
                            if ind(d)==form_soob[b][j]:
                                for i in range(len(tau)):
                                    if d<tau[i]:
                                        if i==0:
                                            t1=0
                                            t2=tau[i]
                                        else:
                                            t1=tau[i-1]
                                            t2=tau[i]
                                        break
                                if (abs(d-t1)<0.01) or (abs(d-t2)<0.01):
                                    DKP_G[b][u][v]=copysign(1,DKP_G[b][u][v])*normalvariate((t1+t2)/2,(t2-t1)/13)
                            else:
                                for i in range(len(tau)):
                                    if d<tau[i]:
                                        if i<=1:
                                            t1=tau[i]
                                            t2=tau[i+1]
                                        else:
                                            t1=tau[i-2]
                                            t2=tau[i-1]
                                        break
                                DKP_G[b][u][v]=copysign(1,DKP_G[b][u][v])*normalvariate((t1+t2)/2,(t2-t1)/13)
                            j+=1
                    if j>=len(form_soob[b]):
                        break
        return DKP_G
    
    if low_mode:
        if RGB_mode:
            if light:
                DKP_Y_G=low(DKP_Y_G,form_soob_data,L_low,clone_mode,min_DKP)
            else:
                if not clone_color_mode:
                    i=0
                    if R_mode:
                        DKP_R_G=low(DKP_R_G,form_soob_data[i],L_low,clone_mode,min_DKP)
                        i+=1
                    if G_mode:
                        DKP_G_G=low(DKP_G_G,form_soob_data[i],L_low,clone_mode,min_DKP)
                        i+=1
                    if B_mode:
                        DKP_B_G=low(DKP_B_G,form_soob_data[i],L_low,clone_mode,min_DKP)
                    del i
                else:
                    if R_mode==True:
                        DKP_R_G=low(DKP_R_G,form_soob_data,L_low,clone_mode,min_DKP)
                    if G_mode==True:
                        DKP_G_G=low(DKP_G_G,form_soob_data,L_low,clone_mode,min_DKP)
                    if B_mode==True:
                        DKP_B_G=low(DKP_B_G,form_soob_data,L_low,clone_mode,min_DKP)
        else:
            DKP_G=low(DKP_G,form_soob_data,L_low,clone_mode,min_DKP)

    def izvl_segm(posl,power_posl,soob_data):
        N_soob=len(soob_data)
        N_mid=len(posl[0])-power_posl
        nu=[]
        for i in range(N_soob):
            if soob_data[i]==1:
                m=random.randint(0,power_posl//2)
                nu.append(posl[i][m:N_mid+m])
            else:
                m=random.randint(power_posl//2,power_posl)
                nu.append(posl[i][m:N_mid+m])
        return np.array(nu),N_mid
    
    def gen_sum(nu,N_del,N_soob):
        Spr=[]
        for b in range(N_del):
            Spr.append((np.sum(nu[N_soob//N_del*b:N_soob//N_del*(b+1)],axis=0)-
                       N_soob/N_del/2)/np.sqrt(N_soob/N_del/12))
        return np.array(Spr)
    
    def mid(DKP_G,Spr,gamma,L_mid,min_DKP):
        N_mid=len(Spr[0])
        N_del=len(DKP_G)
        N1=len(DKP_G[0])
        N2=len(DKP_G[0][0])
        H_mid=L_mid
        
        for b in range(N_del):
            H_mid=L_mid
            d=abs(np.fliplr(DKP_G[b]))>=min_DKP
            l_d=len(d[0])
            if np.sum([np.sum(d.diagonal(l_d-1-i)) for i in range(L_mid,2*l_d)])<N_mid:
                raise AssertionError ('Не хватает коэффициентов ДКП')
            while np.sum([np.sum(d.diagonal(l_d-1-i)) for i in range(L_mid,H_mid+1)])<N_mid:
                H_mid+=1
            j=0
            for u in range(N1):
                if j>=N_mid:
                    break
                for v in range(N2):
                    if L_mid<=u+v<=H_mid:
                        d=abs(DKP_G[b][u][v])
                        if d>=min_DKP:
                            DKP_G[b][u][v]+=gamma*Spr[b][j]
                            j+=1
                    if j>=N_mid:
                        break
        return DKP_G
    
    if mid_mode:
        if RGB_mode:
            if light:
                posl=gen_posl(N_soob,start_gen,power_gen)
                nu,N_mid=izvl_segm(posl,power_posl,soob_data)
                Spr=gen_sum(nu,N_del,N_soob)
                DKP_Y_G=mid(DKP_Y_G,Spr,gamma,L_mid,min_DKP)
            else:
                if not clone_color_mode:
                    i=0
                    N_mid=N_soob.copy()
                    posl=[gen_posl(j,start_gen,power_gen) for j in N_soob] 
                    if R_mode:
                        nu,N_mid[i]=izvl_segm(posl[i],power_posl,soob_data[i])
                        Spr=gen_sum(nu,N_del,N_soob[i])
                        DKP_R_G=mid(DKP_R_G,Spr,gamma,L_mid,min_DKP)
                        i+=1
                    if G_mode:
                        nu,N_mid[i]=izvl_segm(posl[i],power_posl,soob_data[i])
                        Spr=gen_sum(nu,N_del,N_soob[i])
                        DKP_G_G=mid(DKP_G_G,Spr,gamma,L_mid,min_DKP)
                        i+=1
                    if B_mode:
                        nu,N_mid[i]=izvl_segm(posl[i],power_posl,soob_data[i])
                        Spr=gen_sum(nu,N_del,N_soob[i])
                        DKP_B_G=mid(DKP_B_G,Spr,gamma,L_mid,min_DKP)
                        del i
                else:
                    posl=gen_posl(N_soob,start_gen,power_gen)
                    nu,N_mid=izvl_segm(posl,power_posl,soob_data)
                    Spr=gen_sum(nu,N_del,N_soob)
                    if R_mode:
                        DKP_R_G=mid(DKP_R_G,Spr,gamma,L_mid,min_DKP)
                    if G_mode:
                        DKP_G_G=mid(DKP_G_G,Spr,gamma,L_mid,min_DKP)
                    if B_mode:
                        DKP_B_G=mid(DKP_B_G,Spr,gamma,L_mid,min_DKP)
        else:
            posl=gen_posl(N_soob,start_gen,power_gen)
            nu,N_mid=izvl_segm(posl,power_posl,soob_data)
            Spr=gen_sum(nu,N_del,N_soob)
            DKP_G=mid(DKP_G,Spr,gamma,L_mid,min_DKP)

    else:
        N_mid=None
        posl=None
 
    if RGB_mode:
        if light:
            Y_G_vix=IDKP(DKP_Y_G)
        else:
            if R_mode:
                R_G_vix=IDKP(DKP_R_G)
            if G_mode:
                G_G_vix=IDKP(DKP_G_G)
            if B_mode:
                B_G_vix=IDKP(DKP_B_G)
    else:
        G_vix=IDKP(DKP_G)

    if RGB_mode:
        if light:
            Y_C_vix=iconv_data(Y_G_vix)
        else:
            if R_mode:
                R_C_vix=iconv_data(R_G_vix)
            if G_mode:
                G_C_vix=iconv_data(G_G_vix)
            if B_mode:
                B_C_vix=iconv_data(B_G_vix)
    else:
        C_vix=iconv_data(G_vix)
        
    if RGB_mode:
        if light:
            Y_C_vixod=deform_data(Y_C_vix,count1,count2)
        else:
            if R_mode:
                R_C_vixod=deform_data(R_C_vix,count1,count2)
            else:
                R_C_vixod=R_cont_data.ravel()
            if G_mode:
                G_C_vixod=deform_data(G_C_vix,count1,count2)
            else:
                G_C_vixod=G_cont_data.ravel()
            if B_mode:
                B_C_vixod=deform_data(B_C_vix,count1,count2)
            else:
                B_C_vixod=B_cont_data.ravel()
    else:
        C_vixod=deform_data(C_vix,count1,count2)
        
    if RGB_mode:
        if light:
            R1=np.array(Y_C_vixod+1.402*Cr+0.5,dtype=int)
            G1=np.array(Y_C_vixod-0.34414*Cb-0.71414*Cr+0.5,dtype=int)
            B1=np.array(Y_C_vixod+1.772*Cb+0.5,dtype=int)
            skrit=Image.new('RGB',(width_cont,height_cont))
            skrit.putdata([tuple(i) for i in np.column_stack((R1,G1,B1))])
        else:
            skrit=Image.new('RGB',(width_cont,height_cont))
            skrit.putdata([tuple(i) for i in np.column_stack((R_C_vixod,G_C_vixod,B_C_vixod))])
    else:
        skrit=Image.new('L',(width_cont,height_cont))
        skrit.putdata(C_vixod)
    skrit.save(skrit_file, quality=qual, optimize=True, progressive=True)
    return N_soob,N_mid,posl,width_soob,height_soob



def izvlechenie(skrit_file,otkrit_file,N_soob,N_mid,posl,width_soob,height_soob,L_low,L_mid,
                alpha,count1=1,count2=1, low_mode=True,mid_mode=True, clone_mode=True,min_DKP=0,
                R_mode=False,G_mode=False,B_mode=True,clone_color_mode=True,power_posl=16,light=True):
    
    answer=[]
    
    skrit=Image.open(skrit_file)
    
    width_skrit = skrit.size[0]
    height_skrit = skrit.size[1]
    N1=height_skrit//count1
    N2=width_skrit//count2
    N_del=count1*count2
    
    if skrit.getbands()==('R','G','B'):
        RGB_mode=True 
    
        R_skrit_data=np.array(skrit.getdata())[:,0]
        R_skrit_data.shape=(height_skrit,width_skrit)    
    
        G_skrit_data=np.array(skrit.getdata())[:,1]
        G_skrit_data.shape=(height_skrit,width_skrit)
    
        B_skrit_data=np.array(skrit.getdata())[:,2]
        B_skrit_data.shape=(height_skrit,width_skrit)
        if light:
            Y=0.299*R_skrit_data+0.587*G_skrit_data+0.114*B_skrit_data
            Cb=-0.1687*R_skrit_data-0.3313*G_skrit_data+0.5*B_skrit_data
            Cr=0.5*R_skrit_data-0.4187*G_skrit_data-0.0813*B_skrit_data
    else:
        RGB_mode=False
        skrit_data = np.array(skrit,dtype=np.float64)
        
    if RGB_mode:
        if light:
            Y_form_skrit_data=form_data(Y,N1,N2,height_skrit,width_skrit)
        else:
            if R_mode:
                R_form_skrit_data=form_data(R_skrit_data,N1,N2,height_skrit,width_skrit)
            if G_mode:
                G_form_skrit_data=form_data(G_skrit_data,N1,N2,height_skrit,width_skrit)
            if B_mode:
                B_form_skrit_data=form_data(B_skrit_data,N1,N2,height_skrit,width_skrit)
    else:
        form_skrit_data=form_data(skrit_data,N1,N2,height_skrit,width_skrit)
        
    if RGB_mode:
        if light:
            Y_G_skrit=conv_data(Y_form_skrit_data)
        else:
            if R_mode:
                R_G_skrit=conv_data(R_form_skrit_data)
            if G_mode:
                G_G_skrit=conv_data(G_form_skrit_data)
            if B_mode:
                B_G_skrit=conv_data(B_form_skrit_data)
    else:
        G_skrit=conv_data(form_skrit_data)
        
    if RGB_mode:
        if light:
            DKP_Y_G_skrit=DKP(Y_G_skrit)
        else:
            if R_mode:
                DKP_R_G_skrit=DKP(R_G_skrit)
            if G_mode:
                DKP_G_G_skrit=DKP(G_G_skrit)
            if B_mode:
                DKP_B_G_skrit=DKP(B_G_skrit)
    else:
        DKP_G_skrit=DKP(G_skrit)

    def fun_tau(a):
        ans=[]
        i=1
        t=1
        while t<5096:
            t=((1+a)/(1-a))**(i-1)
            ans.append(t)
            i+=1
        return ans
    
    tau=fun_tau(alpha)
    
    def ind(t):
        if t<1:
            return 1
        i=0
        while t>=tau[i]:
            i+=1
        return 1-1*(i%2)
    
    def izvl_low(DKP_G_skrit,N_soob,L_low,clone_mode,min_DKP):
        theory=[]
        N_del=len(DKP_G_skrit)
        N1=len(DKP_G_skrit[0])
        N2=len(DKP_G_skrit[0][0])
        
        if not clone_mode:
            form_soob=[N_soob//N_del for i in range(N_del)]
            form_soob[-1]=N_soob//N_del+N_soob%N_del
        else:
            form_soob=[N_soob]*N_del
            
        for b in range(N_del):
            H_low=L_low
            d=abs(np.fliplr(DKP_G_skrit[b]))>=min_DKP
            l_d=len(d[0])
            while np.sum([np.sum(d.diagonal(l_d-1-i)) for i in range(L_low,H_low+1)])<N_soob:
                H_low+=1
            j=0
            theory.append([])
            for u in range(N1):
                if j>=form_soob[b]:
                    break
                for v in range(N2):
                    if L_low<=u+v<=H_low:
                        d=abs(DKP_G_skrit[b][u][v])
                        if d>=min_DKP:
                            theory[b].append(ind(d))
                            j+=1
                    if j>=form_soob[b]:
                        break
        if not clone_mode:
            answer=[]
            for i in theory:
                answer+=i
            return np.array(answer)
        else:
            return np.mean(np.array(theory),axis=0)
        
    if low_mode:
        if RGB_mode:
            if light:
                theory_low=izvl_low(DKP_Y_G_skrit,N_soob,L_low,clone_mode,min_DKP)
            else:
                if not clone_color_mode:
                    theory_low=[]
                    i=0
                    if B_mode==True:
                        theory_low=np.hstack((izvl_low(DKP_B_G_skrit,N_soob[i],L_low,clone_mode,min_DKP),theory_low))
                        i+=1
                    if G_mode==True:
                        theory_low=np.hstack((izvl_low(DKP_G_G_skrit,N_soob[i],L_low,clone_mode,min_DKP),theory_low))
                        i+=1
                    if R_mode==True:
                        theory_low=np.hstack((izvl_low(DKP_R_G_skrit,N_soob[i],L_low,clone_mode,min_DKP),theory_low))
                        i+=1
                    del i
                else:
                    theory_low=[]
                    if R_mode:
                        theory_low.append(izvl_low(DKP_R_G_skrit,N_soob,L_low,clone_mode,min_DKP))
                    if G_mode:
                        theory_low.append(izvl_low(DKP_G_G_skrit,N_soob,L_low,clone_mode,min_DKP))
                    if B_mode:
                        theory_low.append(izvl_low(DKP_B_G_skrit,N_soob,L_low,clone_mode,min_DKP))
                    theory_low=np.mean(np.array(theory_low),axis=0)
        else:
            theory_low=izvl_low(DKP_G_skrit,N_soob,L_low,clone_mode,min_DKP)
        answer.append(theory_low)
        
    def izvl_mid(DKP_G_skrit,L_mid,N_mid,N_soob,posl,power_posl,min_DKP):
        theory=[]
        N_del=len(DKP_G_skrit)
        N1=len(DKP_G_skrit[0])
        N2=len(DKP_G_skrit[0][0])
        
        for b in range(N_del):
            H_mid=L_mid
            d=abs(np.fliplr(DKP_G_skrit[b]))>=min_DKP
            l_d=len(d[0])
            while np.sum([np.sum(d.diagonal(l_d-1-i)) for i in range(L_mid,H_mid+1)])<N_mid:
                H_mid+=1
            j=0
            mid=[]
            for u in range(N1):
                if j>=N_mid:
                    break
                for v in range(N2):
                    if L_mid<=u+v<=H_mid:
                        d=abs(DKP_G_skrit[b][u][v])
                        if d>=min_DKP:
                            mid.append(DKP_G_skrit[b][u][v])
                            j+=1
                    if j>=N_mid:
                        break
            mid=np.array(mid)
            for n in range(N_soob//N_del):
                nu1=np.array(posl[n+N_soob//N_del*b])
                K=[(np.sum(mid*nu1[m:N_mid+m])) for m in range(power_posl)]
                if K.index(max(K))<power_posl//2:
                    theory.append(1)
                else:
                    theory.append(0)
        return np.array(theory).ravel()
    
    if mid_mode:
        if RGB_mode:
            if light:
                theory_mid.append(izvl_mid(DKP_Y_G_skrit,L_mid,N_mid,N_soob,posl,power_posl,min_DKP))
            else:
                if not clone_color_mode:
                    theory_mid=[]
                    i=0
                    if G_mode:
                        theory_mid=np.hstack((izvl_mid(DKP_G_G_skrit,L_mid,N_mid[i],N_soob[i],posl[i],power_posl,min_DKP),theory_mid))
                        i+=1
                    if B_mode==True:
                        theory_mid=np.hstack((izvl_mid(DKP_B_G_skrit,L_mid,N_mid[i],N_soob[i],posl[i],power_posl,min_DKP),theory_mid))
                        i+=1
                    if R_mode==True:
                        theory_mid=np.hstack((izvl_mid(DKP_R_G_skrit,L_mid,N_mid[i],N_soob[i],posl[i],power_posl,min_DKP),theory_mid))
                        i+=1
                    del i
                else:
                    theory_mid=[]
                    if R_mode:
                        theory_mid.append(izvl_mid(DKP_R_G_skrit,L_mid,N_mid,N_soob,posl,power_posl,min_DKP))
                    if B_mode:
                        theory_mid.append(izvl_mid(DKP_B_G_skrit,L_mid,N_mid,N_soob,posl,power_posl,min_DKP))
                    if G_mode:
                        theory_mid.append(izvl_mid(DKP_G_G_skrit,L_mid,N_mid,N_soob,posl,power_posl,min_DKP))
                    theory_mid=np.mean(np.array(theory_mid),axis=0)
        else:
            theory_mid=izvl_mid(DKP_G_skrit,L_mid,N_mid,N_soob,posl,power_posl,min_DKP)
            
        answer.append(theory_mid)
        
    answer=np.mean(answer,axis=0)
    answer=np.array(answer+0.5,dtype=int)
    
    otkrit=Image.new('L',(width_soob,height_soob))
    otkrit.putdata(answer*255)
    otkrit.save(otkrit_file,mode='L')



def ocenka(soob_file,otkrit_file,cont_file,skrit_file):
    soob=Image.open(soob_file)
    soob_data=np.array(soob.getdata())//255
    
    otkrit=Image.open(otkrit_file)
    otkrit_data=np.array(otkrit.getdata())//255
    
    cont=Image.open(cont_file)
    if cont.getbands()==('R','G','B'):
        cont_data=np.array(cont.getdata())[:,2]
    else:
        cont_data=np.array(cont.getdata())
    
    skrit=Image.open(skrit_file)
    if skrit.getbands()==('R','G','B'):
        skrit_data=np.array(skrit.getdata())[:,2]
    else:
        skrit_data=np.array(skrit.getdata())
    print('Процент верно извлеченных пикселей =',np.sum(soob_data==otkrit_data)/len(soob_data)*100)
    print('Количество ошибочно извлеченных пикселей',len(soob_data)-np.sum(soob_data==otkrit_data))
    print('MSE =',np.sum((skrit_data-cont_data)**2,dtype=np.float64)/len(skrit_data))
    a=sum(cont_data**2,dtype=np.float64)
    b=sum((skrit_data-cont_data)**2,dtype=np.float64)
    c=a/b
    print('SNR =',10*np.log10(c))
          
