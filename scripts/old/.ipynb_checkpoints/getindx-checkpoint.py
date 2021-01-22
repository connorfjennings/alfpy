from alf_vars import *
from linterp import tsum, locate


def intind(lam, spec, lo, hi):
    
    """
    !perform integral over spectrum for index computation
    """

    nn = lam.size

    #!take care of the ends
    ll1 = max(min(locate(lam, lo), nn-1),1)
    ll2 = max(min(locate(lam, hi), nn-1),2)
    f1 = (spec[ll1]-spec[ll1-1])/(lam[ll1]-lam[ll1-1])*(lo-lam[ll1-1])+spec[ll1-1]
    f2 = (spec[ll2]-spec[ll2-1])/(lam[ll2]-lam[ll2-1])*(hi-lam[ll2-1])+spec[ll2-1]
    
    if ll1 == ll2:
        intind = (f2+f1)/2.*(hi-lo)
    else:
        intind = tsum(lam[ll1 : ll2], spec[ll1 : ll2])
        intind += (lam[ll1]-lo)*(f1+spec[ll1])/2.
        intind += (hi-lam[ll2-1])*(f2+spec[ll2-1])/2.

    return intind



def getindx(lam,spec,indices):
    """
    !routine to calculate indices from an input spectrum
    !indices are defined in fsps/data/allindices.dat
    """


    indices = 0.0
    nn = lam.size
    
    for j in range(nindx):
        if indxdef[6,j] <=2:
            #blue continuum
            cb = intind(lam,spec,indxdef(3,j),indxdef(4,j))
            cb = cb / (indxdef(4,j)-indxdef(3,j))     
            lb = (indxdef(3,j)+indxdef(4,j))/2.
        
            #!red continuum 
            cr = intind(lam,spec,indxdef(5,j),indxdef(6,j))
            cr = cr / (indxdef(6,j)-indxdef(5,j))
            lr = (indxdef(5,j)+indxdef(6,j))/2.
        
            #!compute integral(fi/fc)
            #!NB: fc here is a linear interpolation between the red and blue.
            intfifc = intind(lam,spec/((cr-cb)/(lr-lb)*(lam-lb)+cb),&
                 indxdef(1,j),indxdef(2,j))
        
            IF (indxdef(7,j).EQ.1.) THEN
               !compute magnitude
               indices(j) = -2.5*LOG10(intfifc/(indxdef(2,j)-indxdef(1,j)))
            ELSE IF (indxdef(7,j).EQ.2.) THEN
               !compute EW (in Ang)
               indices(j) = (indxdef(2,j)-indxdef(1,j)) - intfifc

        
            #!set dummy values for indices defined off of the wavelength grid
            IF (indxdef(6,j) > lambda(nn)) indices(j) = 999.0
            IF (indxdef(3,j) < lambda(1))  indices(j) = 999.0

    elif (indxdef(7,j) == 3.):
      
        #!compute CaT index
        for i in range(3):
            #!blue continuum
            cb = intind(lambda,spec,indxcat(3,i),indxcat(4,i))
            cb = cb / (indxcat(4,i)-indxcat(3,i))     
            lb = (indxcat(3,i)+indxcat(4,i))/2.
           
            #!red continuum 
            cr = intind(lambda,spec,indxcat(5,i),indxcat(6,i))
            cr = cr / (indxcat(6,i)-indxcat(5,i))
            lr = (indxcat(5,i)+indxcat(6,i))/2.
        
            #!compute integral(fi/fc)
            !NB: fc here is a linear interpolation between the red and blue.
            intfifc = intind(lambda,spec/((cr-cb)/(lr-lb)*(lambda-lb)+cb),&
                 indxcat(1,i),indxcat(2,i))
        
            indices(j) = indices(j) + (indxcat(2,i)-indxcat(1,i)) - intfifc

            #!set dummy values for indices defined off of the wavelength grid
            IF (indxcat(6,i) > lambda(nn)) indices(j) = 999.0
            IF (indxcat(3,i) < lambda(1))  indices(j) = 999.0

