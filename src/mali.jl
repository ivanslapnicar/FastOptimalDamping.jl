P=[1 im;im -1]
q,r=eigen(P)
a=5.0
Pd=CSymDPR1([a-im,a-2-im],[sqrt(im),sqrt(im)],one(ComplexF64))
E=eigen(Pd)
cond(E.vectors),E.values, a-1
