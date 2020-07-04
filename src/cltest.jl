Random.seed!(123)
n=10
x=rand(ComplexF64,n)
y=rand(ComplexF64,n)
y[2:5]=x[2:5]
z=rand(ComplexF64,n)
z[3:7].=y[3:7]
r=rand(ComplexF64,n,3)
s=rand(ComplexF64,n,3)
r1=rand(ComplexF64,n) #,2)
s1=rand(ComplexF64,n) # ,2)
A=CauchyLike(x,y,r,s)
B=CauchyLike(y,z,r1,s1)
C=A*B
norm(Matrix(A)*Matrix(B)-Matrix(C))
