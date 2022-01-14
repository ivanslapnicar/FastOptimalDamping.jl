# Generate Small example, d=1000, n=2001, 2n=4002
example='H'
K,M=GenerateExamples(example)

# Create DampedSystem structure
DS=DampedSystem(M,K,[1.0 0;0 1])
println(" K(K,M) = ", cond(inv(sqrt(M))*K*inv(sqrt(M))))
# Compute changes of basis
Ω,Φ,Ξ,Q,Γ,Φ₀=ChangeOfBasis(DS)

# Eigenvalue decomposition of a CSymDPR1 matrix.
# l is position of damper, ρ is viscosity
T=ComplexF64
l=50
ρ=50.0
y=Φ[l,:]
A=CSymDPR1(Ξ,y,map(T,ρ))
println("Time of single eigenvalue decomposition of a CSymDPR1 matrix.")
println("Second timing is relevant\n")
@time Λ,S=eigen(A)
@time Λ,S=eigen(A)


# Computation of trace(X) for a single damper in position l with viscosity ρ, and
# the number of damped frequencies s
s=20

function f(ρ::Number)
    A=CSymDPR1(Ξ,Φ[l,:],ComplexF64(ρ))
    Λ,S=eigen(A)
    Tr,rest=traceX(S,s)
    return real(Tr)
end
println("\nTime of trace computation for a single damper.")
println("Second timing is relevant.\n")
@time Tr=f(50.0)
# @time Tr=f(50.0)


#  MULTIPLE DAMPERS WITH DIFFERENT VISCOSITIES
# Computation of trace(X) for three dampers in positions l with viscosities ρ, and
# the number of damped frequencies s

# Positions of dampers and viscosities
l=[850,1950,20]
ρ=[100.0,200.0,300.0]

# Auxiliary arrays
n₁=1000
n=2*n₁+1
U=Matrix{T}(undef,2n,length(l))
Uvec=Matrix{T}(undef,2n,length(l))

# Standard matrix
ep1=zeros(n); ep2=zeros(n); ep3=zeros(n)
ep1[l[1]]=1; ep2[l[2]]=1; ep3[l[3]]=1
ep2[l[3]+n₁]=-1
eptimesPhi=[transpose(ep1);transpose(ep2);transpose(ep3)]*Φ₀

function f(ρ::Vector)
    Uvec[:,1]=Φ[l[1],:]
    U[:,1]=Φ[l[1],:]
    A=CSymDPR1(Ξ,Φ[l[1],:],ComplexF64(ρ[1]))
    Λ,S=eigen(A)
    println("cond(S) = ", cond(Matrix(S)))
    y=similar(Λ)
    # Loop
    for j=2:length(l)
        U[:,j]=Φ[l[j],:]
        if j==2
           U[:,j]-=Φ[l[j+1]+n₁,:]
        end
        lmult!(y,S,U[:,j])
        Uvec[:,j]=y
        A=CSymDPR1(Λ,y,ComplexF64(ρ[j]))
        # println(" A j = ", j," cond ", cond(Matrix(A)))
        Λ,S₁=eigen(A)
        println(" S₁  j = ", j," cond ", cond(Matrix(S₁)))
        # Multiplication of linked Cauchy-like matrices
        S=S*S₁
    end
    println(" cond(S) = ",cond(Matrix(S)))

    Tr, S₀, Y₀=traceX(S,s)
    X₀=Matrix(S₀)*Matrix(Y₀)*Matrix(S₀)'
    G₀=[Matrix{Float64}(I,n,s) zeros(n,s);zeros(n,s) Matrix{Float64}(I,n,s)]
    A₀=Diagonal(Ξ)+transpose(Q)*[zero(Diagonal(Ω)) zero(Diagonal(Ω)); zero(Diagonal(Ω)) transpose(eptimesPhi)*Diagonal(ρ)*eptimesPhi]*Q
    # Residual
    no=norm(A₀*X₀+X₀*A₀'+G₀*G₀')/norm(X₀)
    println(" residual ", no)

    println("ρ = ", ρ," trace = ", Tr)
    return real(Tr)
end

println("\nTime of trace computation for three dampers.")
println("Second timing is relevant.\n")
@time Tr=f(ρ)
@time Tr=f(ρ)

# Optimization. Given starting ρ, we compute the optmial viscosities
ρ=[500.0,500,500]
println("\nTime for optimization of viscosities for three dampers.\n")
@time result=optimize(f,ρ,method=ConjugateGradient(eta=0.01),
    x_tol=1e-2,f_tol=1e-6)

println(result)
println("\nOptimal viscosities: ",result.minimizer)
println("\nMinimal trace: ", result.minimum)
