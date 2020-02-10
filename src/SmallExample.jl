# Generate Small example, d=400, n=801, 2n=1602
example='S'
K,M=GenerateExamples(example)

# Create DampedSystem structure
DS=DampedSystem(M,K,[1.0 0;0 1])

# Compute changes of basis
Ω,Φ,Ξ=ChangeOfBasis(DS)

# Eigenvalue decomposition of a CSymDPR1 matrix.
# l is position of damper, ρ is viscosity
T=ComplexF64
l=50
ρ=50.0
y=Φ[l,:]
A=CSymDPR1(Ξ,y,map(T,ρ))
println("Time of single eigenvalue decomposition of a CSymDPR1 matrix")
@time Λ,S=eigen(A)

# Computation of trace(X) for a single damper in position l with viscosity ρ, and
# the number of damped frequencies s
s=27

function f(ρ::Number)
    A=CSymDPR1(Ξ,Φ[l,:],ComplexF64(ρ))
    Λ,S=eigen(A)
    Tr=traceX(S,s)
    return real(Tr)
end
println("Time of trace computation for a single damper")
@time Tr=f(50.0)


#  MULTIPLE DAMPERS WITH DIFFERENT VISCOSITIES
# Computation of trace(X) for three dampers in positions l with viscosities ρ, and
# the number of damped frequencies s

# Positions of dampers
l=[50,550,120]
# Optimial viscosities
ρ=[561.813,651.586,310.602]

# Auxiliary arrays
n₁=400
n=2*n₁+1
U=Matrix{T}(undef,2n,length(l))
Uvec=Matrix{T}(undef,2n,length(l))

function f(ρ::Vector)
    Uvec[:,1]=Φ[l[1],:]
    U[:,1]=Φ[l[1],:]
    A=CSymDPR1(Ξ,Φ[l[1],:],ComplexF64(ρ[1]))
    Λ,S=eigen(A)
    y=similar(Λ)
    # Loop
    for j=2:length(l)
        U[:,j]=Φ[l[j],:]
        if j==2
           U[:,j]-=Φ[l[j+1]+n₁,:]
        end
        At_mul_B!(y,S,U[:,j])
        Uvec[:,j]=y
        A=CSymDPR1(Λ,y,ComplexF64(ρ[j]))
        Λ,S₁=eigen(A)
        # Multiplication of linked Cauchy-like matrices
        S=S*S₁
    end
    Tr=traceX(S,s)
    println("ρ = ", ρ," trace = ", Tr)
    return real(Tr)
end

println("Time of trace computation for three dampers")
@time Tr=f(ρ)

# Optimization. Given starting ρ, the optmial viscosities should be as above
ρ=[100.0,100,100]

println("Time for optimization of viscosities for three dampers")
@time optimize(f,ρ,method=ConjugateGradient(eta=0.01),
    x_tol=1e-2,f_tol=1e-6)
