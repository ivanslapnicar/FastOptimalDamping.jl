# Generate Small example, d=800, n=1601, 2n=3202
example='L'
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
println("Time of single eigenvalue decomposition of a CSymDPR1 matrix.")
println("Second timing is relevant\n")
@time Λ,S=eigen(A)
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
println("\nTime of trace computation for a single damper.")
println("Second timing is relevant.\n")
@time Tr=f(50.0)
@time Tr=f(50.0)


#  MULTIPLE DAMPERS WITH DIFFERENT VISCOSITIES
# Computation of trace(X) for three dampers in positions l with viscosities ρ, and
# the number of damped frequencies s

# Positions of dampers and viscosities
l=[50,950,220]
ρ=[100.0,200.0,300.0]

# Auxiliary arrays
n₁=800
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

println("\nTime of trace computation for three dampers.")
println("Second timing is relevant.\n")
@time Tr=f(ρ)
@time Tr=f(ρ)

# Optimization. Given starting ρ, we compute the optmial viscosities
ρ=[100.0,100,100]
println("\nTime for optimization of viscosities for three dampers.\n")
@time result=optimize(f,ρ,method=ConjugateGradient(eta=0.01),
    x_tol=1e-2,f_tol=1e-6)

println(result)
println("\nOptimal viscosities: ",result.minimizer)
println("\nMinimal trace: ", result.minimum)
