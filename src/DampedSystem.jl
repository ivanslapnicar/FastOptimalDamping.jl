struct DampedSystem{T}
    M::AbstractMatrix{T} # Mass matrix
    K::AbstractMatrix{T} # Stiffness matrix
    D::AbstractMatrix{T} # Internal damping matrix, currently unused
end

using SparseArrays
function ChangeOfBasis(S::DampedSystem)
    T=Complex{Float64}
    n=size(S.K,1)
    # Solve the GEVD
    Kf=deepcopy(S.K)
    for i=1:n,j=1:n
        Kf[j,i]/=(sqrt(S.M.diag[j]*S.M.diag[i]))
    end
    Ω2,q=eigen(Kf)
    Φ=q./sqrt.(S.M.diag)
    Ω=sqrt.(Ω2)
    # Γ is diagonal in the Φ basis
    # Γ=diag(Φ'*D_int*Φ)
    α=0.02
    Γ=α*Ω
    # Form the matrices Ξ and Q
    Ξ=zeros(T,2n)
    Q=spzeros(T,2n,2n)
    # Solve small 2x2 hyperbolic GEVDs
    # Q is kept as a sparse matrix
    # These are direct formulas for the case (3) and (10)
    γ=(sqrt(2.0-α)*sqrt(2.0+α)*im-α)*0.5
    c₁=sqrt((1-conj(γ))*(1+conj(γ)))
    c₂=sqrt((1-γ)*(1+γ))
    Q₂=[1.0/c₁ 1.0/c₂; conj(γ)/c₁ γ/c₂]
    for k=1:n
        Ξ[k]=conj(γ)*Ω[k]
        Ξ[n+k]=γ*Ω[k]
        Q[[k,n+k],[k,n+k]]=Q₂
    end
    Φ=Φ*Q[n+1:2n,:]
    return Ω,Φ,Ξ,Q
end
