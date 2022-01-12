struct DampedSystem{T}
    M::AbstractMatrix{T} # Mass matrix
    K::AbstractMatrix{T} # Stiffness matrix
    D::AbstractMatrix{T} # Internal damping matrix
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
    for k=1:n
        Ξ[[k,n+k]],Q[[k,n+k],[k,n+k]]=eigen([0 Ω[k];  Ω[k] Γ[k]], [1.0 0; 0 -1])
        # Transform Q such that Q^T*[I 0;0 -I]*Q=I
        Θ=diag(transpose(Q[[k,n+k],[k,n+k]])*[1.0 0; 0 -1]*Q[[k,n+k],[k,n+k]])
        Q[[k,n+k],k]./=sqrt(Θ[1])
        Q[[k,n+k],n+k]./=sqrt(Θ[2])
    end
    Φ₀=copy(Φ)
    Φ=Φ*Q[n+1:2n,:]
    return Ω,Φ,Ξ,Q,Γ,Φ₀
end
