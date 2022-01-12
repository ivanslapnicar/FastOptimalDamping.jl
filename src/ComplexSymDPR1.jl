 # module ComplexDPR1

import Base: *, getindex, size, Matrix
import LinearAlgebra: eigen, eigvals, inv, transpose, mul!

T=Complex{Float64}

# export DPR1

# Define general Complex Diagonal+Rank-One Type D+ρ u*transpose(u)
struct CSymDPR1{T} <: AbstractMatrix{T}
    D::Vector{T} # diagonal
    u::Vector{T} # rank one, length n
    ρ::T  # rho
end

# Define its size
size(A::CSymDPR1, dim::Integer) = length(A.D)
size(A::CSymDPR1) = size(A,1), size(A,1)

# Index into a CSymDPR1
function getindex(A::CSymDPR1,i::Integer,j::Integer)
    Aij = A.ρ * (A.u[i] * A.u[j])
    return i == j ? A.D[i] + Aij : Aij
end # getindex

Matrix(A::CSymDPR1)= Matrix(Symmetric(BLAS.syr!('U',A.ρ,A.u,diagm(0=>A.D))))
transpose(A::CSymDPR1{T}) where T = A

# Define CSymDPR1 x vector in O(n) operations
function *(A::CSymDPR1{T}, x::Vector{T}) where T
    dotd= T==ComplexF64 ? BLAS.dotu : BLAS.dot
    y=Diagonal(A.D)*x
    BLAS.axpy!(A.ρ*dotd(A.u, x),A.u,y)
    # return y
end
function mul!(y::Vector{T}, A::CSymDPR1{T}, x::Vector{T}) where T
    dotd= T==ComplexF64 ? BLAS.dotu : BLAS.dot
    mul!(y,Diagonal(A.D),x)
    BLAS.axpy!(A.ρ*dotd(A.u, x),A.u,y)
end

# CSymDPR1 x Matrix
function *(A::CSymDPR1{T}, x::Matrix{T}) where T
    y=Diagonal(A.D)*x
    BLAS.ger!(A.ρ,A.u,conj(transpose(x)*A.u),y)
end
function mul!(y::Matrix{T}, A::CSymDPR1{T}, x::Matrix{T}) where T
    mul!(y,Diagonal(A.D),x)
    BLAS.ger!(A.ρ,A.u,conj(transpose(x)*A.u),y)
end


function inv(A::CSymDPR1{T}) where {T<:T}
    xₕ=A.u ./A.D
    Δₕ=one(T)./A.D
    ρₕ=-A.ρ/(one(T)+BLAS.dotu(xₕ,A.u)*A.ρ)
    return CSymDPR1(Δₕ,xₕ,ρₕ)
end

function inv(A::CSymDPR1{T}) where {T<:Float64}
    xₕ=A.u ./A.D
    Δₕ=one(T)./A.D
    ρₕ=-A.ρ/(one(T)+BLAS.dot(xₕ,A.u)*A.ρ)
    return CSymDPR1(Δₕ,xₕ,ρₕ)
end

function xtAx(A::CSymDPR1{T}, x::Vector{T}) where {T}
    xdx = zero(T)
    xu  = zero(T)
    @inbounds for i in 1:length(x)
        xdx += A.D[i]*x[i]^2
        xu  += x[i]*A.u[i]
    end
    return xdx + A.ρ*xu^2
end

function xtAx(A::CSymDPR1{T}, x::Vector{T}, μ::T) where {T}
    xdx = zero(T)
    xu  = zero(T)
    @inbounds for i in 1:length(x)
        xdx += (A.D[i]-μ)*x[i]^2
        xu  += x[i]*A.u[i]
    end
    return xdx + A.ρ*xu^2
end

function xtAxc(A::CSymDPR1{T}, x::Vector{T}, μ::T) where {T}
    xdx = zero(T)
    xu  = zero(T)
    xuc = zero(T)
    @inbounds for i in 1:length(x)
        xdx += (A.D[i]-μ)*x[i]*conj(x[i])
        xu  += A.u[i]*x[i]
        xuc += A.u[i]*conj(x[i])
    end
    return xdx + A.ρ*xu*xuc
end

function xtAxc(A::CSymDPR1{T}, x::Vector{T}) where {T}
    xdx = zero(T)
    xu  = zero(T)
    xuc = zero(T)
    @inbounds for i in 1:length(x)
        xdx += A.D[i]*x[i]*conj(x[i])
        xu  += A.u[i]*x[i]
        xuc += A.u[i]*conj(x[i])
    end
    return xdx + A.ρ*xu*xuc
end


function dpr1vector!(v::Vector{T}, A::CSymDPR1{T}, λ::T) where  T
    n = length(A.D)
    @inbounds for l in 1:n
        τ=A.D[l] - λ
        if τ==zero(T)
            for k=1:n
                v[k]=zero(T)
            end
            v[l]=one(T)
            return v
        end
        v[l] = A.u[l]/τ #  (A.D[l] - λ)
    end
    normalize!(v)
    # v
end

function mrqi!(z::Vector{T}, Du::Vector{T}, A::CSymDPR1{T}, λ::T) where T
    # Modified Rayleigh Quotient Iteration
    # Returns z=inv(A.D-λ)*A.u if γ is small or too large
    # and z=inv(A.D-λI)*z otherwise
    n=length(z)
    @inbounds for l in 1:n
        τ=A.D[l] - λ
        if τ==zero(T)
            for k=1:n
                z[k]=zero(T)
            end
            z[l]=one(T)
            return
        end
        Du[l] = A.u[l]/τ #  (A.D[l] - λ)
    end
    dotd= (T==Float64) ? BLAS.dot : BLAS.dotu
    γ=A.ρ*dotd(z,Du)/(one(T)+A.ρ*dotd(A.u,Du))
    if abs(γ)<1e-4 || abs(γ)>1e4 || isnan(γ)
        @inbounds for l=1:n
            z[l]=Du[l]
        end
    else
        @inbounds for l=1:n
            z[l]=z[l]/(A.D[l]-λ)-γ*Du[l]
        end
    end
    normalize!(z)
end

function rqi(z::Vector{T}, z₁::Vector{T}, τ::Vector{T}, A::CSymDPR1{T} ,λ::T) where T
    # Rayleigh Quotient Iteration
    # Returns λ=z'*A*z/z'*z
    # and z=inv(A-λI)*z otherwise
    dotd= (T==Float64) ? BLAS.dot : BLAS.dotu
    if count(isequal(λ),A.D)>0
        return λ
    end
    n=length(A.D)
    dpr1vector!(z,A,λ)
    δ = xtAxc(A, z)
    if count(isequal(δ),A.D)>0
        λ=δ
        return λ
    else
        copy!(z₁,A.u)
        copy!(τ,A.D)
        τ.-=δ
        z₁./=τ
        μ=dotd(A.u,z₁)
        γ=-A.ρ/(one(T)+μ*A.ρ)
        ξ=dotd(z₁,z)
        γ*=ξ
        z./=τ
        BLAS.axpy!(γ,z₁,z)
        return xtAxc(A, z)/BLAS.dotc(z,z)
    end
end

function eigvals(A₁::CSymDPR1)
    # A is complex and unreduced, that is, A.u.!=0 and
    # all elements of A.D are different.
    # It is better to use z^T A z / z^T z (modified RQI) instead
    # of standard RQI using z^*
    # This Standard Rayleigh Quotient Iteration, but solving the system
    # (multiplication by the inverse) is replaced by the CSymDPR1()
    # eigenvector formula

    # Preliminaries
    A=deepcopy(A₁)
    T = typeof(A.D[1])
    n₁=length(A.D)
    λ = Vector{T}(undef,n₁)

    if deflation==true
        τ=n₁*norm(A)*eps()
        def=findall(abs.(A.u).<=τ .&& abs.(A.u).<=abs.(A.D)*sqrt(eps()))
        # println("Deflated eigenvalues",count(def))
        λ[n₁-length(def)+1:n₁]=A.D[def]
        deleteat!(A.D,def)
        deleteat!(A.u,def)
    end

    n = length(A.D)
    m = n
    # println("n = ",n)

    # Eigenvalues
    # Auxiliary arrays
    d = Vector{T}(undef,n)
    z = Vector{T}(undef,n)
    Du = Vector{T}(undef,n)
    AD=Vector{T}(undef,n)
    dotd= (T==Float64) ? BLAS.dot : BLAS.dotu

    # Tolerance
    tol = 100.0*eps()
    # Maxiter
    maxiter = 101

    for l in 1:n

        copyto!(AD,A.D)
        # Shift to the absolutely largest pole
        t=1
        μ = A.D[t]
        for i in 1:m
            A.D[i] -= μ
        end
        for i=1:m
            z[i]=zero(T)
        end
        z[t]=one(T)
        k = 1
        δ = one(T)
        γ = zero(T)
        while k < maxiter && abs(δ)>tol
            # use z^T
            ν = dotd(z, z)
            δ = xtAx(A, z)/ν
            γ += δ
            mrqi!(z, Du, A, δ)
            for i in 1:m
                A.D[i] -= δ
            end

            k += 1
        end
        if k==maxiter
            println(l," ",abs(δ)," ",tol)
        end
        λ[l] = μ + γ

        # Deflation - eliminating absolutely largest pole
        @inbounds for i in 1:(t - 1)
            d[i] = A.u[i]/sqrt(one(T) - γ/(AD[i] - μ))
            if isnan(d[i])
                println(l," nan")
                d[i]=zero(T)
            end
        end
        @inbounds for i in (t + 1):m
            d[i-1] = A.u[i]/sqrt(one(T) - γ/(AD[i] - μ))
            if isnan(d[i-1])
                d[i-1]=zero(T)
            end
        end
        m -= 1
        deleteat!(A.D,t)
        deleteat!(AD,t)
        deleteat!(z,t)
        deleteat!(Du,t)
        copyto!(A.D,AD)
        deleteat!(A.u,t)
        for i=1:m
            A.u[i]=d[i]
        end
    end
    # Two steps of RQI on the original CSymDPR1 matrix
    # Generate workspace for each thread

    nthreads=Threads.nthreads()
    # nthreads=1
    z₀ = [Vector{T}(undef,n₁) for i = 1:nthreads]
    z₁ = [Vector{T}(undef,n₁) for i = 1:nthreads]
    τ = [Vector{T}(undef,n₁) for i = 1:nthreads]
    # Threaded loop
    Threads.@threads for l=1:n
        tid=Threads.threadid()
        μ=λ[l]
        λ[l]=rqi(z₀[tid],z₁[tid],τ[tid],A₁,μ)
    end
    return λ
end

# Scaling
function scale!(X::CauchyLike{T}) where T
    n=length(X.x)
    dotd= (T==Float64) ? BLAS.dot : BLAS.dotu
    a=[Vector{T}(undef,n) for i=1:Threads.nthreads()]
    Ψ=Vector{T}(undef,Threads.nthreads())
    rzero=count(iszero,X.r)
    Threads.@threads for i=1:n
        tid=Threads.threadid()
        getindex!(a[tid],X,:,i)
        Ψ[tid]=dotd(a[tid],a[tid])
        if !isfinite(Ψ[tid]) || count(iszero,a[tid])>rzero
            X.s[i]=zero(T)
        else
            X.s[i]/=conj(sqrt(Ψ[tid]))
        end
    end
end

function eigen(A::CSymDPR1{T}) where T
    # A is complex and unreduced, that is, A.u.!=0 and
    # all elements of A.D are different.
    # eig() is just eigvals with normalized eigenvectors returned
    # as Cauchy-like matrix
    Λ=eigvals(A)
    n=length(A.D)
    # Boley-Golub Lemma
    z=Vector{T}(undef,n)
    Threads.@threads for i=1:n
        δ=A.D[i]
        τ=Λ[i]-δ
        for j=1:i-1
            τ*=(Λ[j]-δ)/(A.D[j]-δ)
        end
        for j=i+1:n
            τ*=(Λ[j]-δ)/(A.D[j]-δ)
        end
        τ/=A.ρ
        τ=sqrt(τ)
        if isnan(τ)
            # println(" nan ")
            z[i]=A.u[i]
        else
            if T==Float64
                z[i]=copysign(τ,A.u[i])
            else
                z[i]=copysign(real(τ),real(A.u[i]))+im*copysign(imag(τ),imag(A.u[i]))
            end
        end
    end

    nz=norm(A.u-z,Inf)
    if nz>100*eps()
        # println("warning: nz=",nz)
    end
    X=CauchyLike(A.D,Λ,z,ones(T,length(A.u)))
    scale!(X)
    return Λ,X
end

function traceX(S::CauchyLike{T},s::Int) where T
    n=div(length(S.x),2)
    G=Matrix{T}(undef,2n,2s)
    a=[Vector{T}(undef,2n) for i=1:Threads.nthreads()]
    b=[Vector{T}(undef,2n) for i=1:Threads.nthreads()]
    Threads.@threads for i=1:2n
        tid=Threads.threadid()
        getindex!(a[tid],S,:,i)
        G[i,:].=view(a[tid],[1:s;n+1:n+s])
    end
    Y=CauchyLikeS(S.y,G,conj(G))
    # trace(X)=trace(SYS')=trace(S'(SY))
    S₁=S*Y

    traceX=zeros(T,Threads.nthreads())
    Threads.@threads for i=1:2n
        tid=Threads.threadid()
        getindex!(a[tid],S,:,i)
        getindex!(b[tid],S₁,:,i)
        traceX[tid]+=BLAS.dotc(a[tid],b[tid])
    end
    return sum(traceX),S,Y
end
