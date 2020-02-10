import Base: size, transpose, -, *, getindex, Matrix
import LinearAlgebra: mul!

# Define a CauchyLike Matrix Type
struct CauchyLike{T} <: AbstractMatrix{T}
    x::Vector{T}
    y::Vector{T}
    r::Array{T}  #ovaj ulaz može biti vektor ili matrica
    s::Array{T}  #ovaj ulaz može biti vektor ili matrica
end # struct

size(A::CauchyLike{T}, dim::Integer) where T = dim==1 ? length(A.x) : length(A.y)
size(A::CauchyLike{T}) where T = size(A,1), size(A,2)

struct CauchyLikeS{T} <: AbstractMatrix{T}
    x::Vector{T}
    r::AbstractArray{T}  # ovaj ulaz može biti vektor ili matrica
    s::AbstractArray{T}  # s=conj(r) tako da je kasnije brže
end # struct

# Define its size
size(A::CauchyLikeS{T}, dim::Integer) where T = length(A.x)
size(A::CauchyLikeS{T}) where T = size(A,1), size(A,2)

# Index into a CauchyLikeS
function getindex(A::CauchyLikeS{T},i::Integer,j::Integer) where T
    return (-A.r[j,:]⋅A.r[i,:])/(A.x[i]+conj(A.x[j]))
end # getindex
Matrix(A::CauchyLikeS{T}) where T =(-A.r*A.r')./(A.x.+Adjoint(A.x))

function getindex!(y::Vector{T},A::CauchyLikeS{T},i::Colon,j::Int) where T
    n=length(A.x)
    α=conj(A.x[j])
    if ndims(A.r)==1
        β=-conj(A.r[j])
        Future.copy!(y,A.r)
        y.*=β
        y./=(A.x.+α)
    else
        BLAS.gemv!('N',-one(T),A.r,view(A.s,j,:),zero(T),y)
        y./=(A.x.+α)
    end
end # getindex

function Ac_mul_B!(y::Array{T},A::CauchyLikeS{T},x::Array{T}) where T
    # Computes y=adjoint(A)*x
    dotd= T==ComplexF64 ? BLAS.dotc : BLAS.dot
    n=size(A,1)
    a=[Vector{T}(undef,n) for i=1:Threads.nthreads()]
    # Threads.@threads
    # Iz nekog razloga je nestabilno kada se ovo uključi
    # U 1.1. izgleda OK
    Threads.@threads for i=1:n
        tid=Threads.threadid()
        getindex!(a[tid],A,:,i)
        for j=1:size(y,2)
            # ne poziva se - c ili u
            y[i,j]=dotd(a[tid],view(x,:,j))
        end
    end
end

function *(A::CauchyLike{T},Y::CauchyLikeS{T}) where T
    if A.y==Y.x
        # Fast multiplication of Chained Cauchy-like matrices
        # (aka. V.Pan, A.Zheng)
        n=div(size(A,1),2)
        s=div(size(Y.r,2),2)
        M=[A.r Matrix{T}(-I,2n,s) [zeros(T,n,s);Matrix{T}(-I,n,s)]]
        # M=[A.r [-I,fill(zero(T),2n,s] [fill(zero(T),n,s);-I,fill(zero(T),n-s,s)]]
        # M=[A.r -A*Y.r]
        z=similar(A.s)
        Ac_mul_B!(z,Y,A.s)
        N=[z Y.r]
        return CauchyLike(A.x,-conj(Y.x),M,N)
    else
        # Standard multiplication
        return Matrix(A)*Matrix(B)
    end
end

function househ!(a::AbstractVector{T},g::Vector{T}) where T
    dotd= T==ComplexF64 ? BLAS.dotu : BLAS.dot
    v=similar(g)
    copyto!(v,g)
    β=dotd(v,v)
    # v1=v[1]
    v[1]+=sqrt(β)
    # H=I-(2.0/(β+2*v1*sqrt(β)+β))*v*transpose(v)
    τ=(2.0*dotd(v,Vector(a)))/dotd(v,v)
    if !isnan(τ)
        for i=1:length(a)
            a[i]-=τ*v[i]
        end
    end
    # a.-=(2.0*BLAS.dotu(v,a)/(β+2*v1*sqrt(β)+β))*v
end


# Index into a CauchyLike
function getindex(A::CauchyLike{T},i::Integer,j::Integer) where T
    if ndims(A.r)==1
        return (conj(A.s[j])*A.r[i])/(A.x[i]-A.y[j])
    else
        return (A.s[j,:]⋅A.r[i,:])/(A.x[i]-A.y[j])
    end
end # getindex


# Submatrices
function getindex(A::CauchyLike{T},i::Union{UnitRange,Vector{Int},Colon},
    j::Union{UnitRange,Vector{Int},Colon}) where T
    return CauchyLike(A.x[i],A.y[j],A.r[i,:],A.s[j,:])
end # getindex

# Column
function getindex(A::CauchyLike{T},i::Colon,j::Int) where T
    return (A.r*A.s[j,:]')./(A.x.-A.y[j])
end # getindex

# Row
function getindex(A::CauchyLike{T},i::Int,j::Colon) where T
    return (A.r[i]*A.s')./(A.x[i].-A.y)
end # getindex

# Column in place
import Future
function getindex!(y::Vector{T},A::CauchyLike{T},i::Colon,j::Int) where T
    n=length(A.x)
    α=A.y[j]
    if ndims(A.r)==1
        β=conj(A.s[j])
        Future.copy!(y,A.r)
        y.*=β
        y./=(A.x.-α)
        # y=(A.r*conj(A.s[j]))./(A.x.-A.y[j])
    else
        BLAS.gemv!('N',one(T),A.r,conj(A.s[j,:]),zero(T),y)
        y./=(A.x.-α)
    end
    # This part is for CauchyLike{T} matrix which is
    # eigenvector matrix of some SymDPR1{T} matrix
    if count(isnan,y)>0
        # typeof(findfirst(isnan,y))!=Nothing
        ind=findall(isnan,y)
        # Our case
        for i=1:n
            y[i]=zero(T)
        end
        if ndims(A.r)==1
            y[j]=one(T)
            if length(ind)>1
                househ!(view(y,ind),A.r[ind])
            end
        else
            m=size(A.r,2)
            y[j]=one(T)
            if length(ind)>1
                for i=m:-1:1
                    househ!(view(y,ind[i:end]),Uvec[ind[i:end],i])
                end
            end
        end
    end
end # getindex

# Basic functions for CauchyLike
# Matrix(A::CauchyLike{T}) where T =(A.r*A.s')./(A.x.-transpose(A.y))
function Matrix(A::CauchyLike{T}) where T
    n=length(A.x)
    Z=Matrix{T}(undef,n,n)
    a=[Vector{T}(undef,n) for i=1:Threads.nthreads()]
    # Threads.@threads
    for i=1:n
        tid=Threads.threadid()
        getindex!(a[tid],A,:,i)
        # view?
        # Future.copy!(Z[:,i],a[tid])
        Z[:,i]=a[tid]
    end
    return Z
end
transpose(A::CauchyLike{T}) where T =CauchyLike(-A.y,-A.x,conj(A.s),conj(A.r))
adjoint(A::CauchyLike{T}) where T =CauchyLike(conj(-A.y),conj(-A.x),A.s,A.r)
-(A::CauchyLike{T}) where T =CauchyLike(-A.x,-A.y,A.r,A.s)


# Multiplication with CauchyLike
function *(A::CauchyLike{T}, x::AbstractVector{T}) where T
    n=length(x)
    y=[zeros(T,n) for i=1:Threads.nthreads()]
    z=[Vector{T}(undef,n) for i=1:Threads.nthreads()]
    # Ovo napravi malu numeričku razliku
    Threads.@threads for i=1:n
        tid=Threads.threadid()
        getindex!(z[tid],A,:,i)
        BLAS.axpy!(x[i],z[tid],y[tid])
    end
    return sum(y)
end

function mul!(y1::Vector{T}, A::CauchyLike{T}, x::Vector{T}) where T
    n=length(x)
    y=[zeros(T,n) for i=1:Threads.nthreads()]
    z=[Vector{T}(undef,n) for i=1:Threads.nthreads()]
    Threads.@threads for i=1:n
        tid=Threads.threadid()
        getindex!(z[tid],A,:,i)
        BLAS.axpy!(x[i],z[tid],y[tid])
    end
    Future.copy!(y1,sum(y))
end

function *(A::CauchyLike{T}, x::Matrix{T}) where T
    # This could be improved?
    # It works well for narrow x
    dotd= T==ComplexF64 ? BLAS.dotc : BLAS.dot
    n=size(A,1)
    a=[Vector{T}(undef,n) for i=1:Threads.nthreads()]
    y=similar(x)
    #=
    for i=1:size(x,2)
        y[:,i]=A*view(x,:,i)
    end
    =#
    Threads.@threads for i=1:n
        tid=Threads.threadid()
        getindex!(a[tid],A,:,i)
        for j=1:size(y,2)
            # ne poziva se - c ili u
            y[i,j]=dotd(a[tid],view(x,:,j))
        end
    end
    return y
end

full(A::CauchyLike)=(A.r*A.s')./(A.x.-transpose(A.y))

# Multiplication of CauchyLike
function *(A::CauchyLike{T}, B::CauchyLike{T}) where T
    if A.y==B.x
        # Fast multiplication of Chained Cauchy-like matrices
        # (aka. V.Pan, A.Zheng)
        M=[A.r (A*B.r)]
        N=similar(A.s)
        Ac_mul_B!(N,B,A.s)
        N=hcat(N, B.s)
        C=CauchyLike(A.x,B.y,M,N)
        return C
    else
        # Standard multiplication
        return Matrix(A)*Matrix(B)
    end
end

function *(A::CauchyLike{T}, B::CauchyLike{T}, y::Vector{T}) where T
    if A.y==B.x
        # Fast multiplication of Chained Cauchy-like matrices
        # (aka. V.Pan, A.Zheng)
        M=[A.r y]
        N=similar(A.s)
        Ac_mul_B!(N,B,A.s)
        N=hcat(N, B.s)
        C=CauchyLike(A.x,B.y,M,N)
        return C
    else
        # Standard multiplication
        return Matrix(A)*Matrix(B)
    end
end

function At_mul_B!(y::Vector{T},A::CauchyLike{T},x::Vector{T}) where T
    # Computes y=transpose(A)*x
    dotd= T==ComplexF64 ? BLAS.dotu : BLAS.dot
    n=length(x)
    a=[Vector{T}(undef,n) for i=1:Threads.nthreads()]
    Threads.@threads for i=1:n
        tid=Threads.threadid()
        getindex!(a[tid],A,:,i)
        y[i]=dotd(a[tid],x)
    end
end

function Ac_mul_B!(y::Vector{T},A::CauchyLike{T},x::Vector{T}) where T
    # Computes y=Adjoint(A)*x
    dotd= T==ComplexF64 ? BLAS.dotc : BLAS.dot
    n=length(x)
    a=[Vector{T}(undef,n) for i=1:Threads.nthreads()]
    Threads.@threads for i=1:n
        tid=Threads.threadid()
        getindex!(a[tid],A,:,i)
        y[i]=dotd(a[tid],x)
    end
end

function Ac_mul_B!(y::Matrix{T},A::CauchyLike{T},x::Matrix{T}) where T
    # Computes y=adjoint(A)*x
    dotd= T==ComplexF64 ? BLAS.dotc : BLAS.dot
    n=size(x,1)
    a=[Vector{T}(undef,n) for i=1:Threads.nthreads()]
    Threads.@threads for i=1:n
        tid=Threads.threadid()
        getindex!(a[tid],A,:,i)
        for j=1:size(y,2)
            # ne poziva se - c ili u
            y[i,j]=dotd(a[tid],x[:,j])
        end
    end
end
