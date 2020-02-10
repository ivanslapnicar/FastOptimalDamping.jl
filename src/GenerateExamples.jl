function GenerateExamples(example::Char)
    # Generate damping examples of dimension n=801, 1601, 2001
    # with example=='S','L'or 'H', for Small, Large and Homogeneous, respectively.

    if example=='S'
        n₁=400
    elseif example=='L'
        n₁=800
    else
        n₁=1000
    end

    # Dimension of the system
    n=2*n₁+1
    m₁=1
    m₂=1

    if example=='S'
        m₀=1200.0
    elseif example=='L'
        m₀=1800.0
    else
        m₀=2000.0
    end


    k₁=100
    k₂=150
    k₃=200

    first_part=Array{Int,1}(undef,n₁)
    second_part=Array{Int,1}(undef,n₁)

    if example=='S'
        for ii=1:400
            first_part[ii]=1000-4*ii
        end
        for ii=201:400
            first_part[ii]=3*ii-400
        end
        for ii=1:400
            second_part[ii]=500+ii
        end
    elseif example=='L'
        for ii=1:div(n₁,2)
            first_part[ii]=2000-4*ii
        end
        for ii=div(n₁,2)+1:n₁
            first_part[ii]=3*ii-800
        end
        for ii=1:n₁
            second_part[ii]=500+ii
        end
    else
        for ii=1:n₁
           first_part[ii]=1000
        end
        for ii=1:n₁
          second_part[ii]=1500
        end
    end

    M=Diagonal([first_part; second_part; m₀])

    K₁=SymTridiagonal(2*ones(n₁),-ones(n₁-1))
    α=0.02
    K₁=Matrix(K₁)

    K=zeros(2*n₁+1,2*n₁+1)
    K[1:n₁,1:n₁]=k₁*K₁
    K[n₁+1:2*n₁,n₁+1:2*n₁]=k₂*K₁

    K[1:n₁,end]=-[zeros(1,n₁-1) k₁]'
    K[n₁+1:2*n₁,end]=-[zeros(1,n₁-1) k₂]'

    K[end,1:n₁]=-[zeros(1,n₁-1) k₁]
    K[end,n₁+1:2*n₁]=-[zeros(1,n₁-1) k₂]
    K[2*n₁+1, 2*n₁+1]=k₁+k₂+k₃

    # Return stiffness and mass matrix
    return K, M
end
