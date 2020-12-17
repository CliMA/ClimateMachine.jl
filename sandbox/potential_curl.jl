function grad(Q::MPIStateArray, _1)
    Q_x1 = view(Q, (:,:,_1))

    v_∇Q_x1 = MPIStateArray{FT}(mpicomm, ArrayType, ijksize, nrealelem, 3)
    s_∇Q_x1 = MPIStateArray{FT}(mpicomm, ArrayType, ijksize, nrealelem, 3)

    event = launch_volume_gradient!(grid, v_∇Q, Q_x1)
    wait(event)

    event = launch_interface_gradient!(grid, s_∇Q, Q_x1)
    wait(event)

    grad = v_∇Q_x1 + s_∇Q_x1

    return grad
end

function grad(Q::MPIStateArray, _1, _2, _3)
    Q_x1 = view(Q, (:,:,_1))
    Q_x2 = view(Q, (:,:,_2))
    Q_x3 = view(Q, (:,:,_3))

    v_∇Q_x1 = MPIStateArray{FT}(mpicomm, ArrayType, ijksize, nrealelem, 3)
    v_∇Q_x2 = MPIStateArray{FT}(mpicomm, ArrayType, ijksize, nrealelem, 3)
    v_∇Q_x3 = MPIStateArray{FT}(mpicomm, ArrayType, ijksize, nrealelem, 3)

    s_∇Q_x1 = MPIStateArray{FT}(mpicomm, ArrayType, ijksize, nrealelem, 3)
    s_∇Q_x2 = MPIStateArray{FT}(mpicomm, ArrayType, ijksize, nrealelem, 3)
    s_∇Q_x3 = MPIStateArray{FT}(mpicomm, ArrayType, ijksize, nrealelem, 3)

    ## strong form gradients here
    event_x1 = launch_volume_gradient!(grid, v_∇Q_x1, Q_x1)
    event_x2 = launch_volume_gradient!(grid, v_∇Q_x2, Q_x2)
    event_x3 = launch_volume_gradient!(grid, v_∇Q_x3, Q_x3)
    wait(event_x1)
    wait(event_x2)
    wait(event_x3)

    ## strong form gradients here
    event_x1 = launch_interface_gradient!(grid, s_∇Q_x1, Q_x1)
    event_x2 = launch_interface_gradient!(grid, s_∇Q_x2, Q_x2)
    event_x3 = launch_interface_gradient!(grid, s_∇Q_x3, Q_x3)
    wait(event_x1)
    wait(event_x2)
    wait(event_x3)

    ∇Q_x1 = v_∇Q_x1 + s_∇Q_x1
    ∇Q_x2 = v_∇Q_x2 + s_∇Q_x2
    ∇Q_x3 = v_∇Q_x3 + s_∇Q_x3

    return (; ∇Q_x1, ∇Q_x2, ∇Q_x3)
end

function div(Q::MPIStateArray, _1, _2, _3)
    ∇Q_x1, ∇Q_x2, ∇Q_x3 = grad(Q, _1, _2, _3)

    div = MPIStateArray{FT}(mpicomm, ArrayType, ijksize, nrealelem, 1)
    div[:,:,1] = ∇Q_x1[:,:,1] + ∇Q_x2[:,:,2] + ∇Q_x3[:,:,3]

    return div
end

function curl(Q::MPIStateArray, _1, _2, _3)
    ∇Q_x1, ∇Q_x2, ∇Q_x3 = grad(Q, _1, _2, _3)

    curl = MPIStateArray{FT}(mpicomm, ArrayType, ijksize, nrealelem, 3)

    curl[:,:,1] = ∇Q_x3[:,:,2] - ∇Q_x2[:,:,3]
    curl[:,:,2] = ∇Q_x1[:,:,3] - ∇Q_x3[:,:,1]
    curl[:,:,3] = ∇Q_x2[:,:,1] - ∇Q_x1[:,:,2]

    return curl
end

