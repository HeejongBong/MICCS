!file: coord_desc_init.f90
subroutine coord_desc_init(prec, corr, latent, weight, & !outputs
                           observation, dims_pop, & !inputs :: data
                           lambda_ridge, & !inputs :: parameters
                           num_pop, dim_obs, num_trial) !inputs :: dimensions
                           
    implicit none
    
    !inputs :: dimensions
    integer, intent(in) :: num_pop, dim_obs, num_trial
    
    !inputs :: data
    real(8), dimension(dim_obs, num_trial), intent(in) :: observation
    integer, dimension(num_pop), intent(in) :: dims_pop
    
    !inputs :: parameters
    real(8), dimension(num_pop), intent(in) :: lambda_ridge
    
    !outputs
    real(8), dimension(num_pop, num_pop), intent(out) :: prec, corr
    real(8), dimension(num_pop, num_trial), intent(out) :: latent
    real(8), dimension(dim_obs), intent(out) :: weight

    !local variables
    integer :: i, j, info, ind_from, ind_to
    integer, dimension(num_pop) :: ipiv
    real(8), dimension(num_pop) :: latent_mean, std, work
    
    external DGETRF
    external DGETRI
    
    if (sum(dims_pop) /= dim_obs) then
        stop 'Dimensions of population does not match with observation'
    end if    
    
    !initialie correlation
    weight = 1
    do i = 1, num_pop
        ind_from = sum(dims_pop(:i-1)) + 1
        ind_to = sum(dims_pop(:i))
        
        latent(i,:) = sum(spread(weight(ind_from:ind_to), 2, num_trial) * &
                          observation(ind_from:ind_to,:), 1)
        latent_mean(i) = sum(latent(i,:)) / num_trial
        
        std(i) = sqrt(dot_product(latent(i,:), latent(i,:)) / num_trial - &
                      latent_mean(i) * latent_mean(i) + lambda_ridge(i))
        weight(ind_from:ind_to) = weight(ind_from:ind_to) / std(i)
        latent(i,:) = latent(i,:) / std(i)
        latent_mean(i) = latent_mean(i) / std(i)
        
        corr(i,i) = 1
        do j = 1, i-1
            corr(i,j) = dot_product(latent(i,:), latent(j,:)) / num_trial - &
                        latent_mean(i) * latent_mean(j)
            corr(j,i) = corr(i,j)
        end do
    end do
   
    !initialize precision    
    prec = corr
    
    call DGETRF(num_pop, num_pop, prec, num_pop, ipiv, info)
    
    if (info /= 0) then
        stop 'Initial correlation is numerically singular!'
    end if
    
    call DGETRI(num_pop, prec, num_pop, ipiv, work, num_pop, info)
    
    if (info /= 0) then
        stop 'Initial correlation inversion failed!'
    end if
    
end subroutine coord_desc_init