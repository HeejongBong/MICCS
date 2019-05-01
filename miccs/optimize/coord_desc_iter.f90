!file: coord_desc_iter.f90
subroutine coord_desc_iter(prec, corr, latent, weight, & !inplace 
                           observation, dims_pop, & !inputs :: data
                           lambda_ridge, lambda_glasso, & !inputs :: parameters
                           num_pop, dim_obs, num_trial) !inputs :: dimensions
    implicit none
    
    !inputs :: dimensions
    integer, intent(in) :: num_pop, dim_obs, num_trial
    
    !inputs :: data
    real(8), dimension(dim_obs, num_trial), intent(in) :: observation
    integer, dimension(num_pop), intent(in) :: dims_pop
    
    !inputs :: parameters
    real(8), dimension(num_pop, num_pop), intent(in) :: lambda_glasso
    real(8), dimension(num_pop), intent(in) :: lambda_ridge
    
    ! in/outputs
    real(8), dimension(num_pop, num_pop) :: prec, corr
    real(8), dimension(num_pop, num_trial) :: latent
    real(8), dimension(dim_obs) :: weight

!f2py intent(inplace) prec
!f2py intent(inplace) corr
!f2py intent(inplace) latent
!f2py intent(inplace) weight

    ! local variables
    integer :: i, j, k, ind_from, ind_to
    integer :: num_pop_dep, num_pop_indep


    external DGETRF
    !external coord_desc_iter_p
    
    ! iterate on populations
    do i = 1, num_pop
        ind_from = sum(dims_pop(:i-1)) + 1
        ind_to = sum(dims_pop(:i))
        
        num_pop_dep = count((lambda_glasso(i,:) >= 0) .and. &
                            ((/(k,k=1,num_pop)/) /= i))
        num_pop_indep = count((lambda_glasso(i,:) < 0) .and. &
                              ((/(k,k=1,num_pop)/) /= i))
        
        call coord_desc_iter_p(prec, corr, latent, weight(ind_from:ind_to), &
            observation(ind_from:ind_to,:), lambda_ridge(i), &
            pack(lambda_glasso(i,:), (lambda_glasso(i,:) >= 0) .and. &
                                     ((/(k,k=1,num_pop)/) /= i)), &
            i, pack((/(j,j=1,num_pop)/), (lambda_glasso(i,:) >= 0) .and. &
                                         ((/(k,k=1,num_pop)/) /= i)), &
            pack((/(j,j=1,num_pop)/), (lambda_glasso(i,:) < 0) .and. &
                                      ((/(k,k=1,num_pop)/) /= i)), &
            dims_pop(i), num_pop_dep, num_pop_indep, num_pop, num_trial)
    end do
           
end subroutine coord_desc_iter