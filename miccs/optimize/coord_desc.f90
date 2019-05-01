!file: coord_desc.f90
subroutine coord_desc(converged, prec, corr, latent, weight, & !outputs
                      observation, dims_pop, & !inputs :: data
                      lambda_ridge, lambda_glasso, & !inputs :: parameters
                      ths, max_iter, & !inputs, optional :: opt. parameters
                      verbose, verb_period, & !inputs, optional :: etc.
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
    
    !inputs, optional :: opt.parameters
    integer, intent(in) :: max_iter
    real(8), intent(in) :: ths

!f2py integer optional, intent(in) :: max_iter = 10000
!f2py real(8) optional, intent(in) :: ths = 0.0001

    !inputs, optional :: etc.
    logical, intent(in) :: verbose
    integer, intent(in) :: verb_period

!f2py integer optional, intent(in) :: verbose = 0
!f2py integer optional, intent(in) :: verb_period = 10
    
    !outputs
    logical, intent(out) :: converged
    real(8), dimension(num_pop, num_pop), intent(out) :: prec, corr
    real(8), dimension(num_pop, num_trial), intent(out) :: latent
    real(8), dimension(dim_obs), intent(out) :: weight

    !local variables
    integer :: i, info
    real(8) :: cost, cost_old, diff
    real(8), dimension(num_pop) :: w
    real(8), dimension(num_pop, num_pop) :: prec_copy
    real(8), dimension(max(1, 3*num_pop-1)) :: work_eigen

    external DSYEV
    
    if (sum(dims_pop) /= dim_obs) then
        stop 'Dimensions of population does not match with observation'
    end if
    
    call coord_desc_init(prec, corr, latent, weight, & !outputs
                         observation, dims_pop, & !inputs :: data
                         lambda_ridge, & !inputs :: parameters
                         num_pop, dim_obs, num_trial)
    
    !initial setting
    cost_old = huge(1.)
    converged = .false.
    
    !iteration   
    do i = 1, max_iter
        call coord_desc_iter(prec, corr, latent, weight, &
                             observation, dims_pop, lambda_ridge, lambda_glasso, &
                             num_pop, dim_obs, num_trial)
                             
        ! calculate cost
        prec_copy = prec

        call DSYEV('N', 'U', num_pop, prec_copy, num_pop, w, work_eigen, &
                    max(3*num_pop-1, 1), info)
        
        if ( w(1) <= 0 .or. product(w) <= 0 ) then
            if (verbose .and. mod(i, verb_period) == 0) then
                print *, 'Iteration ', i, ', Precision is not positive definite!'
            end if
            cost_old = huge(1.)
        else
            cost = - log(product(w)) + sum(prec * corr) + &
                   sum(max(lambda_glasso, 0.0) * abs(prec))
            diff = cost_old - cost

            if (verbose .and. mod(i, verb_period) == 0) then 
                print *, 'Iteration ', i, ', cost: ', cost, ', difference: ', diff
            end if

            if (abs(diff) < ths) then
                converged = .true.
                exit
            end if
        
            cost_old = cost
        end if
    end do
    
end subroutine coord_desc
        
        