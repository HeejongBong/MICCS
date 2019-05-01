!file: coord_desc_iter_p.f90
subroutine coord_desc_iter_p(prec, corr, latent, weight_p, & !output
                             observation_p, & !inputs :: data
                             lambda_ridge_p, lambda_glasso_p, & !inputs :: parameters
                             p, pop_dep, pop_indep, &
                             dim_pop, num_pop_dep, num_pop_indep, & !inputs :: p-wise
                             num_pop, num_trial) !inputs :: dimension
    implicit none
    
    ! inputs
    integer, intent(in) :: p, dim_pop, num_pop, num_trial
    integer, intent(in) :: num_pop_dep, num_pop_indep
    integer, dimension(num_pop_dep), intent(in) :: pop_dep
    integer, dimension(num_pop_indep), intent(in) :: pop_indep
    real(8) :: lambda_ridge_p
    real(8), dimension(num_pop_dep), intent(in) :: lambda_glasso_p
    real(8), dimension(dim_pop, num_trial), intent(in) :: observation_p

    ! in-outputs    
    real(8), dimension(num_pop, num_pop) :: prec, corr
    real(8), dimension(num_pop, num_trial) :: latent
    real(8), dimension(dim_pop) :: weight_p

!f2py intent(inplace) prec
!f2py intent(inplace) corr
!f2py intent(inplace) latent
!f2py intent(inplace) weightp

    ! local variables
    integer :: i, info
    integer, dimension(num_pop_indep) :: ipiv_indep
    real(8), dimension(num_pop_indep) :: work_indep
    real(8), dimension(num_pop_indep, num_pop_indep) :: cinv_prec_indep
    
    integer, dimension(num_pop_dep) :: ipiv_dep
    real(8), dimension(num_pop_dep) :: work_dep
    real(8), dimension(num_pop_dep, num_pop_dep) :: inv_prec_dep
    
    real(8), dimension(num_pop_dep) :: prec_p_dep, corr_p_dep
    
    integer, dimension(dim_pop) :: ipiv_p
    real(8), dimension(dim_pop) :: obs_p_mean, work_p, weight_p_old
    real(8), dimension(dim_pop, dim_pop) :: cov_p, inv_cov_p
    real(8), dimension(num_pop_dep) :: latent_dep_mean
    real(8), dimension(dim_pop, num_pop_dep) :: cov_p_dep
    
    integer, dimension(num_pop-1) :: pop_notp
    real(8) :: latent_p_mean
    real(8), dimension(num_pop-1) :: latent_notp_mean
    
    external DGETRF
    external DGETRI
    
    ! update precision matrix
    if (num_pop_dep == 0) then
        prec(p,p) = 1
        prec(p,pop_indep) = 0
        prec(pop_indep,p) = 0
    else
        if (num_pop_indep == 0) then
            inv_prec_dep = prec(pop_dep, pop_dep)
        else
            cinv_prec_indep = prec(pop_indep, pop_indep)

            call DGETRF(num_pop_indep, num_pop_indep, cinv_prec_indep, num_pop_indep, &
                        ipiv_indep, info)

            if (info /= 0) then
                stop 'Conditional precision is numerically singular!'
            end if

            call DGETRI(num_pop_indep, cinv_prec_indep, num_pop_indep, ipiv_indep, &
                        work_indep, num_pop_indep, info)

            if (info /= 0) then
                stop 'Conditional precision inversion failed!'
            end if

            inv_prec_dep = prec(pop_dep, pop_dep) - &
            matmul(prec(pop_dep, pop_indep), &
                   matmul(cinv_prec_indep, prec(pop_indep, pop_dep)))
        end if

        call DGETRF(num_pop_dep, num_pop_dep, inv_prec_dep, num_pop_dep, &
                    ipiv_dep, info)

        if (info /= 0) then
            stop 'Precision is numerically singular!'
        end if

        call DGETRI(num_pop_dep, inv_prec_dep, num_pop_dep, ipiv_dep, work_dep, &
                    num_pop_dep, info)

        if (info /= 0) then
            stop 'Precision inversion failed!'
        end if

        prec_p_dep = prec(p, pop_dep)
        corr_p_dep = corr(p, pop_dep)

        do i = 1, num_pop_dep
            prec_p_dep(i) = - corr_p_dep(i) - &
                dot_product(inv_prec_dep(i,:i-1), prec_p_dep(:i-1)) - &
                dot_product(inv_prec_dep(i,i+1:), prec_p_dep(i+1:))        
            prec_p_dep(i) = sign(max(abs(prec_p_dep(i))-lambda_glasso_p(i), 0.0) / &
                                 inv_prec_dep(i,i), &
                                 prec_p_dep(i))
        end do   

        prec(p,p) = 1 + dot_product(prec_p_dep, &
                        sum(inv_prec_dep * spread(prec_p_dep, 1, num_pop_dep), dim = 2))
        prec(p,pop_dep) = prec_p_dep
        prec(pop_dep,p) = prec_p_dep
        prec(p,pop_indep) = 0
        prec(pop_indep,p) = 0       
    end if
    
    ! update weight
    weight_p_old = weight_p
    
    obs_p_mean = sum(observation_p, dim = 2) / num_trial
    cov_p = matmul(observation_p, transpose(observation_p)) / num_trial - &
            matmul(reshape(obs_p_mean, (/dim_pop,1/)), reshape(obs_p_mean, (/1,dim_pop/)))
    do i = 1, dim_pop
        cov_p(i,i) = cov_p(i,i) + lambda_ridge_p
    end do
    
    latent_dep_mean = sum(latent(pop_dep,:), dim = 2) / num_trial
    cov_p_dep = matmul(observation_p, transpose(latent(pop_dep,:))) / num_trial - &
    matmul(reshape(obs_p_mean, (/dim_pop,1/)), reshape(latent_dep_mean, (/1,num_pop_dep/)))
    
    inv_cov_p = cov_p
    
    call DGETRF(dim_pop, dim_pop, inv_cov_p, dim_pop, ipiv_p, info)
                
    if (info /= 0) then
        stop 'Covariance is numerically singular!'
    end if
    
    call DGETRI(dim_pop, inv_cov_p, dim_pop, ipiv_p, work_p, dim_pop, info)
    
    if (info /= 0) then
        stop 'Covariance inversion failed!'
    end if
    
    weight_p = reshape(matmul(inv_cov_p, &
        matmul(cov_p_dep, reshape(prec_p_dep ,(/num_pop_dep,1/)))),(/dim_pop/))
    
    if (dot_product(weight_p, weight_p) == 0) then
        weight_p = weight_p_old
    else
        weight_p = - weight_p / sqrt( &
        dot_product(weight_p, sum(cov_p * spread(weight_p, 1, dim_pop), dim = 2)))
    end if    
    
    ! update latent
    latent(p,:) = sum(observation_p * spread(weight_p, 2, num_trial), dim = 1)
                               
    ! update correlation
    latent_p_mean = sum(latent(p,:)) / num_trial
    pop_notp = pack((/(i,i=1,num_pop)/), (/(i,i=1,num_pop)/) /= p)
    latent_notp_mean = sum(latent(pop_notp,:), dim=2) / num_trial
    
    corr(p,pop_notp) = sum(latent(pop_notp,:) * &
        spread(latent(p,:), 1, num_pop-1), dim = 2) / num_trial - &
        latent_p_mean * latent_notp_mean
    corr(pop_notp,p) = corr(p,pop_notp)
    
end subroutine coord_desc_iter_p