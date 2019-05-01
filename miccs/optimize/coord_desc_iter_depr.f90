!file: coord_desc_iter.f90
subroutine coord_desc_iter(prec, corr, latent, weight, & !inplace 
                           observation, dims_pop, & !inputs :: data
                           lambda_ridge, lambda_glasso, & !inputs :: parameters
                           num_pop, dim_obs, num_trial) !inputs :: dimensions
    use coord_desc_mod
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
    integer :: nDp, nIp, dimp, p, i, j, ind_from, ind_to
    integer, allocatable, dimension(:) :: Dp, Ip
    real(8), allocatable, dimension(:,:) :: RDp, Sp, SDp, obsp
    real(8), allocatable, dimension(:) :: Wp, Rp, weightp
        
    ! iterate on populations
    do p=1, num_pop
        nDp = count((lambda_glasso(p,:) >= 0) .and. ((/(j,j=1,num_pop)/) /= p))
        nIp = count((lambda_glasso(p,:) < 0) .and. ((/(j,j=1,num_pop)/) /= p))
        dimp = dims_pop(p)
        ind_from = sum(dims_pop(:p-1))+1
        ind_to = sum(dims_pop(:p))
        
        allocate(Dp(nDp), Ip(nIp), RDp(nDp,nDp), Wp(nDp), Rp(nDp), &
                 Sp(dimp,dimp), SDp(dimp,nDp), weightp(dimp), obsp(dimp,num_trial))
                 
        Dp = pack((/(i,i=1,num_pop)/), &
                  (lambda_glasso(p,:) >= 0) .and. ((/(j,j=1,num_pop)/) /= p))
        Ip = pack((/(i,i=1,num_pop)/), &
                  (lambda_glasso(p,:) < 0) .and. ((/(j,j=1,num_pop)/) /= p))
        obsp = observation(ind_from:ind_to,:)
        
        ! update precision matrix
        if (nDp == 0) then
            prec(p,p) = 1
            prec(p,Ip) = 0
            prec(Ip,p) = 0 
        else
            if (nIp == 0) then
                RDp = inv(prec(Dp, Dp))
            else
                RDp = inv(prec(Dp, Dp) - &
                      matmul(prec(Dp,Ip), matmul(inv(prec(Ip,Ip)), prec(Ip,Dp))))
            end if
            
            Wp = prec(p, Dp)
            Rp = corr(p, Dp)

            do i = 1, nDp
                Wp(i) = - Rp(i) - dot_product(RDp(i,:i-1), Wp(:i-1)) &
                        - dot_product(RDp(i,i+1:), Wp(i+1:))        
                Wp(i) = sgn(Wp(i)) * &
                    max(abs(Wp(i))-lambda_glasso(p, Dp(i)), 0.0) / RDp(i,i)
            end do   

            prec(p,p) = 1 + bilinear_same(RDp, Wp)
            prec(p,Dp) = Wp
            prec(Dp,p) = Wp
            prec(p,Ip) = 0
            prec(Ip,p) = 0 
        end if
        
        ! update weight
        Sp = cov_same(obsp)
        do i = 1, dimp
            Sp(i,i) = Sp(i,i) + lambda_ridge(p)
        end do
        SDp = cov(obsp, latent(Dp,:))

        weightp = matvec(inv(Sp), matvec(SDp, Wp))
        
        if (dot_product(weightp, weightp) /= 0) then
            weightp = - weightp / sqrt(bilinear_same(Sp, weightp))
        end if
        weight(ind_from:ind_to) = weightp

        ! update latent
        latent(p,:) = vecmat(weightp, obsp)
        
        ! update correlation
        corr(p:p,:p-1) = cov(latent(p:p,:), latent(:p-1,:))
        corr(p:p,p+1:) = cov(latent(p:p,:), latent(p+1:,:))
        corr(:p-1,p) = corr(p,:p-1)
        corr(p+1:,p) = corr(p,p+1:)
        
        deallocate(Dp, Ip, RDp, Wp, Rp, Sp, SDp, weightp, obsp)
    end do
           
end subroutine coord_desc_iter