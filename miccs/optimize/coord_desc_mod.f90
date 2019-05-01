module coord_desc_mod

contains

    function vecmat(X, A)
        implicit none
        real(8), dimension(:,:), intent(in) :: A
        real(8), dimension(:), intent(in) :: X
        real(8), dimension(size(A,2)) :: vecmat
        
        integer i
        
        vecmat = 0
        do i=1, size(A,1)
            vecmat = vecmat + X(i)* A(i,:)
        end do
    end function vecmat
    
    function matvec(A, X)
        implicit none
        real(8), dimension(:,:), intent(in) :: A
        real(8), dimension(:), intent(in) :: X
        real(8), dimension(size(A,1)) :: matvec
        
        integer i
        
        matvec = 0
        do i=1, size(A,2)
            matvec = matvec + X(i) * A(:,i)
        end do
    end function matvec
    
    function bilinear_same(A, X)
        implicit none
        real(8), dimension(:), intent(in) :: X
        real(8), dimension(:,:), intent(in) :: A
        real(8) :: bilinear_same
        
        integer i, j
        
        bilinear_same = 0
        do i=1, size(X)
            bilinear_same = bilinear_same + (X(i)**2) * A(i,i)
            do j=i+1, size(X)
                bilinear_same = bilinear_same + 2 * X(i) * A(i,j) * X(j)
            end do
        end do
    end function bilinear_same
        
    function bilinear(X, A, Y)
        implicit none
        real(8), dimension(:,:), intent(in) :: A
        real(8), dimension(:), intent(in) :: X
        real(8), dimension(:), intent(in) :: Y
        real(8) :: bilinear
        
        integer i, j
        
        bilinear = 0
        do i=1, size(X)
            do j=1, size(Y)
                bilinear = bilinear + X(i) * A(i,j) * X(j)
            end do
        end do
    end function bilinear
        
    function outer_same(X)
        implicit none
        real(8), dimension(:), intent(in) :: X
        real(8), dimension(size(X), size(X)) :: outer_same
        
        integer i, j
        
        do i=1, size(X)
            outer_same(i,i) = X(i) * X(i)
            do j=i+1, size(X)
                outer_same(i,j) = X(i) * X(j)
                outer_same(j,i) = outer_same(i,j)
            end do
        end do
    end function outer_same
    
    function outer(X, Y)
        implicit none
        real(8), dimension(:), intent(in) :: X
        real(8), dimension(:), intent(in) :: Y
        real(8), dimension(size(X), size(Y)) :: outer
        
        integer i, j
        
        do i=1, size(X)
            do j=1, size(Y)
                outer(i,j) = X(i) * Y(j)
            end do
        end do
    end function outer

    function cov_same(X)
        implicit none
        real(8), dimension(:,:), intent(in) :: X
        real(8), dimension(size(X,1),size(X,1)) :: cov_same
        
        real(8), dimension(size(X,1)) :: meanX
        integer :: i,j
        
        meanX = sum(X, 2)/size(X, 2)
        do i=1, size(X,1)
            cov_same(i,i) = dot_product(X(i,:), X(i,:))
            do j=i+1, size(X,1)
                cov_same(i,j) = dot_product(X(i,:), X(j,:))
                cov_same(j,i) = cov_same(i,j)
            end do
        end do
        cov_same = cov_same/size(X,2) - outer_same(meanX)
    end function cov_same

    function cov(X, Y)
        implicit none
        real(8), dimension(:,:), intent(in) :: X
        real(8), dimension(:,:), intent(in) :: Y
        real(8), dimension(size(X,1), size(Y,1)) :: cov

        real(8), dimension(size(X,1)) :: meanX
        real(8), dimension(size(Y,1)) :: meanY
        integer :: i, j

        if (size(X,2) /= size(Y,2)) then
            stop 'Two datasets have different number of observations!'
        end if

        meanX = sum(X, 2)/size(X,2)
        meanY = sum(Y, 2)/size(Y,2)
        do i=1, size(X,1)
            do j=1, size(Y,1)
                cov(i,j) = dot_product(X(i,:), Y(j,:))
            end do
        end do
        cov = cov/size(X,2) - outer(meanX, meanY)
        !cov = matmul(X, transpose(Y))/size(X,2) &
        !      - matmul(reshape(meanX,(/size(X,1),1/)),reshape(meanY,(/1,size(Y,1)/)))
    end function cov

    function inv(A)
        implicit none
        real(8), dimension(:,:), intent(in) :: A
        real(8), dimension(size(A,1),size(A,2)) :: inv

        real(8), dimension(size(A,1)) :: work  ! work array for LAPACK
        integer, dimension(size(A,1)) :: ipiv   ! pivot indices
        integer :: n, info
        
        ! External procedures defined in LAPACK
        external DGETRF
        external DGETRI
        
        ! Store A in Ainv to prevent it from being overwritten by LAPACK
        inv = A
        n = size(A,1)

        ! DGETRF computes an LU factorization of a general M-by-N matrix A
        ! using partial pivoting with row interchanges.
        call DGETRF(n, n, inv, n, ipiv, info)

        if (info /= 0) then
         stop 'Matrix is numerically singular!'
        end if

        ! DGETRI computes the inverse of a matrix using the LU factorization
        ! computed by DGETRF.
        call DGETRI(n, inv, n, ipiv, work, n, info)

        if (info /= 0) then
         stop 'Matrix inversion failed!'
        end if
    end function inv
    
    function det(A)
        implicit none
        real(8), dimension(:,:), intent(in) :: A
        real(8), dimension(size(A,1), size(A,2)) :: copy
        integer, dimension(size(A,1)) :: ipiv   ! pivot indices
        real(8) :: det
        integer :: i, n, info
        
        external DGETRF
        
        copy = A
        n = size(A,1)
        
        call DGETRF(n, n, copy, n, ipiv, info)
        
        det = 1
        do i=1,n
            if (ipiv(i) /= i) then
                det = -det
            end if
            det = det*copy(i,i)
        end do
    end function det
    
    function tr(X)
        implicit none
        real(8), dimension(:,:) :: X
        real(8) :: tr
        integer :: i
        
        if (size(X,1) /= size(X,2)) then
            stop 'Matrix is not square!'
        end if
        
        tr=0
        do i=1, size(X,1)
            tr = tr + X(i,i)
        end do        
    end function tr

    function sgn(x)
        implicit none
        real(8) :: x
        real(8) :: sgn
        
        sgn=0.0
        if (x>0) then
            sgn=1.0
            return
        end if
        if (x<0) then
            sgn=-1.0
            return
        end if
    end function sgn
    
end module coord_desc_mod