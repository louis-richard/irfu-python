#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
.py

@author : Louis RICHARD
"""

def generic_minimum_residue_analysis_engine(eta=None, q=None, constraint_vector=None):
    """
    Calculates L, V and U given the density of conserved quantity eta and transport tensor q
    using constraint that discontinuity normal is perpendicular to the constraint vector.

    Implements general Generic Minimum Residue Analysis (GMRA) solution based on [8]_, [9]_

    Parameters
    ----------
    eta : xarray.DataArray
        Density of conserved quantity Mx3, M is number of points

    q : xarray.DataArray
        Transport tensor

    constraint_vector : list or numpy.ndarray
        Constraint vector

    Returns
    -------
    l :
        Eigenvalues

    v :
        Eigenvectors corresponding to lmin, lmean, lmax

    u :
        Transport velocity, boundary velocity is Un = U * V(:, 1)

    References
    ----------
    .. [8]  Sonnerup, B. U. Ö., S. Haaland, G. Paschmann, M. W. Dunlop, H. Rème, and A. Balogh (
            2006), Orientation and motion ofa plasma discontinuity from single-spacecraft
            measurements : Generic residue analysis of Cluster data, J. Geophys. Res., 111,
            A05203,doi : https://doi.org/10.1029/2005JA011538 .

    .. [9]  Sonnerup, B. U. Ö., S. Haaland, G. Paschmann, M. W. Dunlop, H. Rème, and A. Balogh (
            2007), Correction to ‘Orientationand motion of a plasma discontinuity from
            single-spacecraft measurements : Generic residue analysis of Cluster data´,
            J. Geophys.Res., 11 2 , A04201, doi : hhtps://doi.org/10.1029/2007JA012288 .

    """

    # Defaults
    do_calculate_velocity, do_calculate_q, have_constraint = [True, True, False]

    # Input check

    if constraint_vector is not None:
        have_constraint = True

    # Check inputs
    if eta.ndim == 1:
        eta = np.tile(eta, (len(q), 1))

    # U estimate. U = < deta dq > / < | deta | ^ 2 >
    d_eta = eta - np.namean(eta, axis=0)  # Eq. 10
    d_q = q - np.namean(q, axis=0)
    #d_eta_d_q_average = np.nanmean(matrix_dot(deta, 1, dq, 1), 1)
    #d_eta2_average = np.nanmean(np.dot(deta, deta, 2), 1)
    u = d_eta_d_q_average / d_eta2_averag  # Eq. 12

    # Q estimate
    # dq_dq_average = shiftdim(np.nanmean(matrix_dot(dq, 1, dq, 1), 1), 1)
    # d_eta_d_q_average2_mat = detadqAver.T * detadqAver
    if eta == 0:
        # Eq.19
        q_mat = dq_dq_average
    else:
        # Eq.15b (see correction in Sonnerup 2007)
        q_mat = dq_dq_average - d_eta_dq_aver2_mat / d_eta2_average

    # Correct Q by the number of dimensions so that eigenvalues coorespond to the variance of dq
    q_mat = q_mat / d_q.shape[1]


    """
    if q is not None:
        
    
    if nargin == 0 & & nargout == 0
        help irf_generic_minimum_residue_analysis_engine
        return;
    elseif nargin == 1 && isstruct(varargin{1})
        InputParameters = varargin{1};
        inputParameterFields = fieldnames(InputParameters);
        for j in range(len(input_parameter_field)):
            fieldname = input_parameter_fields{j};
            eval([fieldname ' = InputParameters.' fieldname ';']);
        end
    elseif nargin > 1
    
    
        args = varargin;
        while numel(args) >= 2
            switch args{1}
                case 'Q'
                    Q = args{2};
                    doCalculateVelocity = false;
                    doCalculateQ = false;
                case 'eta'
                    eta = args{2};
                case 'q'
                    q = args{2};
                case 'constraint'
                    constraintVector = args{2};
                    haveConstraint = true;
                otherwise
                    irf.log('critical', 'unrecognized input');
                    return
    
            args(1: 2)=[];
    
    
    # Check inputs
    if numel(eta) == 1 % scalar
        eta = repmat(eta, size(q, 1), 1);
    end
    
    # Calculate Q from eta and q
        if doCalculateQ
            # U estimate. U = < deta dq > / < | deta | ^ 2 > deta = bsxfun( @ minus, eta, irf.nanmean(eta, 1)); % Eq. 10
            dq = bsxfun( @ minus, q, irf.nanmean(q, 1))
            deta_dq_aver = np.nanmean(matrix_dot(deta, 1, dq, 1), 1)
            deta2_aver = np.nanmean(np.dot(deta, deta, 2), 1)
            U = detadqAver / deta2Aver  # Eq. 12
    
            # Q estimate
            dqdqAver = shiftdim(np.nanmean(matrix_dot(dq, 1, dq, 1), 1), 1)
            detadqAver2Matrix = detadqAver.T * detadqAver
            if eta == 0:
                Q = dq_dq_aver # Eq.19
            else:
                Q = dq_dq_aver - deta_dq_aver2_matrix / deta2_aver # Eq.15b (see correction in Sonnerup 2007)
    
    
            # Correct Q by the number of dimensions so that eigenvalues coorespond to the variance of dq
            Q = Q / size(dq, 2)
    
        # Check for constraints
            if haveConstraint
                P = eye(numel(constraintVector)) - constraintVector(:) * constraintVector(:)'; % Eq 41
                Q = P * Q * P
    
    
        # Calculate eigenvalues and eigenvectors from Q
        [V, lArray] = eig(Q)            # L is diagonal matrix of eigenvalues and V matrix with columns eigenvectors
        [L, I] = sort(diag(lArray))
        V = V(:, I)
    
        if ~doCalculateVelocity
            U = NaN
            return
        end
    
        # Calculate normal velocity
    
        Un = dot(U, V(:, 1))
    
        # Print output
        if nargout == 0
            disp(['Eigenvalues: ' sprintf('%7.5f ', L)]);
            disp(vector_disp('N', V(:, 1)));
            disp(vector_disp('M', V(:, 2)));
            disp(vector_disp('L', V(:, 3)));
            disp(vector_disp('U', U, 'km/s'));
            disp(['Un = ' num2str(Un, 3), ' km/s']);
        end
    
        % % Define
        output
        if nargout == 0
            clear
            L
            V
            U;
        end
    
        % % Functions
        function out = vector_disp(vectSymbol, vect, vectUnit)
        if nargin == 2, vectUnit ='';end
        out = sprintf(['|' vectSymbol '| = %8.4f ' vectUnit...
                       ', ' vectSymbol ' = [ %8.4f %8.4f %8.4f ] ' vectUnit], ...
        norm(vect), vect(1), vect(2), vect(3));
    
    function out = matrix_dot(inp1, ind1, inp2, ind2)
    % MATRIX_DOT summation over one index multiplication
    %
    % MATRIX_DOT(inp1, ind1, inp2, ind2)
    % inp1, inp2 are the matrixes and summation is over dimensions(ind1 + 1) and
    % (ind2 + 1). + 1 because first dimension is always time.
    szinp1 = size(inp1);
    ndimsInp1 = ndims(inp1) - 1;
    szinp2 = size(inp2);
    ndimsInp2 = ndims(inp2) - 1;
    szout1 = szinp1;
    szout1(ind1 + 1) = [];
    szout2 = szinp2;
    szout2([1 ind2 + 1]) = [];
    szout = [szout1 szout2];
    out = zeros(szout);
    
    if ndimsInp1 == 1
        if ndimsInp2 == 1
            out = sum(inp1. * inp2, 2);
        elseif
            ndimsInp2 == 2 & & ind2 == 1
            for jj = 1:szinp2(3)
                for kk = 1:szinp1(2)
                    out(:, jj) = out(:, jj) + inp1(:, kk).*inp2(:, kk, jj);
                end
            end
        elseif
            ndimsInp2 == 2 & & ind2 == 2
            for jj = 1:szinp2(2)
                for kk = 1:szinp1(2)
                    out(:, jj) = out(:, jj) + inp1(:, kk).*inp2(:, jj, kk);
                end
            end
        else
            error('Not yet implemented'); % not implemented
        end
    elseif
        ndimsInp1 == 2 & & ndimsInp2 == 1 & & ind1 == 1
        for jj = 1:szinp1(2)
            for kk = 1:szinp2(2)
                out(:, jj) = out(:, jj) + inp1(:, kk, jj).*inp2(:, kk);
            end
        end
    elseif
        ndimsInp1 == 2 & & ndimsInp2 == 1 & & ind1 == 2
        for jj = 1:szinp1(2)
            for kk = 1:szinp2(2)
                out(:, jj) = out(:, jj) + inp1(:, jj, kk).*inp2(:, kk);
            end
        end
    elseif
        ndimsInp1 == 2 & & ndimsInp2 == 2 & & ind1 == 1 & & ind2 == 1
        for jj = 1:szinp1(3)
            for kk = 1:szinp2(3)
                for ss = 1:szinp1(ind1 + 1)
                    out(:, jj, kk) = out(:, jj, kk) + inp1(:, ss, jj).*inp2(:, ss, kk);
                end
            end
        end
    else
        error('Not yet implemented'); % not implemented
    end
    """