function [mt_func , hp_solver ] = mtclf_SRMTL(solver)
    mt_func = @mtclf_func;
    hp_solver = get_hyperparams_solver(solver);
end

function [ AUC, ACC ] = mtclf_func( x_train, y_train, x_test, y_test, pars )
%% R encodes structure relationship
%   1)Structure order is given by using [1 -1 0 ...; 0 1 -1 ...; ...]
%    e.g.: R=zeros(t,t-1);R(1:(t+1):end)=1;R(2:(t+1):end)=-1;
%   2)Ridge penalty term by setting: R = eye(t)
%   3)All related regularized: R = eye (t) - ones (t) / t

    % TODO: hyperparameter search
    t = size(x_train, 2); 
    R = eye(t);

    [y_train, y_test ] = convert2sign(y_train, y_test);
    [W, C, loss_values]=Logistic_SRMTL(x_train, y_train, R, pars.rho1, pars.rho2, pars);
    [AUC, ACC]=mtclf_test_sign(x_test, W, C, y_test, pars); 
end

function [hp_solver] = get_hyperparams_solver(solver)
    if strcmp(solver, 'grid_small') == 1
        hp_solver = make_solver('grid search',...
            'rho1', logspace(-3,1,2), ... 
            'rho2',logspace(-3,1,2)%, ...
            %'init', 0:2
        );  
    elseif strcmp(solver, 'grid_large') == 1
        hp_solver = make_solver('grid search',...
            'rho1', logspace(-3,3,5), ...  
            'rho2',logspace(-3,3,5) %, ...
            % 'init', 0:2, ...
            % 'tFlag', 0:3
        );  
    elseif strcmp(solver, 'swarm') == 1
        hp_solver = make_solver('particle swarm', 
            'num_particles', 2, ...
            'num_generations', 4, ...
            'max_speed', 0.1, ...
            'rho1', logspace(-3,3,2), ...
            'rho2',logspace(-3,3,2)
            % 'init', [0,2], ...
            % 'tFlag', [0,3], ...
            % 'tol', logspace(-5,-4,2), ...
            % 'maxIter', logspace(1,4, 2)
        );
    elseif strcmp(solver, 'json') == 1   
        hp_solver = solver ;
    else
        error('hyperparams solver not implemented', model);
    end
end