function [mt_func , hp_solver ] = mtreg_SWMTL31(solver)
    mt_func = @mtreg_func;
    hp_solver = get_hyperparams_solver(solver);
end

function [ MSE, MSLE, MAE, EV, R2 ] = mtreg_func(x_train, y_train, x_test, y_test, pars )
    %pars.lr = 1e-3;
    %pars.tol= 1e-4;
    % T = length(x_train);
    % for t=1:T 
    %     % [x_train{t}, mu_x(:,t), sigma_x(:,t)] = featureNormalize(x_train{t});           
    %     % [y_train{t}, mu_y(:,t), sigma_y(:,t)] = featureNormalize(x_test{t});           
    %     % x_test{t} = (x_test{t} -  mu_x(:,t)) / sigma_x(:,t);
    %     % y_test{t} = (y_train{t} -  mu_y(:,t)) / sigma_y(:,t);
    % end

    [W, loss_values]=Least_SWMTL31(x_train, y_train, pars);
    [MSE, MSLE, MAE, EV, R2]=mtreg_test(W, x_test, y_test, pars);
end

function [X_norm, mu, sigma] = featureNormalize(X)
    mu = mean(X);
    sigma = std(X);
    X_norm = [(X - mu) ./ sigma];
end

function [hp_solver] = get_hyperparams_solver(solver)
    if strcmp(solver, 'grid_small') == 1
        %performance = mtaskfunc(struct('alpha',100));
        hp_solver = make_solver('grid search',...
            'lambda', logspace(-2,-1,2), ... 
            'rho', logspace(-1,1,2), ...
            'c', logspace(-2,-1,2)
            % 'init', 0:2, ...
            % 'tFlag', 0:3, ...
            % 'tol', logspace(-5,-4,2), ...
            % 'maxIter', logspace(1,4, 4)
        );  
    elseif strcmp(solver, 'grid_large') == 1
        hp_solver = make_solver('grid search',...
            'lambda', logspace(-5,2,8), ...
            'rho',logspace(-1,3,5), ...
            'c', logspace(-5,1,7)
            % 'init', 0:2, ...
            % 'tFlag', 0:3, ...
            % 'tol', logspace(-5,-4,2), ...
            % 'maxIter', logspace(1,4, 4)
        );  
    elseif strcmp(solver, 'swarm') == 1
        hp_solver = make_solver('particle swarm', 
        'num_particles', 5, ...
        'num_generations', 30, ...
        'max_speed', 0.03, ...
        'alpha', logspace(-3,3,2), ...
        'rho', logspace(-3,3,2)
        );
    elseif strcmp(solver, 'json') == 1   
        hp_solver = solver ;    
    else
        error('hyperparams solver not implemented', model);
    end    
end

