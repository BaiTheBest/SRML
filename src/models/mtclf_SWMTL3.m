function [mt_func , hp_solver ] = mtclf_SWMTL3(solver)
    mt_func = @mtclf_func;
    hp_solver = get_hyperparams_solver(solver);
end

function [ AUC, ACC ] = mtclf_func( x_train, y_train, x_test, y_test, pars )
    [W, obj_hist]= Logistic_SWMTL3(x_train, y_train, pars);
    [AUC, ACC] = mtclf_test_sigmoid(x_test, W, y_test, pars); 
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
            'lambda', logspace(-5,2,5), ...
            'rho',logspace(-1,3,5), ...
            'c', logspace(-5,1,3)
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