function [ mt_func , hp_solver ] = mtreg_LRST(solver)
    mt_func = @mtreg_func;
    hp_solver = get_hyperparams_solver(solver);
end

function [ MSE, MSLE, MAE, EV, R2 ] = mtreg_func( x_train, y_train, x_test, y_test, pars )
    [W, loss_values]=Least_LRST(x_train, y_train, pars);
    [MSE, MSLE, MAE, EV, R2]=mtreg_test(W,x_test,y_test, pars); 
end

function [hp_solver] = get_hyperparams_solver(solver)
    if strcmp(solver, 'grid_small') == 1
        hp_solver = make_solver('grid search',...
            'lambda',logspace(-1,1,3) % [0.1, 1, 10]
        );  
    elseif strcmp(solver, 'grid_large') == 1
        hp_solver = make_solver('grid search',...
            'lambda',logspace(-3,3,5)
        );  
    elseif strcmp(solver, 'swarm') == 1
        hp_solver = make_solver('particle swarm', 
            'num_particles', 2, ...
            'num_generations', 4, ...
            'max_speed', 0.1, ...
            'rho1', logspace(-3,3,2), ...
            'rho_L2',logspace(-3,3,2)
            % 'init', [0,2], ...
            % 'tFlag', [0,3], ...
            % 'tol', logspace(-5,-4,2), ...
            % 'maxIter', logspace(1,4, 2)
        );
    else
        error('hyperparams solver not implemented', model);
    end
end

