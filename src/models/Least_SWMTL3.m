function [W, obj_history, primal_residual_val, dual_residual_val, delta_hist,pchange] = Least_SWMTL3(X, Y, opts)

    % hyperparameters
    opts = init_opts(opts) ;  

    % data variables
    T = length(X);
    d = size(X{1},2);   
    
    if opts.scaling
        %d = d + 1;
        d = d;
    end
    
    % output variables
    if isfield(opts, 'W0')
        W = opts.W0;
    else
        W = zeros(d,T);     
    end

    % internal variables
    U = zeros(d,T);
    L = zeros(d,T);
    i = 1; % the iteration counter
    
    % optimizers
    mlr_V = W;     
    mlr_mu = 0.9;
    mlr_alpha = opts.lr;
    nag_V = W;
    nag_mu = 0.9;
    nag_alpha = opts.lr;
    adagrad_hist=W;
    rmsprop_gamma = 0.1;
    rmsprop_r=W;
    
    % data normalization
    if opts.scaling
        for t=1:T 
            if opts.scaling == 1
                [X{t}, mu_X(:,t), sigma_X(:,t)] = featureNormalize(X{t});
                [Y{t}, mu_Y(:,t), sigma_Y(:,t)] = featureNormalize(Y{t});
            elseif opts.scaling == 2
                [X{t}, mu_X(:,t), sigma_X(:,t)] = featureNormalize2(X{t});
                [Y{t}, mu_Y(:,t), sigma_Y(:,t)] = featureNormalize2(Y{t});
            end
        end
    end
    
    % gradient descend
    while i <= opts.max_iters 
    
        U_previous = U;
        
        for t = 1:T
            obj_history(i, t) = computeCost(X, Y, W, U, L, t, opts.lambda, opts.rho, opts.c);

            % Update W's
            if opts.optimizer == 3
                nag_W = W + nag_mu * nag_V;
                grad_W = computeGradients(X, Y, nag_W, U, L, t, opts.lambda, opts.rho, opts.c);
            else
                grad_W = computeGradients(X, Y, W, U, L, t, opts.lambda, opts.rho, opts.c);
            end

            if opts.optimizer == 1 % GD
                W(:, t) = W(:, t) - opts.lr * grad_W;
            elseif opts.optimizer == 2 % momentum GD
                mlr_V(:, t) = mlr_mu * mlr_V(:, t) - mlr_alpha * grad_W;
                W(:, t) = W(:, t) + mlr_V(:, t);
            elseif opts.optimizer == 3 % NAG
                nag_V(:, t) = nag_mu * nag_V(:, t) - nag_alpha * grad_W;
                W(:, t) = W(:, t) + nag_V(:, t);
            elseif opts.optimizer == 4
                adagrad_hist(:, t) += grad_W;
                adagrad_W = grad_W ./ sqrt(sum(adagrad_hist(:,t).^2))';
                W(:, t) = W(:, t) - opts.lr * adagrad_W;
            elseif opts.optimizer == 5 % rmsprop
                rmsprop_r(:,t) = rmsprop_gamma*rmsprop_r(:,t) + (1-rmsprop_gamma)*grad_W.^2;
                W(:, t) = W(:, t) - opts.lr * grad_W./ sqrt(rmsprop_r(:,t));
            end   
        end
        
        for t = 1:T
            U(:,t) = updateU(X, Y, W, U, L, t, opts.lambda, opts.rho, opts.c);                                
        end

        


        primal_residual = W-U ;
        dual_residual = -opts.rho * (U - U_previous);
        primal_residual_val(i) = sqrt(sum(sum(primal_residual.^2)));
        dual_residual_val(i) = sqrt(sum(sum(dual_residual.^2)));
            
        % stop criteria
        if i >1 
            prev_hist = obj_history(i-1,:);
            curr_hist = obj_history(i, :);

            if opts.tflag == 1
                delta_hist(i, 1:T) = abs(prev_hist - curr_hist) ./ prev_hist;
                
                if sum(delta_hist(end, :) > opts.tol) == 0
                    break
                end                                
            elseif opts.tflag == 2
                delta_hist(i, 1:T) = abs(prev_hist - curr_hist);                
                if sum(delta_hist(end, :) > opts.tol) == 0
                    break
                end
            else
                delta_hist(i, 1:T) = abs(prev_hist - curr_hist);
            end
        end
        
        fprintf('\r training iter: %d cost: %d delta: %d', i, sum(obj_history(i, t)) );
        L = L + opts.rho*(W-U);
        i = i + 1;


        
        
    end
        
    if opts.scaling == 1
        W=[W./sigma_X.*sigma_Y];             
    elseif opts.scaling == 2
        ix_sigma = sigma_X > 10;
        W(ix_sigma,:)=W(ix_sigma,:)'./sigma_x(ix_sigma).*sigma_y;              
    end

end


function J = computeCost(X, Y, W, U, L, t, lambda, rho, c)   
    T = length(X);
    xt = X{t};
    yt = Y{t};
    wt = W(:,t);
    ut = U(:,t); 
    lt = L(:,t); 
    [m, d] = size(xt);
    
    yt_hat = xt*wt;
    least = 1/(2*m) * sum((yt - yt_hat).^2);
    reg = lambda/(2*m) * sum(wt.^2);

    if t == 1
        utn = U(:,t+1);
        slack = c * sum(max(0, - ut.*utn));
    elseif t < T
        utp = U(:,t-1);
        utn = U(:,t+1);
        slack = c * sum(max(0, - ut.*utp));
        slack = slack + sum(max(0, - ut.*utn));
    else
        utp = U(:,t-1);
        slack = c * sum(max(0, - ut.*utp));
    end
    
    J = least + reg + slack;
end

function grad = computeGradients(X, Y, W, U, L, t, lambda, rho, c)
    T = length(X);
    xt = X{t};
    yt = Y{t};
    wt = W(:,t);
    ut = U(:,t); 
    lt = L(:,t);
    [m, d] = size(xt);
        
    grad = (1/m)*xt'*(xt*wt - yt) + lambda/m*wt + rho*(wt-ut+lt/rho);
    %grad = (2*xt'*xt + (2*lambda+rho) * eye(d)) * wt - (2*xt'*yt + rho*(ut-lt))
end

function ut = updateU(X, Y, W, U, L, t, lambda, rho, c)
    T = length(X);
    xt = X{t};
    yt = Y{t};
    wt = W(:,t);
    ut = U(:,t);
    lt = L(:,t);
    d = size(xt, 2);   
    
    if rho != 0
    if t == 1 
        utn = U(:,t+1);
        slackGrad = ut .* utn;
        ut(slackGrad >= 0) = wt(slackGrad >= 0) + 1/(rho) * lt(slackGrad >= 0);
        ut(slackGrad < 0) = wt(slackGrad < 0) + 1/(rho) * ( c*utn(slackGrad < 0) + lt(slackGrad < 0) );
    elseif t == T
        utp = U(:,t-1);  
        slackGrad = ut .* utp;
        ut(slackGrad >= 0) = wt(slackGrad >= 0) + 1/(rho) * lt(slackGrad >= 0);
        ut(slackGrad < 0) = wt(slackGrad < 0) + 1/(rho) * (c*utp(slackGrad < 0) + lt(slackGrad < 0) );    
    else
        utp = U(:,t-1);
        utn = U(:,t+1);        
        slackGrad1 = ut .* utn;
        slackGrad2 = ut .* utp;
        
        ix = slackGrad1 >= 0 & slackGrad2 >= 0;
        ut(ix) = wt(ix) + 1/(rho) * lt(ix) ;
        
        ix = slackGrad1 >= 0 & slackGrad2 < 0;
        ut(ix) = wt(ix) + 1/(rho) * ( c*utp(ix) + lt(ix) );   
        
        ix = slackGrad1 < 0  & slackGrad2 >= 0;
        ut(ix) = wt(ix) + 1/(rho) * ( c*utn(ix) + lt(ix) );
        
        ix = slackGrad1 < 0  & slackGrad2 < 0;
        ut(ix) = wt(ix) + 1/(rho) * ( c*utp(ix) +  c*utn(ix) + lt(ix) );
    end
    end

end

function [X_norm, mu, sigma] = featureNormalize(X)
    mu = mean(X);
    sigma = std(X);
    sigma(sigma==0)=1;   
    X_norm = [(X - mu) ./ sigma];
end

function [X_norm, mu, sigma] = featureNormalize2(X)
    mu = mean(X);
    sigma = std(X);
    sigma(sigma==0)=1;   
    ix_norm = sigma > 10;
    X_norm = X;

    if sum(ix_norm) > 0
        X_norm(:,ix_norm) = [(X(:,ix_norm) - mu(ix_norm)) ./ sigma(ix_norm)];
    end
end

function [X_norm, mu, sigma] = minMaxFeatureNormalize(X)
    mu = min(X);
    sigma = max(X) - min(X) ;
    X_norm =  [(X - mu) ./ sigma];
end

function [X_norm, mu, sigma] = maxAbsFeatureNormalize(X)
    mu = 0;
    sigma = max(abs(X)) ;
    X_norm =  [(X ) ./ sigma];
end

function opts = init_opts(opts)
    if ~isfield(opts, 'rho')
        opts.rho = 10;
    end 
    
    if ~isfield(opts, 'lambda')
        opts.lambda = 5; 
    end
    
    if ~isfield(opts, 'c')
        opts.c = 1; 
    end
    
    if ~isfield(opts, 'lr')
        opts.lr = 10^-3;
    end
    
    if ~isfield(opts, 'optimizer')
        opts.optimizer = 3;
    end
    
    if ~isfield(opts, 'max_iters')
        opts.max_iters = 1000;
    end
    
    if ~isfield(opts, 'tol')
        opts.tol = 10^-4;
    end
    
    if ~isfield(opts, 'scaling')
        opts.scaling = 1; % 1 is Z normal scaling, 2 is min max scaling
    end
        
    if ~isfield(opts, 'tflag')
        opts.tflag = 1; % percentage change
    end

end