function [W, J_hist, primal_residual_val, dual_residual_val] = Least_SWMTL32(X, Y, opts)

    % hyperparameters
    opts = init_opts_swmtl(opts) ;  

    % data variables
    T = length(X);
    d = size(X{1},2);   
    
    
    % output variables
    if isfield(opts, 'W0')
        W = opts.W0;
    else
        W = rand(d,T);     
    end

    % internal variables
    U = W;
    L = W;

    
    % data normalization
    if opts.scaling
        disp('scaling data')
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
    
    %init history
    disp('initializing hist')
    for t=1:T
        J_hist{t} =[];
    end
        
    
    i = 1; % the iteration counter
    while i <= opts.max_iters 

        U_previous = U;
        
        J_hist_all(i) = 0;
        
        for t = 1:T        
            [wt, jt]=minimizeJ(X, Y, W, U, L, t, opts);            
            W(:, t) = wt;
            J_hist{t} = vertcat([J_hist{t}, jt]);            
            last_cost= J_hist{t}(end);
            J_hist_all(i) =J_hist_all(i) + last_cost;
            
            if iscomplex(last_cost)
                last_cost=real(last_cost); %TODO: why complex???
            end
            fprintf('\r iter: %d task: %d cost: %d' ,i, t, last_cost);
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
            prev_hist = J_hist_all(end-1);
            curr_hist = J_hist_all(end);

            if opts.tflag == 1
                delta_hist = abs(prev_hist - curr_hist) ./ prev_hist;
                if (delta_hist > opts.tol) == 0
                    break
                end                                
            elseif opts.tflag == 2
                delta_hist = abs(prev_hist - curr_hist);                
                if (delta_hist > opts.tol) == 0
                    break
                end
            else
                delta_hist = abs(prev_hist - curr_hist);
            end
        end
        
        
        L = L + opts.rho*(W-U);
        i = i + 1;

    end
        
    if opts.scaling == 1
        W=[W./sigma_X.*sigma_Y];             
    elseif opts.scaling == 2
        ix_sigma = sigma_X > 10;
        W(ix_sigma)=W(ix_sigma)./sigma_X(ix_sigma).*(sigma_Y.*ones(size(W)))(ix_sigma);              
    end

end


function [wt, jt] = minimizeJ(X, Y, W, U, L, t,opts)
    T = length(X);
    xt = X{t};
    yt = Y{t};
    wt = W(:,t);
    ut = U(:,t); 
    lt = L(:,t);
    [m, d] = size(xt);

    % momentum
    mlr_V = ones(d,1);     
    mlr_mu = 0.9;
    mlr_alpha = opts.lr;
    % accelerated gradient
    nag_V = ones(d,1);
    nag_mu = 0.9;
    nag_alpha = opts.lr;
    % adagrad
    % rmsprop
    rmsprop_gamma = 0.1;
    rmsprop_r=wt;


    for iter = 1:opts.max_iters
    
        if opts.optimizer == 3
            nag_W = W + nag_mu * nag_V;
            [cost, grad_W] = computeGradients(X, Y, nag_W, U, L, t,  opts.lambda, opts.rho, opts.c);
        else
            [cost, grad_W] = computeGradients(X, Y, W, U, L, t,  opts.lambda, opts.rho, opts.c);
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
            adagrad_hist(iter,:) = grad_W;
            adagrad_W = grad_W ./ sqrt(sum(adagrad_hist.^2))';
            W(:, t) = W(:, t) - opts.lr * adagrad_W;
        elseif opts.optimizer == 5 % rmsprop
            rmsprop_r = rmsprop_gamma*rmsprop_r + (1-rmsprop_gamma)*grad_W.^2;
            W(:, t) = W(:, t) - opts.lr * grad_W./ sqrt(rmsprop_r);
        end   
            
        J_history(iter) = cost;
        
        if iter >1 
            prev_hist = J_history(iter-1);
            curr_hist = J_history(iter);

            if opts.tflag == 1
                delta_hist(iter) = abs(prev_hist - curr_hist) ./ prev_hist;
                if sum(delta_hist(end) > opts.tol) == 0
                    break
                end                                
            elseif opts.tflag == 2
                delta_hist(iter) = abs(prev_hist - curr_hist);                
                if sum(delta_hist(end) > opts.tol) == 0
                    break
                end
            else # 0 : not stop criteria
                delta_hist(iter) = abs(prev_hist - curr_hist);
            end
        end
    
    end
    
    wt = W(:,t);
    jt = J_history;
    
end


function [J,grad] = computeGradients(X, Y, W, U, L, t, lambda, rho, c)
    T = length(X);
    xt = X{t};
    yt = Y{t};
    wt = W(:,t);
    ut = U(:,t); 
    lt = L(:,t);
    [m, d] = size(xt);
    
    yt_hat = xt*wt;
    yt_error = yt_hat-yt;
    least = 1/(2*m) * sum(yt_error.^2);
    reg = lambda/(2*m) * sum(wt.^2);

    if t == 1
        utn = U(:,t+1);
        slack = c * sum(max(0, - ut.*utn));
    elseif t < T
        utp = U(:,t-1);
        utn = U(:,t+1);
        slack = c * (sum(max(0, - ut.*utp)) + sum(max(0, - ut.*utn)));
    else
        utp = U(:,t-1);
        slack = c * sum(max(0, - ut.*utp));
    end
    
    J = least + reg + slack;    
    grad = (1/m)*(yt_error'*xt)' + lambda/m*wt + rho*(wt-ut)+ lt;
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

function opts = init_opts_swmtl(opts)
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