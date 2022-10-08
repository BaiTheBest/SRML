function [W, obj_history, delta_hist] = Least_SWMTL1(X, Y, opts)

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
        W = ones(d,T);     
    end

    % internal variables
    z = zeros(d,T);
    i = 1; % the iteration counter
    
    % optimizers
    mlr_V = ones(d,T);     
    mlr_mu = 0.9;
    mlr_alpha = opts.lr;
    nag_V = ones(d,T);
    nag_mu = 0.9;
    nag_alpha = opts.lr;
    
    % data normalization
    if opts.scaling
        sigma_X = zeros(d, T);
        mu_X = zeros(d, T);

        for t=1:T 
            if opts.scaling == 1
                [X{t}, mu_X(:,t), sigma_X(:,t)] = featureNormalize(X{t});
            elseif opts.scaling == 2
                [X{t}, mu_X(:,t), sigma_X(:,t)] = minMaxFeatureNormalize(X{t});
            end
            %m = size(X{t}, 1);
            %X{t} = [ones(m, 1) X{t}];
        end
    end
    
    % gradient descend
    while i <= opts.max_iters 
    
        
        for t = 1:T
            obj_history(i, t) = slackObj(X, Y, W, z, opts.rho, opts.lambda, opts.c, t);

            % Update W's
            if opts.optimizer == 3
                nag_W = W + nag_mu * nag_V;
                grad_W = slackGrad(X, Y, nag_W, z, t, opts.lambda, opts.rho, opts.c);
            else
                grad_W = slackGrad(X, Y, W, z, t, opts.lambda, opts.rho, opts.c);
            end

            if opts.optimizer == 1 % GD
                W(:, t) = W(:, t) - opts.lr * grad_W;
            elseif opts.optimizer == 2 % momentum GD
                mlr_V(:, t) = mlr_mu * mlr_V(:, t) - mlr_alpha * grad_W;
                W(:, t) = W(:, t) + mlr_V(:, t);
            elseif opts.optimizer == 3 % NAG
                nag_V(:, t) = nag_mu * nag_V(:, t) - nag_alpha * grad_W;
                W(:, t) = W(:, t) + nag_V(:, t);
            end                        

        end
        
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

        %W = Wtmp;
        z = W;
        i = i + 1;
    end
    
    if opts.scaling == 1
        W=[W./sigma_X];        
    end
end

function [X_norm, mu, sigma] = featureNormalize(X)
    mu = mean(X);
    sigma = std(X);
    X_norm = [(X - mu) ./ sigma];
end

function [W] = gdOptimizer(W, grad_W, t, lr)
    W(:, t) = W(:, t) - lr * grad_W;
end

function f = slackObj(X, Y, W, z, rho, lambda, c, t)   
    T = length(X);
    xt = X{t};
    yt = Y{t};
    wt = W(:,t);
    zt = z(:,t); 
    [m, d] = size(xt);    
    
    yt_hat = xt*wt;
    least = 1/(2*m) * sum((yt - yt_hat).^2);
    reg = lambda/(2*m) * sum(wt.^2);
    auglag = (rho/2*m) * sum((wt-zt).^2);

    if t == 1
        wtn = W(:,t+1);
        slack = c * sum(max(0, - wt.*wtn));
    elseif t < T
        wtp = W(:,t-1);
        wtn = W(:,t+1);
        slack = c * sum(max(0, - wt.*wtp));
        slack = slack + sum(max(0, - wt.*wtn));
    else
        wtp = W(:,t-1);
        slack = c * sum(max(0, - wt.*wtp));
    end
    
    f = least + reg + auglag + slack;
end

function grad = slackGrad(X, Y, W, z, t, lambda, rho, c)
    T = length(X);
    xt = X{t};
    yt = Y{t};
    wt = W(:,t);
    zt = z(:,t); 
    [m, d] = size(xt);
    
    grad = (1/m)*xt'*(xt*wt - yt) + (lambda/m)*wt + (rho/m)*(wt-zt);

    if t == 1 
        wtn = W(:,t+1);
        slackGrad = wt .* wtn;
        slackGrad(slackGrad >= 0) = 0;
        slackGrad(slackGrad < 0) = -wtn(slackGrad < 0);
    elseif t == T
        wtp = W(:,t-1);  
        slackGrad = wt .* wtp;
        slackGrad(slackGrad >= 0) = 0;
        slackGrad(slackGrad < 0) = -wtp(slackGrad < 0);    
    else
        wtp = W(:,t-1);
        wtn = W(:,t+1);
        slackGrad1 = wt .* wtn;
        slackGrad1(slackGrad1 >= 0) = 0;
        slackGrad1(slackGrad1 < 0) = -wtn(slackGrad1 < 0);
        slackGrad2 = wt .* wtp;
        slackGrad2(slackGrad2 >= 0) = 0;
        slackGrad2(slackGrad2 < 0) = -wtp(slackGrad2 < 0);
        slackGrad = slackGrad1 + slackGrad2;
    end

    grad = grad + c * slackGrad;
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
        opts.lr = 10^-1;
    end
    
    if ~isfield(opts, 'optimizer')
        opts.optimizer = 3;
    end
    
    if ~isfield(opts, 'max_iters')
        opts.max_iters = 1000;
    end
    
    if ~isfield(opts, 'tol')
        opts.tol = 10^-5;
    end
    
    if ~isfield(opts, 'scaling')
        opts.scaling = 0; % 1 is Z normal scaling, 2 is min max scaling
    end
        
    if ~isfield(opts, 'tflag')
        opts.tflag = 1; % percentage change
    end

end