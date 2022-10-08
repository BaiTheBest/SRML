
function [W, J_history] = Least_LRST(X, y, opts)
    opts = init_opts(opts)
    num_tasks = length(X);
    num_features = size(X{1},2); %+ 1; % add intercept
    W = zeros(num_features, num_tasks);
    
    for i=1:num_tasks        
        xt = X{i};
        yt = y{i};
        
        % data normalization
        if opts.scaling == 1
            [xt, mu_x, sigma_x] = featureNormalize(xt);
            [yt, mu_y, sigma_y] = featureNormalize(yt);
        elseif opts.scaling == 2
            [xt, mu_x, sigma_x] = featureNormalize2(xt);
            [yt, mu_y, sigma_y] = featureNormalize2(yt);
        end
    
        [wt, J_hist] = lrfit(xt, yt, opts);


        if opts.scaling == 1
            wt=wt'./sigma_x.*sigma_y;        
        elseif opts.scaling == 2
            ix_sigma = sigma_x > 10;
            wt(ix_sigma)=wt(ix_sigma)'./sigma_x(ix_sigma).*sigma_y;        
        end
        
        W(:, i) = wt;
        J_history{i} = J_hist;
        fprintf('task: %d iterations: %d    \r', i, length(J_hist));
        
    end

end



function [W, J_history] = lrfit(X, Y, opts)
    [m,d] = size(X);
    
    if ~isfield(opts, 'W0')
        W = rand(d, 1);
    else
        W = opts.W0;
    end
    

    
    mlr_V = ones(d,1);     
    mlr_mu = 0.9;
    mlr_alpha = opts.lr;
    
    nag_V = ones(d,1);
    nag_mu = 0.9;
    nag_alpha = opts.lr;

    % rmsprop
    rmsprop_gamma = 0.1;
    rmsprop_r(1,:)=W;

    for iter = 1:opts.max_iters
    
        if opts.optimizer == 3
            nag_W = W + nag_mu * nag_V;
            [cost, grad_W] = computeGradients(X, Y, nag_W, opts.lambda);
        else
            [cost, grad_W] = computeGradients(X, Y, W, opts.lambda);
        end
            
        if opts.optimizer == 1 % GD
            W = W - opts.lr * grad_W;
        elseif opts.optimizer == 2 % momentum GD
            mlr_V = mlr_mu * mlr_V - mlr_alpha * grad_W;
            W = W + mlr_V;
        elseif opts.optimizer == 3 % NAG
            nag_V = nag_mu * nag_V - nag_alpha * grad_W;
            W = W + nag_V;
        elseif opts.optimizer == 4 % adagrad
            adagrad_hist(iter,:) = grad_W;
            adagrad_W = grad_W ./ sqrt(sum(adagrad_hist.^2))';
            W = W - opts.lr * adagrad_W;
        elseif opts.optimizer == 5 % rmsprop
            rmsprop_r(iter+1,:) = rmsprop_gamma*rmsprop_r(iter,:) + (1-rmsprop_gamma)*grad_W'.^2;
            W = W - opts.lr * grad_W./ sqrt(rmsprop_r(iter+1,:)');
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
end


function [J, grads] = computeGradients(X, y, theta, lambda)
    m = length(y); 
    y_error =X*theta-y; 
    J = sum(y_error.^2)/(2*m) + lambda/(2*m) * sum(theta.^2); 
    grads = (1/m) * (y_error' * X)' + lambda/m*theta;
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



function opts = init_opts(opts)    
    if ~isfield(opts, 'lambda')
        opts.lambda = 0.1; 
    end
        
    if ~isfield(opts, 'lr')
        opts.lr = 3*10^-1;
    end
    
    if ~isfield(opts, 'optimizer')
        opts.optimizer = 4;
    end
    
    if ~isfield(opts, 'max_iters')
        opts.max_iters = 1000;
    end
    
    if ~isfield(opts, 'tol')
        opts.tol = 10^-4;
    end
    
    if ~isfield(opts, 'scaling')
        opts.scaling = 0; % 1 is Z normal scaling, 2 is min max scaling
    end
        
    if ~isfield(opts, 'tflag')
        opts.tflag = 1; % percentage change
    end

end