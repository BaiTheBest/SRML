function [W, obj_history, primal_residual_val, dual_residual_val] = Logistic_SWMTL31(X, Y, opts)

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
        W = rand(d,T);     
    end

    % internal variables
    U = zeros(d,T);
    L = zeros(d,T);
    i = 1; % the iteration counter
    
    % optimizers
    mlr_V = ones(d,T);     
    mlr_mu = 0.9;
    mlr_alpha = opts.lr;
    nag_V = ones(d,T);
    nag_mu = 0.9;
    nag_alpha = opts.lr;
    %primal_residual_val(1)=10^1;
    %dual_residual_val(1)=10^1;    
    
    % data normalization
    if opts.scaling
        sigma_X = zeros(d, T);
        mu_X = zeros(d, T);

        for t=1:T 
            if opts.scaling == 1
                [X{t}, mu_X(:,t), sigma_X(:,t)] = featureNormalize(X{t});
                %[Y{t}, mu_Y(:,t), sigma_Y(:,t)] = featureNormalize(Y{t});
            elseif opts.scaling == 2
                [X{t}, mu_X(:,t), sigma_X(:,t)] = minMaxFeatureNormalize(X{t});
                %[Y{t}, mu_Y(:,t), sigma_Y(:,t)] = minMaxFeatureNormalize(Y{t});                
            end
            %m = size(X{t}, 1);
            %X{t} = [ones(m, 1) X{t}];
        end
    end
    
    % gradient descend
    while i <= opts.max_iters 
    
        U_previous = U;
        
        for t = 1:T
            obj_history(i, t) = slackObj(X, Y, W, U, L, t, opts.lambda, opts.rho, opts.c);

            % Update W's
            if opts.optimizer == 3
                nag_W = W + nag_mu * nag_V;
                grad_W = slackGrad(X, Y, nag_W, U, L, t, opts.lambda, opts.rho, opts.c);
            else
                grad_W = slackGrad(X, Y, W, U, L, t, opts.lambda, opts.rho, opts.c);
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
        
        for t = 1:T
            for j = 1:d
                U(j,t) = updateU(X, W, U, L, t, j, opts.rho, opts.c);    
            end
        end

        primal_residual = W-U ;
        dual_residual = -opts.rho * (U - U_previous);

        primal_residual_val(i) = sqrt(sum(sum(primal_residual.^2)));
        dual_residual_val(i) = sqrt(sum(sum(dual_residual.^2)));
            
            %if primal_residual_val(i-1) > 0
                %primal_residual_val(i) = sqrt(sum(sum(primal_residual.^2))) / primal_residual_val(i-1);
            %else
            %    primal_residual_val(i) = 0;
            %end
            %if dual_residual_val(i-1) > 0    
                %dual_residual_val(i) = sqrt(sum(sum(dual_residual.^2))) /  dual_residual_val(i-1);
            %else
            %    dual_residual_val(i) = 0;
            %end
            %ASSUMPTION: input values magnitudes across tasks are similar
            #abs_tol = sqrt(mean(mean(X{1})))
            %m = size(X{1},1);
            %tol = 0.01 * sqrt(m*d) + opts.tol * max(norm(primal_residual), norm(dual_residual));
%         if i>1
%             primal_residual_perc(i) = abs(primal_residual_val(i)-primal_residual_val(i-1)) / primal_residual_val(i-1);
%             dual_residual_perc(i) = abs(dual_residual_val(i)-dual_residual_val(i-1)) / dual_residual_val(i-1)    ;        
%             if primal_residual_perc(i) <= opts.tol && dual_residual_perc(i) <= opts.tol
%                 break
%             end            
%         end

       %stop criteria
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
        
        L = L + opts.rho*(W-U);
        i = i + 1;        
    end
    
    if opts.scaling > 0
        W=W./sigma_X;  
    end
end

function [X_norm, mu, sigma] = featureNormalize(X)
    mu = mean(X);
    sigma = std(X);
    X_norm = [(X - mu) ./ sigma];
end


function J = slackObj(X, Y, W, U, L, t, lambda, rho, c)   
    T = length(X);
    xt = X{t};
    yt = Y{t};
    wt = W(:,t);
    ut = U(:,t); 
    lt = L(:,t); 
    [m, d] = size(xt);
    
    h = sigmoid(xt*wt);
    smooth_cost = 1/(2*m) * ( -yt' * log(max(h,eps)) - (1-yt)' * log(1-min(h,1-eps)) );
    reg_cost = lambda/(2*m) * sum(wt.^2);

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

    J = smooth_cost + reg_cost + slack;
end

function grad = slackGrad(X, Y, W, U, L, t, lambda, rho, c)
    T = length(X);
    xt = X{t};
    yt = Y{t};
    wt = W(:,t);
    ut = U(:,t); 
    lt = L(:,t);
    [m, d] = size(xt);
        
    h = sigmoid(xt*wt);
    smooth_grad = (1/m)*xt' * (h - yt) + (1/m)*lambda*wt;
    lagrange_grad = rho * (wt-ut) + lt;
    grad = smooth_grad + lagrange_grad;
end

function v = slackingFunVal(x, un, up, w, l, rho, c)
    v = c*(max(0,-un*x) + max(0,-up*x)) - l*x + (rho/2)*(x-w)^2;
end

function u = updateU(X, W, U, L, t, j, rho, c)   
    % Update u(t,j), i.e. the u variable in t-th task and j-th feature 
    xt = X{t};
    T = length(X);
    wtj = W(j,t);
    utj = U(j,t);
    ltj = L(j,t);
    d = size(xt, 2);
    
    if rho != 0
    if t == 1 
        utjn = U(j,t+1);
        if utjn <= 0
            temp_candidate_1 = min(0, wtj + ltj/rho);
            temp_candidate_2 = max(0, wtj + (c*utjn + ltj)/rho);
            temp_val_1 = slackingFunVal(temp_candidate_1, utjn, 0, wtj, ltj, rho, c);
            temp_val_2 = slackingFunVal(temp_candidate_2, utjn, 0, wtj, ltj, rho, c);
            if temp_val_1 < temp_val_2
                u = temp_candidate_1;
            else
                u = temp_candidate_2;
            end
        else
            temp_candidate_1 = max(0, wtj + ltj/rho);
            temp_candidate_2 = min(0, wtj + (c*utjn + ltj)/rho);
            temp_val_1 = slackingFunVal(temp_candidate_1, utjn, 0, wtj, ltj, rho, c);
            temp_val_2 = slackingFunVal(temp_candidate_2, utjn, 0, wtj, ltj, rho, c);
            if temp_val_1 < temp_val_2
                u = temp_candidate_1;
            else
                u = temp_candidate_2;
            end
        end

    elseif t == T
        utjp = U(j,t-1);
        if utjp <= 0
            temp_candidate_1 = min(0, wtj + ltj/rho);
            temp_candidate_2 = max(0, wtj + (c*utjp + ltj)/rho);
            temp_val_1 = slackingFunVal(temp_candidate_1, 0, utjp, wtj, ltj, rho, c);
            temp_val_2 = slackingFunVal(temp_candidate_2, 0, utjp, wtj, ltj, rho, c);
            if temp_val_1 < temp_val_2
                u = temp_candidate_1;
            else
                u = temp_candidate_2;
            end
        else
            temp_candidate_1 = max(0, wtj + ltj/rho);
            temp_candidate_2 = min(0, wtj + (c*utjp + ltj)/rho);
            temp_val_1 = slackingFunVal(temp_candidate_1, 0, utjp, wtj, ltj, rho, c);
            temp_val_2 = slackingFunVal(temp_candidate_2, 0, utjp, wtj, ltj, rho, c);
            if temp_val_1 < temp_val_2
                u = temp_candidate_1;
            else
                u = temp_candidate_2;
            end
        end
        
    else
        utjp = U(j,t-1);
        utjn = U(j,t+1);        
        if utjn <= 0 && utjp <= 0
            temp_candidate_1 = min(0, wtj + ltj/rho);
            temp_candidate_2 = max(0, wtj + (c*(utjn+utjp) + ltj)/rho);
            temp_val_1 = slackingFunVal(temp_candidate_1, utjn, utjp, wtj, ltj, rho, c);
            temp_val_2 = slackingFunVal(temp_candidate_2, utjn, utjp, wtj, ltj, rho, c);
            if temp_val_1 < temp_val_2
                u = temp_candidate_1;
            else
                u = temp_candidate_2;
            end
        elseif utjn <= 0 && utjp > 0
            temp_candidate_1 = min(0, wtj + (c*utjp + ltj)/rho);
            temp_candidate_2 = max(0, wtj + (c*utjn + ltj)/rho);
            temp_val_1 = slackingFunVal(temp_candidate_1, utjn, utjp, wtj, ltj, rho, c);
            temp_val_2 = slackingFunVal(temp_candidate_2, utjn, utjp, wtj, ltj, rho, c);
            if temp_val_1 < temp_val_2
                u = temp_candidate_1;
            else
                u = temp_candidate_2;
            end
        elseif utjn > 0 && utjp <= 0
            temp_candidate_1 = min(0, wtj + (c*utjn + ltj)/rho);
            temp_candidate_2 = max(0, wtj + (c*utjp + ltj)/rho);
            temp_val_1 = slackingFunVal(temp_candidate_1, utjn, utjp, wtj, ltj, rho, c);
            temp_val_2 = slackingFunVal(temp_candidate_2, utjn, utjp, wtj, ltj, rho, c);
            if temp_val_1 < temp_val_2
                u = temp_candidate_1;
            else
                u = temp_candidate_2;
            end
        else
            temp_candidate_1 = max(0, wtj + ltj/rho);
            temp_candidate_2 = min(0, wtj + (c*(utjn+utjp) + ltj)/rho);
            temp_val_1 = slackingFunVal(temp_candidate_1, utjn, utjp, wtj, ltj, rho, c);
            temp_val_2 = slackingFunVal(temp_candidate_2, utjn, utjp, wtj, ltj, rho, c);
            if temp_val_1 < temp_val_2
                u = temp_candidate_1;
            else
                u = temp_candidate_2;
            end
        end
        
    end
    end
    
end

function [h] = sigmoid(z)
    h = 1./(1+ e.^-z);
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
        opts.tol = 10^-4;
    end
    
    if ~isfield(opts, 'scaling')
        opts.scaling = 0; % 1 is Z normal scaling, 2 is min max scaling
    end
        
    if ~isfield(opts, 'tflag')
        opts.tflag = 0; % percentage change
    end

end