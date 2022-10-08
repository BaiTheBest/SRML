function [beta,obj_history] = Least_SWMTL0(X,Y)

    %%%
    %  hyperparams: rho, lambda, c
    %%

    [~,T] = size(X);
    [~,d] = size(X{1});
    
    beta = ones(T,d);
    
    z = ones(T,d);

    rou = 10; lambda = 5; c = 1; % hyperparameters
    l = 1; % the iteration counter
    
    obj_history = zeros(2000,T,d);
    
    
    while l <= 3000

        ittime = tic();
        elapsed_time=0;
        fval_total = 0;
        % Update beta's
        for k = 1:T
            for p = 1:d
                fun = @(x)Obj_fun(x,X,Y,beta,z,k,p,rou,lambda,c); 
                lb = -100000;
                ub = 100000;
                tic ();
                [new_x,fval] = fminbnd(fun,lb,ub);
                elapsed_time = elapsed_time + toc ();
                beta(k,p) = new_x;
                obj_history(l,k,p) = fval;
                fval_total = fval_total + fval;
            end
        end
        % Update z
        for k = 1:T
            for p = 1:d
                z(k,p) = beta(k,p);
            end
        end
        l = l + 1;

        allittime = toc (ittime);

        fprintf('\r training iter: %d cost: %d it time: %d fminbnd time: %d', l, fval_total, allittime, elapsed_time);

    end

    beta = beta';
end

function f = Obj_fun(x,X,Y,beta,z,k,p,rou,lambda,c)

    [~,T] = size(X);
    Xk = X{k};
    
    f = 0;
    
    f = f + norm(Y{k}-X{k}*(beta(k,:).')+Xk(:,p)*(beta(k,p)-x))^2;
    
    f = f + lambda*(x^2);
    
    if k == 1
        f = f + c*(max(0,-x*beta(k+1,p)));
    elseif k < T
        f = f + c*(max(0,-x*beta(k+1,p)) + max(0,-x*beta(k-1,p)));
    end
    
    f = f + (rou/2)*(x-z(k,p))^2;
    
end