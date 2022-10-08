function [AUC, ACC, MAP, Y_hat] = mtclf_test_sign(X, W, C, Y, opts)
    
    t=size(X,2)

    for i = 1: t
        Y_hat{i} = sign(X{i} * W(:, i) + C(i));   
    end

    [AUC, ACC, MAP] = mtclf_test(X, W, Y, Y_hat, opts);
end

