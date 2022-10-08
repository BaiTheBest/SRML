function [AUC, ACC, MAP] = mtclf_test_sigmoid(X, W, Y, opts)
    sigmoid = @(z) 1./(1 + exp(-z));
    t=size(X,2)

    for i = 1: t
        Y_hat{i} = round(sigmoid(X{i} * W(:,i)));    
    end

    [AUC, ACC, MAP] = mtclf_test(X, W, Y, Y_hat, opts);
end

