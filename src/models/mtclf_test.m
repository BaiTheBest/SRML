function [AUC, ACC, MAP] = mtclf_test(X, W, Y, Y_hat, opts)
    
    t=length(Y);

    for i = 1: t
        y_pred = Y_hat{i};
        y_true = Y{i};    
        num_samples = length(y_true);
        auc(i) = fastAUC(y_true == 1, y_pred == 1, false);
        acc(i) = nnz(y_pred==y_true)/num_samples;

        t_pred = sum(y_pred==1);
        if t_pred > 0
            precision(i) = sum(y_pred==y_true & y_true==1) / t_pred;
        else
            precision(i) = 0;
        end
    end

    ACC = mean(acc);
    ACC_std = std(acc);
    MAP = mean(precision);
    MAP_std = std(precision);
    AUC = mean(auc);
    AUC_std = std(auc);    

    if isfield(opts, 'results_path')
        disp('saving model...')
        results_path = strcat(opts.results_path, 'weights.mat');
        save(results_path, 'W',  '-v6');
        results_path = strcat(opts.results_path, 'predictions.mat');
        save(results_path, 'Y', 'Y_hat', '-v6');
        results_path = strcat(opts.results_path, 'metrics.csv');
        r = [AUC, AUC_std, ACC, ACC_std, MAP, MAP_std]
        csvwrite(results_path, r);
        disp('saving model...[ok]')
    end 

end

