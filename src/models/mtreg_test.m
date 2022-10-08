function [MSE,MSLE,MAE,EV,R2]=mtreg_test(W,X,Y, opts)
    n=size(W,2);
    m=0;
    y=[];
    y_hat=[];
    for i=1:n
        m=m+length(Y{i});
        y=[y;Y{i}];
        y_task = X{i}*W(:,i);
        y_hat=[y_hat; y_task];
        Y_hat{i} = y_task; 
        %rmse(i) =    sqrt(mean((Y{i}-Y_hat{i}).^2));   
    end
    MSE=norm(y-y_hat)^2/m;
    %MSE=sqrt(mean((y-y_hat).^2));
    %MSE = mean(rmse);
    MSLE=norm(log(1+y)-log(1+y_hat))^2/m;
    MAE=norm(y-y_hat,1)/m;
    EV=1-var(y-y_hat)/var(y);
    R2=1-norm(y-y_hat)^2/norm(y-mean(y))^2;

    if isfield(opts, 'results_path')
        disp('saving model...')        
        results_path = strcat(opts.results_path, 'weights.mat');
        save(results_path, 'W', '-v6');
        results_path = strcat(opts.results_path, 'predictions.mat');
        save(results_path, 'Y_hat', '-v6');
        results_path = strcat(opts.results_path, 'metrics.csv');
        r = [MSE, MSLE, MAE, EV, R2]
        csvwrite(results_path, r);
        disp('saving model...[ok]')
    end
end