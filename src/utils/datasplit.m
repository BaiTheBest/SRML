

function [ X_train, Y_train, X_test, Y_test ] = datasplit( X, Y, train_size, random_state )

    n=length(Y);
    for i=1:n
        m=length(Y{i});
        seq=datasamplenorepl(1:m,round(m * train_size), random_state);
        X_train{i}=X{i}(seq,:);
        Y_train{i}=Y{i}(seq,:);
        X_test{i}=X{i};
        Y_test{i}=Y{i};
        X_test{i}(seq,:)=[];
        Y_test{i}(seq,:)=[]; 
    end

end