
function [ y_train, y_test ] = convert2sign( y_train, y_test )

    % t = size(x_train, 2);
    % for i=1:t
    %     y_train{i}(y_train{i}==0,1)=-1;
    %     y_test{i}(y_test{i}==0,1)=-1;
    % end


    t = size(y_train, 2);
    for i=1:t
        y_train{i}(y_train{i}==0,1)=-1;
        y_test{i}(y_test{i}==0,1)=-1;
    end

end