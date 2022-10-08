function [y,i] = datasamplenorepl(x,k, random_state)

% Sample without replacement

    dim = find(size(x)~=1, 1); 
    if isempty(dim), dim = 1; end
    n = size(x,dim);

    replace = false;


    if k > n
        error(message('sample Too Large'));
    end

    rand('state', random_state);
    i = randperm(n,k);
        
    if dim == 1
        y = x(i,:);
    elseif dim == 2
        y = x(:,i);
    else
        reps = [ones(1,dim-1) k];
        y = repmat(x,reps);
    end

end