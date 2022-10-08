
function [w, obj_history, s_history] = Least_WMTL(X, Y, alpha, opts)
    if nargin <3
        error('\n Inputs: X, Y, and alpha should be specified!\n');
    end

    if nargin <4
        opts = [];
    end
    
    % initialize options.
    opts=init_opts(opts);

    if isfield(opts, 'rho')
        rho = opts.rho;
    else
        rho = 1000;
    end

    nntype = opts.nntype;
    
    n=length(X);
    p=size(X{1},2);    
    z1=zeros(p,n);
    w=2*rand(p,n)-1;
    u1=zeros(p,n);

    for i=1:opts.maxIter
        
        z1_old=z1;
        % update w
        for j=1:n
            if opts.nntype == 0 % NNTYPE_ORI            
                if(j==1)
                    [w(:,j)]=update_first_last( X{j},Y{j},alpha,rho,z1(:,j),u1(:,j),w(:,j+1),p);
                elseif (j==n)
                    [w(:,j)]=update_first_last( X{j},Y{j},alpha,rho,z1(:,j),u1(:,j),w(:,j-1),p);
                else
                    [w(:,j)]=update_w(          X{j},Y{j},alpha,rho,z1(:,j),u1(:,j),w(:,j-1),w(:,j+1),p);
                end
            elseif opts.nntype == 1
                if (j==1) nn_ix1 = n; else nn_ix1 = j-1; end;
                if (j==n) nn_ix2 = 1; else nn_ix2 = j+1; end;
                [w(:,j)]=update_w(          X{j},Y{j},alpha,rho,z1(:,j),u1(:,j),w(:,nn_ix1),w(:,nn_ix2),p);
            elseif opts.nntype == 2
                nn_ix1 = j; 
                if (j==n) nn_ix2 = 1; else nn_ix2 = j+1; end;
                [w(:,j)]=update_w(          X{j},Y{j},alpha,rho,z1(:,j),u1(:,j),w(:,nn_ix1),w(:,nn_ix2),p);    
            
            elseif opts.nntype == 3
                nn_ix1 = j; 
                if (j==n) nn_ix2 = 1; else nn_ix2 = j+1; end;
                [w(:,j)]=update_w(          X{j},Y{j},alpha,rho,z1(:,j),u1(:,j),w(:,nn_ix1),w(:,nn_ix2),p); 
                nn_ix2+=1;
                if (nn_ix2>=n) nn_ix2 = 1+nn_ix2-n; end;
                w(:,j) =update_w(          X{j},Y{j},alpha,rho,z1(:,j),u1(:,j),w(:,nn_ix1),w(:,nn_ix2),p);    
            end

        end
        % update z1
        z1=update_z1(w,rho,u1);
        s1=rho*(z1_old-z1);
        u1=u1+rho*(w-z1);
        s=norm([reshape(s1,size(s1,1)*size(s1,2),1)]);
        s_history(i)=s;
        obj=objective(alpha,X,Y,w,u1,z1,rho,n);
        obj_history(i)=obj;
        fprintf('\r training iter: %d cost: %d s: %d', i, obj, s);
        fflush(stdout);
        if(s<opts.tol)
            break;
        end
    end
    fprintf('\n');
    num_iters=i;
end

function obj=objective(alpha,X,Y,w,u1,z1,rho,n)
    obj=0;
    for i=1:n
        obj=obj+norm(Y{i}-X{i}*w(:,i))^2/length(Y)+alpha*norm(w(:,i))^2;
    end
    obj=obj+sum(sum(u1.*(w-z1)))+rho/2*norm(w-z1,'fro')^2;
end

function [w]=update_w(X,y,alpha,rho,z1,u1,w_last,w_next,p)
    n=length(y);
    %w=inv(2*X'*X/n+2*alpha*eye(p)+rho*eye(p)) * (2*X'*(y)/n+rho*z1-u1);
    %X = (2/n)* X; %(X - mean(X)) / std(X);
    w=inv((2/n)* X'*X  +alpha*eye(p)+rho*eye(p)) * ( (2/n)* X'*y    + rho*z1-u1);
    w(w_last.*w_next<0)=0;
    index=find(w_last>0&w_next>0);
    index=[index;find(w_last>0&w_next==0)];
    index=[index;find(w_last==0&w_next>0)];
    w(index)=max(w(index),0);
    index=find(w_last<0&w_next<0);
    index=[index;find(w_last<0&w_next==0)];
    index=[index;find(w_last==0&w_next<0)];
    w(index)=min(w(index),0);
end

function [w]=update_first_last(X,y,alpha,rho,z1,u1,w_neighbor,p)
    n=length(y);
    %w=inv(2*X'*X+2*alpha*eye(p)+rho*eye(p))*(2*X'*y+rho*z1-u1);
    %X = (X - mean(X)) / std(X); 
    w=inv((2/n)* X'*X +  2*alpha*eye(p)+rho*eye(p))*((2/n)* X'*y + rho*z1-u1);
    index=find(w_neighbor>0);
    w(index)=max(w(index),0);
    index=find(w_neighbor<0);
    w(index)=min(w(index),0);
end

function z1=update_z1(w,rho,u)
    z1=w+u/rho;
end


function opts = init_opts (opts)

    NNTYPE_ORI = 0;
    NNTYPE_SEQ = 1;

    %% Neighbors Type
    if !isfield(opts, 'nntype')
        opts.nntype = NNTYPE_ORI; %  use "0" for original implementation
    end

    disp(opts)

    %% Default values
    DEFAULT_MAX_ITERATION = 1000;
    DEFAULT_TOLERANCE     = 1e-4;
    MINIMUM_TOLERANCE     = eps * 100;
    DEFAULT_TERMINATION_COND = 1;
    DEFAULT_INIT = 2;
    DEFAULT_PARALLEL_SWITCH = false;
    
    %% Starting Point
    if isfield(opts,'init')
        if (opts.init~=0) && (opts.init~=1) && (opts.init~=2)
            opts.init=DEFAULT_INIT; % if .init is not 0, 1, or 2, then use the default 0
        end
        
        if (~isfield(opts,'W0')) && (opts.init==1)
            opts.init=DEFAULT_INIT; % if .W0 is not defined and .init=1, set .init=0
        end
    else
        opts.init = DEFAULT_INIT; % if .init is not specified, use "0"
    end
    
    %% Tolerance
    if isfield(opts, 'tol')
        % detect if the tolerance is smaller than minimum
        % tolerance allowed.
        if (opts.tol <MINIMUM_TOLERANCE)
            opts.tol = MINIMUM_TOLERANCE;
        end
    else
        opts.tol = DEFAULT_TOLERANCE;
    end
    
    %% Maximum iteration steps
    if isfield(opts, 'maxIter')
        if (opts.maxIter<1)
            opts.maxIter = DEFAULT_MAX_ITERATION;
        end
    else
        opts.maxIter = DEFAULT_MAX_ITERATION;
    end
    
    %% Termination condition
    if isfield(opts,'tFlag')
        if opts.tFlag<0
            opts.tFlag=0;
        elseif opts.tFlag>3
            opts.tFlag=3;
        else
            opts.tFlag=floor(opts.tFlag);
        end
    else
        opts.tFlag = DEFAULT_TERMINATION_COND;
    end
    
    %% Parallel Options
    if isfield(opts, 'pFlag')
        if opts.pFlag == true && ~exist('matlabpool', 'file')
            opts.pFlag = false;
            warning('MALSAR:PARALLEL','Parallel Toolbox is not detected, MALSAR is forced to turn off pFlag.');
        elseif opts.pFlag ~= true && opts.pFlag ~= false
            % validate the pFlag.
            opts.pFlag = DEFAULT_PARALLEL_SWITCH;
        end
    else
        % if not set the pFlag to default.
        opts.pFlag = DEFAULT_PARALLEL_SWITCH;
    end
    
    if opts.pFlag
        % if the pFlag is checked,
        % check segmentation number.
        if isfield(opts, 'pSeg_num')
            if opts.pSeg_num < 0
                opts.pSeg_num = matlabpool('size');
            else
                opts.pSeg_num = ceil(opts.pSeg_num);
            end
        else
            opts.pSeg_num = matlabpool('size');
        end
    end
end    