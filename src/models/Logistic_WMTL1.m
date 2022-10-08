function [W,obj_history]=Logistic_WMTL1(X, y, alpha, opts)

    if nargin <3
        error('\n Inputs: X, Y, and alpha should be specified!\n');
    end

    if nargin <4
        opts = [];
    end
    
    % initialize options.
    opts=Logistic_WMTL_options(opts);

    if isfield(opts, 'rho')
        rho = opts.rho;
    else
        %rho = 1000;
        opts = setfield(opts, 'rho', 1000);
    end

    disp(opts);

    global wmtl_links;
    links = wmtl_links;
    num_tasks = size(X,2)

    for i=1:num_tasks
        Xs{1} = X{i}(:,1:50);
        Xs{2} = X{i}(:,51:100);
        Xs{3} = X{i}(:,101:150);
        Ys = y{i};
        E12 = links;
        E13 = links;
        E23 = links;
        [beta1,beta2,beta3,obj_history]=logistic_wmtl_h1(Xs,Ys,E12,E13,E23, alpha, opts, i);
        W{i}=[beta1;beta2;beta3];    
    end
end


function [beta1,beta2,beta3,obj_history]= logistic_wmtl_h1(X,y,E12,E13,E23, nu, opts, task_id)
    m1=size(X{1},2);
    m2=size(X{2},2);
    m3=size(X{3},2);
    m=m1+m2+m3;
    z=zeros(m,1);
    beta1=zeros(m1,1);
    beta2=zeros(m2,1);
    beta3=zeros(m3,1);
    C=sum(y==0)/sum(y==1)*2;
    rho=opts.rho;
    max_iter=opts.maxIter;
    tolerance=opts.tol;

    for i=1:max_iter
        iter=i;
        %if mod(iter, 100) ==0
        fprintf('solving task %d iteration %d \r',task_id, i);
        %end
        z_old=z;
        % update beta1
        beta1=update_beta1(X,y,beta1,beta2,beta3,nu,rho,z,E12,E13,m1,C);
        % update beta2
        beta2=update_beta2(X,y,beta1,beta2,beta3,nu,rho,z,E12,E23,m1,m2,C);
        % update beta3
        beta3=update_beta3(X,y,beta1,beta2,beta3,nu,rho,z,E13,E23,m1,m2,m3,C);
        % update z
        z=update_z(beta1,beta2,beta3);
        s=rho*(z_old-z);
        s=norm(s);
        s_history(i)=s;
        obj=objective(nu,X,y,beta1,beta2,beta3,C);
        obj_history(i)=obj;
        if(s<tolerance)
            break;
        end
    end
    fprintf('\n');
end


function beta1=update_beta1(X,y,beta1,beta2,beta3,nu,rho,z,E12,E13,m1,C)
    % solve the epsilon update
    % via FISTA
    % global constants and defaults
    n = length(y);
    ybeta1 = 10e10;
    MAX_ITER = 500;
    TOLERANCE =10e-5;
    pos12=find(sum(E12,2)>0);
    pos13=find(sum(E13,2)>0);
    f = @(beta1) (mean(-C*y.*log(1./(1 + exp(-X{1}*beta1-X{2}*beta2-X{3}*beta3)))-(1-y).*log(1-1./(1 + exp(-X{1}*beta1-X{2}*beta2-X{3}*beta3))))+ nu*norm(beta1)^2+(rho/2)*norm(z(1:m1) - beta1,2)^2);
    lambda =1;
    zeta =beta1;
    eta = 10e-7;
    % FISTA
    for iter = 1:MAX_ITER
        ybeta1_old =ybeta1;
        ybeta1 = f(beta1);
        if(abs(ybeta1-ybeta1_old)<TOLERANCE)
            break;
        end
        lambda_old =lambda;
        lambda =(1+sqrt(1+4*lambda^2))/2;
        gamma =(1-lambda_old)/lambda;
        gradient=(X{1})'*((C*y-y+1).*exp(X{1}*beta1+X{2}*beta2+X{3}*beta3)./(1+exp(X{1}*beta1+X{2}*beta2+X{3}*beta3))-C*y)/n;
        zeta_old =zeta;
        zeta = (eta*rho*z(1:m1)+beta1-eta*gradient)/(2*nu*eta+rho*eta+1);
        %check ineuqality constraint
        for j=1:length(pos12)
            product=zeta(pos12(j))*beta2(E12(pos12(j),:)==1);
            if any(product<0)
                zeta(j)=0;
            end
        end
        for j=1:length(pos13)
            product=zeta(pos13(j))*beta2(E13(pos13(j),:)==1);
            if any(product<0)
                zeta(j)=0;
            end
        end       
        beta1 =(1-gamma)*zeta+gamma*zeta_old;
    end
end

function beta2=update_beta2(X,y,beta1,beta2,beta3,nu,rho,z,E12,E23,m1,m2,C)
    % via FISTA
    % global constants and defaults
    n = length(y);
    ybeta2 = 10e10;
    MAX_ITER = 500;
    TOLERANCE =10e-5;
    pos12=find(sum(E12,1)>0);
    pos23=find(sum(E23,2)>0);
    f = @(beta2) (mean(-C*y.*log(1./(1 + exp(-X{1}*beta1-X{2}*beta2-X{3}*beta3)))-(1-y).*log(1-1./(1 + exp(-X{1}*beta1-X{2}*beta2-X{3}*beta3))))+ nu*norm(beta2)^2+(rho/2)*norm(z(m1+1:m1+m2) - beta2,2)^2);
    lambda =1;
    zeta =beta2;
    eta = 10e-7;
    % FISTA
    for iter = 1:MAX_ITER
        ybeta2_old =ybeta2;
        ybeta2 = f(beta2);
        if(abs(ybeta2-ybeta2_old)<TOLERANCE)
            break;
        end
        lambda_old =lambda;
        lambda =(1+sqrt(1+4*lambda^2))/2;
        gamma =(1-lambda_old)/lambda;
        gradient=(X{2})'*((C*y-y+1).*exp(X{1}*beta1+X{2}*beta2+X{3}*beta3)./(1+exp(X{1}*beta1+X{2}*beta2+X{3}*beta3))-C*y)/n;
        zeta_old =zeta;
        zeta = (eta*rho*z(m1+1:m1+m2)+beta2-eta*gradient)/(2*nu*eta+rho*eta+1);
        for j=1:length(pos12)
            product=zeta(pos12(j))*beta1(E12(:,pos12(j))==1);
            if any(product<0)
                zeta(j)=0;
            end
        end
        for j=1:length(pos23)
            product=zeta(pos23(j))*beta3(E23(pos23(j),:)==1);
            if any(product<0)
                zeta(j)=0;
            end
        end       
        beta2 =(1-gamma)*zeta+gamma*zeta_old;
    end
end

function beta3=update_beta3(X,y,beta1,beta2,beta3,nu,rho,z,E13,E23,m1,m2,m3,C)
    % via FISTA
    % global constants and defaults
    n = length(y);
    ybeta3 = 10e10;
    MAX_ITER = 500;
    TOLERANCE =10e-5;
    pos13=find(sum(E13,1)>0);
    pos23=find(sum(E23,1)>0);
    f = @(beta3) (mean(-C*y.*log(1./(1 + exp(-X{1}*beta1-X{2}*beta2-X{3}*beta3)))-(1-y).*log(1-1./(1 + exp(-X{1}*beta1-X{2}*beta2-X{3}*beta3))))+ nu*norm(beta3)^2+(rho/2)*norm(z(m1+m2+1:m1+m2+m3) - beta3,2)^2);
    lambda =1;
    zeta =beta3;
    eta = 10e-7;
    % FISTA
    for iter = 1:MAX_ITER
        ybeta3_old =ybeta3;
        ybeta3 = f(beta3);
        if(abs(ybeta3-ybeta3_old)<TOLERANCE)
            break;
        end
        lambda_old =lambda;
        lambda =(1+sqrt(1+4*lambda^2))/2;
        gamma =(1-lambda_old)/lambda;
        gradient=(X{3})'*((C*y-y+1).*exp(X{1}*beta1+X{2}*beta2+X{3}*beta3)./(1+exp(X{1}*beta1+X{2}*beta2+X{3}*beta3))-C*y)/n;
        zeta_old =zeta;
        zeta = (eta*rho*z(m1+m2+1:m1+m2+m3)+beta3-eta*gradient)/(2*nu*eta+rho*eta+1);
        for j=1:length(pos13)
            product=zeta(pos13(j))*beta1(E13(:,pos13(j))==1);
            if any(product<0)
                zeta(j)=0;
            end
        end
        for j=1:length(pos23)
            product=zeta(pos23(j))*beta2(E23(:,pos23(j))==1);
            if any(product<0)
                zeta(j)=0;
            end
        end
        beta3 =(1-gamma)*zeta+gamma*zeta_old;
    end
end

function z=update_z(beta1,beta2,beta3)
    z=[beta1;beta2;beta3];
end

function obj=objective(nu,X,y,beta1,beta2,beta3,C)

    p=1./( 1 + exp(- X{1}*beta1 - X{2}*beta2 - X{3}*beta3) );
    obj=mean( -C*y.*log(p) - (1-y).*log(1-p)) + nu*( norm(beta1)^2+norm(beta2)^2+norm(beta3)^2);

end


function [opts] = init_options(opts)
    NNTYPE_ORI = 0;
    NNTYPE_SEQ = 1;

    %% Neighbors Type
    if !isfield(opts, 'nntype')
        opts.nntype = NNTYPE_ORI; %  use "0" for original implementation
    end

    disp(opts)

    %% Default values
    DEFAULT_MAX_ITERATION = 1000; % 100
    DEFAULT_TOLERANCE     =  1e-4; % 1e-3;
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