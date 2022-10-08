function [model, hyperparams, maximize] = getmodel(model_name, solver)

    switch (model_name)
        
        % regression methods
        
        case 'mtreg_lrst'
            [mt_func, hyperpars_space] = mtreg_LRST(solver);
        case 'mtreg_lasso'
            [mt_func, hyperpars_space] = mtreg_Lasso(solver);
        case 'mtreg_l21'
            [mt_func, hyperpars_space] = mtreg_L21(solver);
        case 'mtreg_caso'
            [mt_func, hyperpars_space] = mtreg_cASO(solver);
        case 'mtreg_rmtl'
            [mt_func, hyperpars_space] = mtreg_RMTL(solver);
        case 'mtreg_wmtl'
            [mt_func, hyperpars_space] = mtreg_WMTL(solver);
        case 'mtreg_swmtl0'
            [mt_func, hyperpars_space] = mtreg_SWMTL0(solver);
        case 'mtreg_swmtl1'
            [mt_func, hyperpars_space] = mtreg_SWMTL1(solver);    
        case 'mtreg_swmtl3'
            [mt_func, hyperpars_space] = mtreg_SWMTL3(solver);    
        case 'mtreg_swmtl31'
            [mt_func, hyperpars_space] = mtreg_SWMTL31(solver);            
        case 'mtreg_swmtl32'
            [mt_func, hyperpars_space] = mtreg_SWMTL32(solver);            
        case 'mtreg_swmtl33'
            [mt_func, hyperpars_space] = mtreg_SWMTL33(solver);            

        % classification methods    

        case 'mtclf_lasso'
            [mt_func, hyperpars_space] = mtclf_Lasso(solver);
        case 'mtclf_l21'
            [mt_func, hyperpars_space] = mtclf_L21(solver);
        case 'mtclf_caso'
            [mt_func, hyperpars_space] = mtclf_cASO(solver);
        case 'mtclf_srmtl'
            [mt_func, hyperpars_space] = mtclf_SRMTL(solver);
        case 'mtclf_wmtl1'
            [mt_func, hyperpars_space] = mtclf_WMTL1(solver);  
        case 'mtclf_swmtl3'
            [mt_func, hyperpars_space] = mtclf_SWMTL3(solver);    
        case 'mtclf_swmtl31'
            [mt_func, hyperpars_space] = mtclf_SWMTL31(solver);        
        case 'mtclf_swmtl32'
            [mt_func, hyperpars_space] = mtclf_SWMTL32(solver);               
        case 'mtclf_swmtl33'
            [mt_func, hyperpars_space] = mtclf_SWMTL33(solver);               
        otherwise
            error('method not implemented', model)
    end

    model = mt_func;
    hyperparams = hyperpars_space;

    
    maximize = false;
    if length(strfind(model_name, "mtclf")) > 0
        disp('maximizing')
        maximize = true;
    end

end