

function [dataset_path, results_path, model, solver, min_train_size, max_train_size, iter, opts_path, test_dataset_path] = process_args( arg_list )

    %arg_list = argv ()
    num_args = size(arg_list, 1);
    opts_path = "";
    test_dataset_path="";
    
    if  num_args > 1
        fprintf('arguments %d \n',size(arg_list,1))
        dataset_path = arg_list{1};
        results_path = arg_list{2};
        model = arg_list{3};
        solver = arg_list{4};       
        min_train_size = str2num(arg_list{5}); 
        max_train_size = str2num(arg_list{6}); 
        iter = str2num(arg_list{7});

        if num_args > 7
            opts_path = arg_list{8};
        end
        if num_args > 8
            test_dataset_path = arg_list{9};
        end
    else        
        dataset_path = '~/data/multitask/school/school.mat'
        results_path = '~/data/multitask/school/results/L21'
        iter = 1;
        model = 'mtreg_Lasso'
        min_train_size = 0.5;
        max_train_size = 0.5;
        solver = 'grid_small'
    end    

    

    %results_path = strcat(results_path, '/test', num2str(test_size*100) , '_i', num2str(iter) , '_'  );
    %results_path = strcat(results_path, '/test', num2str(test_size*100)   );
    %results_path = strcat(results_path, '/test' );
    results_path = strcat(results_path, '/' );
end