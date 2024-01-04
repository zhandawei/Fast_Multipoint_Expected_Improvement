% The matlab code of the fast multipoint expected improvement
% Dawei Zhan, Yun Meng and Huanlai Xing. A Fast Multi-Point Expected
% Improvement for Parallel Expesive Optimization. IEEE Transactions on
% Evolutioanry Compuatation, 2022, DOI: 10.1109/TEVC.2022.3168060
clearvars; close all;
% setting of the problem
fun_name = 'Rosenbrock';
num_vari = 50;
lower_bound = -2.048*ones(1,num_vari);
upper_bound = 2.048*ones(1,num_vari);
% the number of initial design points
num_initial = 200;
% maximum number of evaluations
max_evaluation = 360;
% the number of points selected in each iteration
num_q = 8;
% initial design points using Latin hypercube sampling method
sample_x = lower_bound + (upper_bound-lower_bound).*lhsdesign(num_initial,num_vari,'criterion','maximin','iterations',1000);
sample_y = feval(fun_name,sample_x);
% the current iteration and evaluation
evaluation = size(sample_x,1);
iteration = 0;
% the current best solution
fmin = min(sample_y);
% print the current information to the screen
fprintf('EGO-CCGA-FqEI on %d-D %s function, iteration: %d, evaluation: %d, current best solution: %.2f\n',num_vari,fun_name,iteration,evaluation,fmin);
% the iteration
while evaluation < max_evaluation
    % build (or rebuild) the initial Kriging model
    kriging_model = Kriging_Train(sample_x,sample_y,lower_bound,upper_bound,1*ones(1,num_vari),0.001*ones(1,num_vari),1000*ones(1,num_vari));
    % maximize the qEI function
    [best_x,max_EI]= Optimizer_CCGA(@(x)-Infill_FqEI(x,kriging_model,fmin),num_vari,lower_bound,upper_bound,num_vari,100,num_q);
    infill_x = reshape(best_x,num_vari,[])';
    % evaluating the candidate with the real function
    infill_y = feval(fun_name,infill_x);
    % add the new point to design set
    sample_x = [sample_x;infill_x];
    sample_y = [sample_y;infill_y];
    % updating some parameters
    evaluation = size(sample_x,1);
    iteration = iteration + 1;
    fmin = min(sample_y);
    % print the current information to the screen
    fprintf('EGO-CCGA-FqEI on %d-D %s function, iteration: %d, evaluation: %d, current best solution: %.2f\n',num_vari,fun_name,iteration,evaluation,fmin);
end




