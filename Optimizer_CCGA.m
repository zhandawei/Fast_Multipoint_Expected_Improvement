function [best_x,best_y] = Optimizer_CCGA(obj_fun,num_vari,lower_bound,upper_bound,pop_size,max_gen,num_q)
% the internal optimization using CC genetic algorithm+
generation  = 1;
count = 0;
% the initial generation
pop_vari_all = lhsdesign(pop_size, num_vari*num_q).*(repmat(upper_bound,1,num_q) - repmat(lower_bound,1,num_q)) + repmat(lower_bound,1,num_q);
% calculate the objective values
pop_fitness_all = zeros(pop_size,1);
for ii = 1:pop_size
    pop_fitness_all(ii) = feval(obj_fun,pop_vari_all(ii,:));
end
count = count + pop_size;
[fmin,index] = min(pop_fitness_all);
context_vector = pop_vari_all(index,:);
% decompose the problem
pop_vari = cell(1,num_q);
pop_fitness = cell(1,num_q);
for ii = 1:num_q
    pop_vari{ii} = pop_vari_all(:,(ii-1)*num_vari + 1:ii*num_vari);
    pop_fitness{ii} = pop_fitness_all;
end
max_iter = 10;
while generation <= max_gen/max_iter
    for ii = 1: num_q
        for iter  = 1: max_iter
            % parent selection using k-tournament (default k=2) selection
            k = 2;
            temp = randi(pop_size,pop_size,k);
            [~,index] = min(pop_fitness{ii}(temp),[],2);
            pop_parent = pop_vari{ii}(sum(temp.*(index == 1:k),2),:);
            % crossover (simulated binary crossover)
            % dic_c is the distribution index of crossover
            dis_c = 10;
            mu  = rand(pop_size/2,num_vari);
            parent1 = pop_parent(1:2:pop_size,:);
            parent2 = pop_parent(2:2:pop_size,:);
            beta = 1 + 2*min(min(parent1,parent2)-lower_bound,upper_bound-max(parent1,parent2))./max(abs(parent2-parent1),1E-6);
            alpha = 2 - beta.^(-dis_c-1);
            betaq = (alpha.*mu).^(1/(dis_c+1)).*(mu <= 1./alpha) + (1./(2-alpha.*mu)).^(1/(dis_c+1)).*(mu > 1./alpha);
            % the crossover is performed randomly on each variable
            betaq = betaq.*(-1).^randi([0,1],pop_size/2,num_vari);
            betaq(rand(pop_size/2,num_vari)>0.5) = 1;
            offspring1 = 0.5*((1+betaq).*parent1 + (1-betaq).*parent2);
            offspring2 = 0.5*((1-betaq).*parent1 + (1+betaq).*parent2);
            pop_crossover = [offspring1;offspring2];
            % mutation (ploynomial mutation)
            % dis_m is the distribution index of polynomial mutation
            dis_m = 20;
            pro_m = 1/num_vari;
            rand_var = rand(pop_size,num_vari);
            mu  = rand(pop_size,num_vari);
            deta = min(pop_crossover-lower_bound, upper_bound-pop_crossover)./(upper_bound-lower_bound);
            detaq = zeros(pop_size,num_vari);
            position1 = rand_var<=pro_m & mu<=0.5;
            position2 = rand_var<=pro_m & mu>0.5;
            detaq(position1) = ((2*mu(position1) + (1-2*mu(position1)).*(1-deta(position1)).^(dis_m+1)).^(1/(dis_m+1))-1);
            detaq(position2) = (1 - (2*(1-mu(position2))+2*(mu(position2)-0.5).*(1-deta(position2)).^(dis_m+1)).^(1/(dis_m+1)));
            pop_mutation = pop_crossover + detaq.*(upper_bound-lower_bound);
            pop_mutation  = max(min(pop_mutation,upper_bound),lower_bound);
            % fitness calculation
            pop_mutation_fitness = zeros(pop_size,1);
            for jj = 1:pop_size
                temp = context_vector;
                temp(:,(ii-1)*num_vari + 1:ii*num_vari) = pop_mutation(jj,:);
                pop_mutation_fitness(jj) = feval(obj_fun, temp);
            end
            count = count+pop_size;
            if count >= pop_size*num_q*max_gen
                break;
            end
            % environment selection
            pop_vari_iter = [pop_vari{ii};pop_mutation];
            pop_fitness_iter = [pop_fitness{ii};pop_mutation_fitness];
            [~,win_num] = sort(pop_fitness_iter);
            pop_vari{ii} = pop_vari_iter(win_num(1:pop_size),:);
            pop_fitness{ii} = pop_fitness_iter(win_num(1:pop_size),:);
        end
        % update the context vector
        if pop_fitness{ii}(1) <= fmin
            context_vector(:,(ii-1)*num_vari + 1:ii*num_vari) = pop_vari{ii}(1,:);
            fmin = pop_fitness{ii}(1);
        end
    end
    % update generation
    generation = generation + 1;
end
best_x = context_vector;
best_y = fmin;


