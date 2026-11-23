%% Create a schedule from simData

function [schedule] = createSchedule(simData, bc, W)

nt = simData.nt;
dt = simData.dt;
nTraining = simData.nTraining; 
nNewSim = simData.nNewSim; 
nControls = simData.nControls; 
trainingNames = simData.trainingNames;
newSimNames = simData.newSimNames;
inputSchedulesTraining = simData.inputSchedulesTraining;
inputSchedulesNewSim = simData.inputSchedulesNewSim;
injectionRates = simData.injectionRates;
ntTraining = height(inputSchedulesTraining);
ntNewSim = height(inputSchedulesNewSim);


for i =1:nTraining
    schedule.(trainingNames(i)).step.val =  ones(ntTraining,1)*dt*year;
    schedule.(trainingNames(i)).step.control = table2array(inputSchedulesTraining(:, trainingNames{i} + "_control"))';
    for cntl = 1: nControls
        % Setting up two copies of the well and boundary specifications. 
        schedule.(trainingNames(i)).control(cntl) = struct('W', W, 'bc', bc);
        schedule.(trainingNames(i)).control(cntl).W.val = injectionRates(cntl);
    end
end


for i =1:nNewSim
    schedule.(newSimNames(i)).step.val =  ones(ntNewSim,1)*dt*year;
    schedule.(newSimNames(i)).step.control = table2array(inputSchedulesNewSim(:, newSimNames{i} + "_control"))';
    for cntl = 1: nControls
        % Setting up two copies of the well and boundary specifications. 
        schedule.(newSimNames(i)).control(cntl) = struct('W', W, 'bc', bc);
        schedule.(newSimNames(i)).control(cntl).W.val = injectionRates(cntl);
    end
end


% schedule.sim4rom.step.val =  ones(nt,1)*dt*year;
% schedule.sim4rom.step.control = simData.inputScheduleROM(:,1);
% for cntl = 1: nControls
%     % Setting up two copies of the well and boundary specifications. 
%     schedule.sim4rom.control(cntl) = struct('W', W, 'bc', bc);
%     schedule.sim4rom.control(cntl).W.val = injectionRates(cntl);
% end
% 




