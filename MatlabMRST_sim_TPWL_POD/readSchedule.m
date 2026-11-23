%% Read schdule from Excel file
% function that reads and set the schedule of injection from a dedicated 
% Excel file

function [schedule] = readSchedule(simData, bc, W)

nt = simData.nt;
dt = simData.dt;
nSim = simData.nSim; 
nControls = simData.nControls; 
simNames = simData.simNames;
inputSchedules = simData.inputSchedules;

for i =1:nSim
    simNames(i) = "sim" + num2str(i);
    schedule.(simNames(i)).step.val =  ones(nt,1)*dt*year;
    schedule.(simNames{i}).step.control = table2array(inputSchedules(:, simNames{i} + "_control"));
    for cntl = 1: nControls
        % Setting up two copies of the well and boundary specifications. 
        schedule.(simNames(i)).control(cntl) = struct('W', W, 'bc', bc);
        schedule.(simNames(i)).control(cntl).W.val = table2array(inputSchedules(i, "Controls"));
    end
end







