% Define the path to the "TrainingData" folder
trainingDataFolder = 'TrainingData';

convert2bar = 10^5;

% Load the variables from the files in the "TrainingData" folder
%load(fullfile(trainingDataFolder, 'statesNew.mat')); % statesNew are the ones obtained from the full-scale simulation with the new injection rate (used for comparison with the ROM results) 
load(fullfile(trainingDataFolder, 'statesTraining.mat'));
load(fullfile(trainingDataFolder, 'wellSol.mat'));
load(fullfile(trainingDataFolder, 'schedule.mat'));
load(fullfile(trainingDataFolder, 'jacForward.mat'));
load(fullfile(trainingDataFolder, 'jacReverse.mat'));
load(fullfile(trainingDataFolder, 'cellTopWell.mat'));
load(fullfile(trainingDataFolder, 'muco2.mat'));
load(fullfile(trainingDataFolder, 'WI.mat'));
load(fullfile(trainingDataFolder, 'rhoc.mat'));
load(fullfile(trainingDataFolder, 'fluid.mat'));
simData = loadSimData();


time = cumsum(schedule.training1.step.val)';
statesTraining = states;
nTraining = simData.nTraining;
nt = simData.nt; % number of timesteps per sim
nb = length(statesTraining{1}{1,1}.pressure); % number of blocks in the grid
nu = 1; % number of injection wells
pressureTraining = zeros(nb, nt);
satTraining = zeros(nb, nt);
pressureNewFSS = zeros(nb, nt); % pressure of the new Full-Scale Simulation

for i = 1:nt
        pressureTraining(:,i) = statesTraining{1}{i}.pressure/convert2bar;
        satTraining(:,i) = statesTraining{1}{i}.s(:,2); 
%        pressureFullScaleSim(:,i) = statesNew{1}{i}.pressure;

end
x = [pressureTraining; satTraining]; % rows are gridblocks, columns timesteps

% Initialize cell arrays
J = cell(1, nt);
A = cell(1, nt);
B = cell(1, nt);
Jred = cell(1, nt); % reduced Jacobians
Ared = cell(1, nt);
Bred = cell(1, nt);
Jww = cell(1, nt);
Jwr = cell(1, nt);
Jwc = cell(1, nt);
Jrc = cell(1, nt);
Jtilde = cell(1, nt); % Jacobians adjusted for rate-based simulations
Btilde = cell(1, nt);
u = zeros(1, nt);


res_ix = 1:(2*nb);
well_ix = (2*nb+1):(2*nb + 2);
ctrl_ix = 2*nb + 3;

res_ix_var = 1:(2*nb);
well_ix_var = [2*nb+1, 2*nb+3];
ctrl_ix_var = 2*nb + 2;

for i = 1:nt
    A{i} = jacReverse{i}(1:2*nb, 1:2*nb);
    A{i}(1:2*nb, 1:nb) = jacReverse{i}(1:2*nb, 1:nb)*convert2bar;% convert the pressure columns
    J{i} = jacForward{i}(1:2*nb, 1:2*nb); % same as Jrr
    J{i}(1:2*nb, 1:nb) = jacForward{i}(1:2*nb, 1:nb)*convert2bar; % convert the pressure columns
    Jrc{i} = jacForward{i}(res_ix, ctrl_ix_var);
    B{i} = jacForward{i}(res_ix, well_ix_var)*convert2bar; % same as Jrw
    Jwr{i} = jacForward{i}(well_ix, res_ix_var)*convert2bar;
    Jwc{i} = jacForward{i}(well_ix, ctrl_ix_var);
    Jww{i} = jacForward{i}(well_ix, well_ix_var)*convert2bar;
    u(i) = wellSol{1}{i, 1}.val;
end



% Old code, before implementing Jacobains according to olav
% for i = 1:nt
%     A{i} = jacReverse{i}(1:2*nb, 1:2*nb);
%     A{i}(1:2*nb, 1:nb) = jacReverse{i}(1:2*nb, 1:nb)*convert2bar;% convert the pressure columns
%     J{i} = jacForward{i}(1:2*nb, 1:2*nb); % same as Jrr
%     J{i}(1:2*nb, 1:nb) = jacForward{i}(1:2*nb, 1:nb)*convert2bar; % convert the pressure columns
%     Jrc{i} = jacForward{i}(1:2*nb, 2*nb+2);
%     B{i} = jacForward{i}(1:2*nb, 2*nb+3)*convert2bar; % same as Jrw
%     Jwr{i} = jacForward{i}(2*nb+3, 1:2*nb)*convert2bar;
%     Jwc{i} = jacForward{i}(2*nb+2, 2*nb+3);
%     Jww{i} = jacForward{i}(2*nb+3, 2*nb+3)*convert2bar;
%     u(i) = wellSol{1}{i, 1}.val;
% end


