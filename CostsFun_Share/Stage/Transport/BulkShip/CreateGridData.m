%Use latex font as a default
set(0, 'defaulttextinterpreter', 'latex');
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter', 'latex');
set(groot, 'defaultTextInterpreter', 'latex');

clear all;

%% Access to subfolders and contained functions

folder = fileparts(which('main.m'));
addpath(genpath(folder));

%% Load data

NameExcelTable = 'Data.xlsx';
[plant, working, conditioning, electricity, temporaryStorage, ...
    filling_station, isotainer, vesselLoadingStation, truckBatch, truckBulk, ...
    truck, trainBatch, trainBulk, train, bargeBatch, bargeBulk, barge, ...
    shipBatch, shipBulk7barg, shipBulk15barg, pipelineGas, pipelineDense, ...
    storage, finance, buildNetwork, n_work] = OpenData(NameExcelTable);
TickScenarios = {'Optimistic','Average','Conservative'};
n_scen = length(TickScenarios); %number of scenarios: optimistic, average, pessimistic

%% Data

filenameRoussanalyTable = 'Roussanaly_Table4.xlsx';
r = table2array(readtable('Data.xlsx','Sheet','Finance','Range','D2:D2'));
v_km_per_h = table2array(readtable('Data.xlsx','Sheet','ShipBulk','Range','D5:D5'));
tau_y = table2array(readtable('Data.xlsx','Sheet','ShipBulk','Range','D2:D2'));
C_fuel_EUR_per_t = table2array(readtable('Data.xlsx','Sheet','ShipBulk','Range','D3:D3'));
eta_ship = table2array(readtable('Data.xlsx','Sheet','ShipBulk','Range','D4:D4'));
delta_Stor = table2array(readtable('Data.xlsx','Sheet','ShipBulk','Range','D6:D6'));

% m_t_per_y_lin = [1e3 5e3 1e4 3e4 5e4 1e5 2e5 3e5 4e5 5e5 6e5 7e5 8e5 9e5 1e6 1.5e6 2e6 3e6 4e6 5e6 6e6 7e6 8e6 9e6 1e7 1.5e7 2e7];
% d_km_lin = [10:10:100 150:50:2000]; 
m_t_per_y_lin = [1e6 3e6];
d_km_lin = [1168]; 

%% Alg

ShipBulkOpt = ShipBulkCostsforGrid(m_t_per_y_lin, d_km_lin, shipBulk15barg, ...
    finance, filenameRoussanalyTable);

%%save('ShipBulkOptGrid.mat',"ShipBulkOpt")

%% Plot

LC_p8 = reshape(ShipBulkOpt(1).Data.LC_EUR_per_t,[length(d_km_lin) length(m_t_per_y_lin)]);
LC_p16 = reshape(ShipBulkOpt(2).Data.LC_EUR_per_t,[length(d_km_lin) length(m_t_per_y_lin)]);

%LC
figure()

tiledlayout(1,2)

nexttile()

contourf(M,D,LC_p8,[5 10 15 20 25 30 40 50 100 150 200])

nexttile

contourf(M,D,LC_p16,[5 10 15 20 25 30 40 50 100 150 200])


%UC
figure()

tiledlayout(1,2)

nexttile()

contourf(M,D,LC_p8./D,0:0.01:0.1)

nexttile

contourf(M,D,LC_p16./D,0:0.01:0.1)