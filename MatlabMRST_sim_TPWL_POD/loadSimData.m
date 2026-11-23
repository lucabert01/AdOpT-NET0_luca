%% Import simulation data and schedules from excel to struct inputSchedule

function [simData] = loadSimData()


% Read the Excel file into a cell array

data = readcell('injectionSchedules.xlsx');

% Initialize a cell array to store non-empty data
non_empty_data = cell(size(data));

% Iterate over each column
for col = 1:size(data, 2)
    % Get the column title
    column_title = data{1, col};
    % Find non-empty cells in the column (excluding the title)
    non_empty_cells = data(2:end, col);
    non_empty_cells = non_empty_cells(~cellfun(@isempty, non_empty_cells));
    % Add column title and non-empty cells to the new cell array
    non_empty_data{1, col} = column_title;
    non_empty_data(2:numel(non_empty_cells)+1, col) = non_empty_cells;
end

% Convert the cell array to a table
varNames = cellstr(non_empty_data(1, :));
T = cell2table(non_empty_data(2:end, :), 'VariableNames', non_empty_data(1, :));
simData.nt = table2array(T(1, "num_timesteps"));
simData.dt = table2array(T(1, "timestep_length [years]"));
simData.nTraining = sum(contains(varNames, 'training'))/2;
simData.nControls = sum(~isnan(table2array(T(:, "controls"))));
simData.injectionRates = table2array(T(:, "controls")); % vales associate for each control number
simData.trainingNames = string(ones(simData.nTraining,1));
for i =1:simData.nTraining
    simData.trainingNames(i) = "training" + num2str(i);
end

%save the simulation schedules as table
simData.nNewSim = sum(contains(varNames, 'newSim'))/2;
firstNewSimCol = find(contains(varNames, 'newSim'), 1);
for i =1:simData.nNewSim
    simData.newSimNames(i) = "newSim" + num2str(i);
end

firstTrainingCol = find(contains(varNames, 'training'), 1);
% Replace NaN elements in the table with empty cells
simData.inputSchedulesTraining = T(:, firstTrainingCol:firstNewSimCol);
simData.inputSchedulesNewSim = T(:, firstNewSimCol:end);
simData.inputSchedulesNewSim = rmmissing(simData.inputSchedulesNewSim);



a = 1;




