%% Trajectory-piecewise linearization and POD based on Jin 2017
% This file creates the ROM making the POD on the statesTraining, and
% choosing the derivatives in the TPWL only from the first training simulation. 
% In the paper, they use multiple derivatives, here just one for simplicity


addpath("TrainingData")
addpath("C:\Users\0954659\OneDrive - Universiteit Utrecht\Documents\MATLAB\tools\Crameri colormap")
%% Perform full-scale simulations if needed

performNewTraining = 1;
do_fullscale4rom = 1;

boundary_condition = 'open'; % open/closed saline aquifer

if performNewTraining
    sim4rom = 0;
    [simData, muco2, WI,rhoc] = basic_example_3D_modified(sim4rom);
end

%% load states and derivatives
readTrainingData
mobApprox = 1/muco2;
c_alpha_co2 = mean(satTraining(cellTopWell, 1:10));
satCo2 = c_alpha_co2;
satWater = 1 - c_alpha_co2;
krWater = fluid.krW(satWater);
krCo2 = fluid.krG(satCo2);
muWater = fluid.muW(150); % viscosity at reference pressure
mobTotal = c_alpha_co2.*(krCo2/muco2+krWater/muWater);

%% Implement Jacobians for rate-controlled wells (only J and B are affected)

for i = 1:nt
    Jtilde{i} = J{i} - B{i} *inv(Jww{i}) * Jwr{i}; 
    Btilde{i} = Jrc{i} - B{i} *inv(Jww{i})* Jwc{i};
end

%% run new reduced-order simulation

modelType = {'ROMBase', 'ROMLin', 'ROMLinP'};
nNewSim = simData.nNewSim;
newSimNames = simData.newSimNames;

pbh_rom = struct();
pTopWell_rom = struct();
index = struct();
runtime = struct();
injRateNewSim = {};
for sim = 1:nNewSim
    injRateNewSim{sim} = simData.inputSchedulesNewSim.(newSimNames{sim}+"_val"); 
end
nt =10;


for sim = 1:nNewSim
    for model = 1:length(modelType)
        %% POD
        % Energy retainated from singular value decomposition
        cutoffPress = 0.999999999999999;
        lp = 105; % 76
        if strcmp(modelType{model}, 'ROMLinP')
            cutoffSat = 1;
            ls =1;
        else
            cutoffSat = 1;
            ls = lp;
        end
        % Perform POD
        [basisPODpress, basisPODsat, lp, ls] = performPOD(statesTraining, cutoffPress, cutoffSat, lp, ls);
        phi = [basisPODpress, zeros(size(basisPODsat)); zeros(size(basisPODpress)) basisPODsat];
    
        % Preallocate matrixes
        ltot = lp + ls;
        gamma = zeros(2*nb, ltot); % constraint reduction matrix
        epsilon = cell(1, nt); % reduced state vector, size ltot 
        xAfterPOD = zeros(2*nb, nt);
        
        % Calculate the reduced jacobians (Petrov-Galerkin projection)
        for i =1:nt
            gamma = J{i} * phi; % constraint reduction matrix (Petrov-Galerkin projection)
            Jred{i} = gamma' * Jtilde{i} * phi;
            Ared{i} = gamma' * A{i} * phi;
            Bred{i} = gamma' * Btilde{i};
            epsilon{i} = phi' * x(:,i);
            xAfterPOD(:,i) = phi * epsilon{i};
        end
        % Reconstruct the snapshot matrix from the POD basis
        x_reconstructed = phi * (phi' * x);
        
        % Check the reconstruction error
        reconstructionError = norm(x - x_reconstructed);
        tolerance = 1e-6;  
        % Check if the error is too large
        % if reconstructionError > tolerance
        %     error('Reconstruction error is too large: %f', reconstructionError);
        % end
        %% Online processing - new simulation
        
        epsilonNew = cell(1, nt); % reduced-state of the new simulation
        uNew = injRateNewSim{sim}(1:nt); 
        xNew = zeros(2*nb, nt); % full state of the new simulation
        invJred = cell(1, nt);
        for i = 1:nt
            epsilonNew{i} = zeros(size(epsilon{1}));
            invJred{i} = inv(Jred{i});
        end
        endepsilonNew{1} = epsilon{1}; % same starting point
        searchRange = 1;
        shift = 0.01; % shift to not have /0 in the denominator of the distance
        weight = 3; % experimental parameter to weight the Dcwi part of the distance. In the paper it is called gamma
        index.(modelType{model}) = zeros(1,nt); %index chosen by the minimun distance search (i in the paper)
        Dtot = NaN(nt-1,nt-1);
        Dz_mat = NaN(nt-1,nt-1);
        Dcwi_mat = NaN(nt-1,nt-1);
        tic();
        epsilonNew{1} = epsilon{1};
        for n = 1:nt-1 % run on the new states (rows of the Dtot matrix)
            for c = 1:nt-1 % run on the training states
                if c <= n+searchRange
                    if c>= n - searchRange
                        if strcmp(modelType{model}, 'ROMLin') 
                            % Linear distance with normalization factor
                            Dz = sum(abs(epsilonNew{n}(ls+1:ltot) - epsilon{c}(ls+1:ltot)))/ (abs(sum(epsilon{c}(ls+1:ltot))) + shift);
                            Dcwi = abs(sum(uNew(1:n)'.*time(1:n)) - sum(u(1:c).*time(1:c)) )/ (sum(u(1:c).*time(1:c)) + shift);
                            Dtot(n, c) = Dz + weight * Dcwi; 
                        elseif strcmp(modelType{model}, 'ROMLinP')
                            % Linear distance with normalization factor
                            Dcwi = abs(sum(uNew(1:n)'.*time(1:n)) - sum(u(1:c).*time(1:c)) )/ (sum(u(1:c).*time(1:c)) + shift);
                            Dtot(n, c) = weight * Dcwi; 
                        elseif strcmp(modelType{model}, 'ROMBase')
                            % Standard distance as in Jin 2018
                            Dz = sum(abs(epsilonNew{n}(ls+1:ltot) - epsilon{c}(1:lp)))/ (abs(sum(epsilonNew{n}(ls+1:ltot))) + shift);
                            Dcwi = abs(sum(uNew(1:n)'.*time(1:n)) - sum(u(1:c).*time(1:c)) ) / (sum(uNew(1:n)'.*time(1:n)) + shift);
                            Dtot(n, c) = Dz + weight * Dcwi;
    
                        end
                        Dz_mat(n, c) = Dz;
                        Dcwi_mat(n, c) = Dcwi;
                        
                    end
                end
            end
            [minDist, index.(modelType{model})(n)] = min(Dtot(n,:), [], "omitnan");
            i = index.(modelType{model})(n);
            if strcmp(modelType{model}, 'ROMLinP') 
                %ONLY PRESSURE or some saturation
                l_linP = lp+ls;
                epsilonNew{n+1}(1:l_linP) = epsilon{i+1}(1:l_linP) - invJred{i+1}(1:l_linP,1:l_linP) * ( Ared{i+1}(1:l_linP,1:l_linP) * (epsilonNew{n}(1:l_linP) - epsilon{i}(1:l_linP)) + Bred{i+1}(1:l_linP) * (uNew(n + 1) - u(i+1)) );
            else 
                epsilonNew{n+1} = epsilon{i+1} - invJred{i+1} * ( Ared{i+1} * (epsilonNew{n} - epsilon{i}) + Bred{i+1} * (uNew(n + 1) - u(i+1)) );
            end 
    
            xNew(:,n) = phi * epsilonNew{n}; % rewrite in the original basis

        end
            if strcmp(modelType{model}, 'ROMLinP') 
                runtime.(modelType{model}).(newSimNames{sim}) = toc();
        end
        xNew(:,nt) = phi * epsilonNew{nt}; % rewrite the last timestep in the old basis
        pTopWell_rom.(newSimNames{sim}).(modelType{model}) = xNew(cellTopWell, :);
        pbh_rom.(newSimNames{sim}).(modelType{model}) = xNew(cellTopWell, :) + uNew(:)'./(WI.*mobTotal)/convert2bar./c_alpha_co2;
        
    end
end

lp
ls


%% run full scale simulation with the ROM schedule for comparison
% TODO implement multiple simulations possible
if do_fullscale4rom
    sim4rom = 1;
    basic_example_3D_modified(sim4rom)
end
pTraining_topWell = pressureTraining(cellTopWell,:);
pbh_training = pTraining_topWell + u/(WI*mobApprox)/convert2bar;
load('pbh_fullScaleNewSim.mat')
for s = 1:nNewSim
    pbh_fullScaleNewSim.(newSimNames{s}).pTopWell = pbh_fullScaleNewSim.(newSimNames{s}).pTopWell/convert2bar;
    % pbh_fullScaleNewSim.(newSimNames{s}).("pbhApprox") = pbh_fullScaleNewSim.(newSimNames{s}).pTopWell/convert2bar + injRateNewSim{s}/(WI*mobApprox)/convert2bar;
    pbh_fullScaleNewSim.(newSimNames{s}).("pbhApprox") = pbh_fullScaleNewSim.(newSimNames{s}).pTopWell + injRateNewSim{s}./(WI.*mobTotal)/convert2bar./c_alpha_co2;
end


%% Visualize bhp
seconds1year = 3600*24*365;% Convert seconds to years
convertInjRate = 3600*24*rhoc/1000; % convert m3/s to t/day

% Choose a color scheme for BHP and Injection rate plots
colors_bhp = crameri('batlow', 5);  % Color scheme for BHP


%% Plot for multiple injection rates----------------------------------------------------------------------------------------------------

t_end = 10;
rowsSubplot = round(nNewSim/2) + 1;
labelInjRate = {'Faster ramp-up','Variable inj.','Lower inj.','Higher inj.'};
batlow_colors = crameri("batlow", 4);
batlow_colors(4,:) = [0.83, 0.57, 0.72];
% Define font size and font type for readability
fontSize = 9;
fontName = 'Arial';



%% multiple inj rate plotting pTopWell
% Define a layout to handle spacing
figure()
t = tiledlayout(rowsSubplot, 2, 'Padding', 'compact', 'TileSpacing', 'compact'); % Use tiled layout
set(t, 'TileSpacing', 'compact', 'Padding', 'compact'); % Fine-tune spacing for the layout

% First subplot: Injection rates plot
nexttile([1 2]); % This occupies the top row
hold on
h1 = plot(time(1:t_end)/seconds1year, u(1:t_end)*convertInjRate, 'LineStyle', ":", 'LineWidth', 1.5, "Color", "k", 'DisplayName','Training inj. rate');
box on
for i = 1:nNewSim
    plot(time(1:t_end)/seconds1year, injRateNewSim{i}(1:t_end)*convertInjRate, 'LineWidth', 1.5, 'Color', batlow_colors(i, :), 'DisplayName', "New inj. rates");
    % Adjust the position of text labels
    if i == 1
        text(time(1)/seconds1year*1.1, injRateNewSim{i}(2)*convertInjRate*0.8, labelInjRate{i}, 'Color', batlow_colors(i, :), 'FontSize', fontSize, 'FontName', fontName);
    elseif i == 2
        text(time(2)/seconds1year*0.7, injRateNewSim{i}(2)*convertInjRate*1.15, labelInjRate{i}, 'Color', batlow_colors(i, :), 'FontSize', fontSize, 'FontName', fontName);
    elseif i == 3
        text(time(3)/seconds1year*1.1, injRateNewSim{i}(3)*convertInjRate*1.1, labelInjRate{i}, 'Color', batlow_colors(i, :), 'FontSize', fontSize, 'FontName', fontName);
    elseif i == 4
        text(time(2)/seconds1year*1.35, injRateNewSim{i}(1)*convertInjRate*1.35, labelInjRate{i}, 'Color', batlow_colors(i, :), 'FontSize', fontSize, 'FontName', fontName);
    end
end

% Dummy empty plot for the legend
h2_legend_black = plot(nan, nan, 'LineWidth', 1.5, 'Color', 'k', 'DisplayName', 'New inj. rate');
legend([h1, h2_legend_black], 'Location', 'SouthEast', 'FontSize', fontSize, 'FontName', fontName);ylabel('Inj. rate [t/day]', 'FontSize', fontSize, 'FontName', fontName);  
ylim([0 5000]);
xlim([time(1)/seconds1year time(t_end)/seconds1year]);
xlabel('Time [y]', 'FontSize', fontSize, 'FontName', fontName);

% Other subplots: Bottomhole pressure (BHP) plot
for i = 1:nNewSim
    nexttile;
    hold on
    plot(time(1:t_end)/seconds1year, pbh_fullScaleNewSim.(newSimNames{i}).pTopWell(1:t_end), 'LineWidth', 1.5, 'LineStyle', "-.", 'Color', batlow_colors(i, :));
    plot(time(1:t_end)/seconds1year, pTopWell_rom.(newSimNames{i}).ROMLinP(1:t_end), 'LineWidth', 1.5, 'Color', batlow_colors(i, :));

    if mod(i, 2) ~= 0
        ylabel('p_{well} [bar]', 'FontSize', fontSize, 'FontName', fontName);
    end

    if i == 3 || i ==4 
        xlabel('Time [y]', 'FontSize', fontSize, 'FontName', fontName);
    end

    text(time(1)/seconds1year*1.3, 175, labelInjRate{i}, 'Color', batlow_colors(i, :), 'FontSize', fontSize, 'FontName', fontName);

    ylim([150 180]);
    xlim([time(1)/seconds1year time(t_end)/seconds1year]);
    box on
end

% Create the second legend outside the subplots
h_legend_high_fid_sim = plot(nan, nan, 'LineWidth', 1.5, 'LineStyle', "-.", 'Color', 'k', 'DisplayName', 'Full-scale sim.');
h_legend_lin_model = plot(nan, nan, 'LineWidth', 1.5, 'Color', 'k', 'DisplayName', 'Linear model');

% Add the second legend to the layout
lgd = legend([h_legend_high_fid_sim, h_legend_lin_model], 'Orientation', 'horizontal');
lgd.Layout.Tile = 'south'; % This places the legend below all tiles
lgd.FontSize = fontSize;
lgd.FontName = fontName;

% Save the figure
figureName = "bhp_multiple_inj_validation";
saveFigureForPaper(gcf, figureName, 9);



%% Parity plot pTopWell
pTopWellLinAll = []; 
pTopWellNewSimAll = [];
for i =1:nNewSim
    pTopWellNewSimAll= [pTopWellNewSimAll; pbh_fullScaleNewSim.(newSimNames{i}).pTopWell(1:t_end)];
    pTopWellLinAll = [pTopWellLinAll; pTopWell_rom.(newSimNames{i}).ROMLinP(1:t_end)'];
end


minPtopWellLinAll = min([pTopWellNewSimAll; pTopWellLinAll]);
minPtopWellNewSimAll= max([pTopWellNewSimAll; pTopWellLinAll]);
plusMinusLine = 0.05;
zoomOut = 25;

figure()
hold on;
box on
h_perfect_match = plot([(minPtopWellLinAll-zoomOut) (minPtopWellNewSimAll+zoomOut)], [(minPtopWellLinAll-zoomOut) (minPtopWellNewSimAll+zoomOut)], 'k-', 'LineWidth', 1.5, 'DisplayName','Perfect match');

h_dash_plus = plot([(minPtopWellLinAll-zoomOut) (minPtopWellNewSimAll+zoomOut)], [(minPtopWellLinAll-zoomOut)*(1+plusMinusLine) (minPtopWellNewSimAll+zoomOut)*(1+plusMinusLine)], 'k--', 'LineWidth', 1.2);  % +10% line
h_dash_minus = plot([(minPtopWellLinAll-zoomOut) (minPtopWellNewSimAll+zoomOut)], [(minPtopWellLinAll-zoomOut)*(1-plusMinusLine) (minPtopWellNewSimAll+zoomOut)*(1-plusMinusLine)], 'k--', 'LineWidth', 1.2);  % -10% line

h = [];
% Scatter plot of predicted vs. observed values
for i = 1:nNewSim
    % normalizedNewSim = pbh_fullScaleNewSim.(newSimNames{i}).pbhApprox(1:t_end)/normFactor;
    % normalizedLinModel = pbh_rom.(newSimNames{i}).ROMLinP(1:t_end)/normFactor;
    %scatter(normalizedNewSim, normalizedLinModel, 52, batlow_colors(i, :), 'filled'); % Change marker size and color as needed
    h(i) = scatter(pbh_fullScaleNewSim.(newSimNames{i}).pTopWell(1:t_end), pTopWell_rom.(newSimNames{i}).ROMLinP(1:t_end), 52, batlow_colors(i, :), 'filled', 'DisplayName',  labelInjRate{i}); % Change marker size and color as needed
    hold on
end
text(((minPtopWellLinAll-zoomOut)+(minPtopWellNewSimAll+zoomOut))/2-25, ((minPtopWellLinAll-zoomOut)*(1+plusMinusLine)+(minPtopWellNewSimAll+zoomOut)*(1+plusMinusLine))/2-15, "+5%", 'FontSize', fontSize, 'FontName', fontName )
text(((minPtopWellLinAll-zoomOut)+(minPtopWellNewSimAll+zoomOut))/2-13, ((minPtopWellLinAll-zoomOut)*(1-plusMinusLine)+(minPtopWellNewSimAll+zoomOut)*(1-plusMinusLine))/2-15, "-5%", 'FontSize', fontSize, 'FontName', fontName )
legend([h_perfect_match, h],'Location','southeast')
xlabel('Full-scale sim. (p_{well} [bar])');
ylabel('Linear model (p_{well} [bar])');
xlim([minPtopWellLinAll-zoomOut minPtopWellNewSimAll+zoomOut]);
ylim([minPtopWellLinAll-zoomOut minPtopWellNewSimAll+zoomOut]);
hold off;

figureName = "parity_plot";
saveFigureForPaper(gcf, figureName);

%% plot bhp
% Define a layout to handle spacing
figure()
t = tiledlayout(rowsSubplot, 2, 'Padding', 'compact', 'TileSpacing', 'compact'); % Use tiled layout
set(t, 'TileSpacing', 'compact', 'Padding', 'compact'); % Fine-tune spacing for the layout

% First subplot: Injection rates plot
nexttile([1 2]); % This occupies the top row
hold on
h1 = plot(time(1:t_end)/seconds1year, u(1:t_end)*convertInjRate, 'LineStyle', ":", 'LineWidth', 1.5, "Color", "k", 'DisplayName','Training Injection Rate');
box on
for i = 1:nNewSim
    plot(time(1:t_end)/seconds1year, injRateNewSim{i}(1:t_end)*convertInjRate, 'LineWidth', 1.5, 'Color', batlow_colors(i, :), 'DisplayName', "New injection rates");
    % Adjust the position of text labels
    if i == 1
        text(time(1)/seconds1year*1.1, injRateNewSim{i}(2)*convertInjRate*0.8, labelInjRate{i}, 'Color', batlow_colors(i, :), 'FontSize', fontSize, 'FontName', fontName);
    elseif i == 2
        text(time(2)/seconds1year*0.7, injRateNewSim{i}(2)*convertInjRate*1.1, labelInjRate{i}, 'Color', batlow_colors(i, :), 'FontSize', fontSize, 'FontName', fontName);
    elseif i == 3
        text(time(3)/seconds1year*1.1, injRateNewSim{i}(3)*convertInjRate*1.1, labelInjRate{i}, 'Color', batlow_colors(i, :), 'FontSize', fontSize, 'FontName', fontName);
    elseif i == 4
        text(time(1)/seconds1year*1.1, injRateNewSim{i}(1)*convertInjRate*0.92, labelInjRate{i}, 'Color', batlow_colors(i, :), 'FontSize', fontSize, 'FontName', fontName);
    end
end

% Dummy empty plot for the legend
h2_legend_black = plot(nan, nan, 'LineWidth', 1.5, 'Color', 'k', 'DisplayName', 'New injection rate');
legend([h1, h2_legend_black], 'Location', 'SouthEast', 'FontSize', fontSize, 'FontName', fontName);ylabel('Inj. rate [t/day]', 'FontSize', fontSize, 'FontName', fontName);  
ylim([0 5000]);
xlim([time(1)/seconds1year time(t_end)/seconds1year]);
xlabel('Time [y]', 'FontSize', fontSize, 'FontName', fontName);

% Other subplots: Bottomhole pressure (BHP) plot
for i = 1:nNewSim
    nexttile;
    hold on
    plot(time(1:t_end)/seconds1year, pbh_fullScaleNewSim.(newSimNames{i}).pReal(1:t_end)/convert2bar, 'LineWidth', 1.5, 'LineStyle', "-.", 'Color', batlow_colors(i, :));
    plot(time(1:t_end)/seconds1year, pbh_rom.(newSimNames{i}).ROMLinP(1:t_end), 'LineWidth', 1.5, 'Color', batlow_colors(i, :));
    xlabel('Time [y]', 'FontSize', fontSize, 'FontName', fontName);

    if mod(i, 2) ~= 0
        ylabel('p_{bh} [bar]', 'FontSize', fontSize, 'FontName', fontName);
    end
    text(time(1)/seconds1year*1.3, 195, labelInjRate{i}, 'Color', batlow_colors(i, :), 'FontSize', fontSize, 'FontName', fontName);

    xlim([time(1)/seconds1year time(t_end)/seconds1year]);
    box on
end

% Create the second legend outside the subplots
h_legend_high_fid_sim = plot(nan, nan, 'LineWidth', 1.5, 'LineStyle', "-.", 'Color', 'k', 'DisplayName', 'High-fidelity sim.');
h_legend_lin_model = plot(nan, nan, 'LineWidth', 1.5, 'Color', 'k', 'DisplayName', 'Linear model');

% Add the second legend to the layout
lgd = legend([h_legend_high_fid_sim, h_legend_lin_model], 'Orientation', 'horizontal');
lgd.Layout.Tile = 'south'; % This places the legend below all tiles
lgd.FontSize = fontSize;
lgd.FontName = fontName;


%% Parity plot bhp
pbhLinAll = []; 
pbhNewSimAll = [];
for i =1:nNewSim
    pbhLinAll = [pbhLinAll; pbh_rom.(newSimNames{i}).ROMLinP(1:t_end)'];
    pbhNewSimAll= [pbhNewSimAll; pbh_fullScaleNewSim.(newSimNames{i}).pbhApprox(1:t_end)];
end


minpbhLinAll = min([pbhNewSimAll; pbhLinAll]);
minpbhNewSimAll= max([pbhNewSimAll; pbhLinAll]);
plusMinusLine = 0.05;

figure()
hold on;
box on
h_perfect_match = plot([(minpbhLinAll-5) (minpbhNewSimAll+5)], [(minpbhLinAll-5) (minpbhNewSimAll+5)], 'k-', 'LineWidth', 1.5, 'DisplayName','Perfect match');

h_dash_plus = plot([(minpbhLinAll-5) (minpbhNewSimAll+5)], [(minpbhLinAll-5)*(1+plusMinusLine) (minpbhNewSimAll+5)*(1+plusMinusLine)], 'k--', 'LineWidth', 1.2);  % +10% line
h_dash_minus = plot([(minpbhLinAll-5) (minpbhNewSimAll+5)], [(minpbhLinAll-5)*(1-plusMinusLine) (minpbhNewSimAll+5)*(1-plusMinusLine)], 'k--', 'LineWidth', 1.2);  % -10% line

h = [];
% Scatter plot of predicted vs. observed values
for i = 1:nNewSim
    % normalizedNewSim = pbh_fullScaleNewSim.(newSimNames{i}).pbhApprox(1:t_end)/normFactor;
    % normalizedLinModel = pbh_rom.(newSimNames{i}).ROMLinP(1:t_end)/normFactor;
    %scatter(normalizedNewSim, normalizedLinModel, 52, batlow_colors(i, :), 'filled'); % Change marker size and color as needed
    h(i) = scatter(pbh_fullScaleNewSim.(newSimNames{i}).pbhApprox(1:t_end), pbh_rom.(newSimNames{i}).ROMLinP(1:t_end), 52, batlow_colors(i, :), 'filled', 'DisplayName',  labelInjRate{i}); % Change marker size and color as needed
    hold on
end
text(((minpbhLinAll-5)+(minpbhNewSimAll+5))/2-5, ((minpbhLinAll-5)*(1+plusMinusLine)+(minpbhNewSimAll+5)*(1+plusMinusLine))/2, "+5%", 'FontSize', fontSize, 'FontName', fontName )
text(((minpbhLinAll-5)+(minpbhNewSimAll+5))/2, ((minpbhLinAll-5)*(1-plusMinusLine)+(minpbhNewSimAll+5)*(1-plusMinusLine))/2-2, "-5%", 'FontSize', fontSize, 'FontName', fontName )
legend([h_perfect_match, h],'Location','southeast')
xlabel('Full-scale sim. (p_{bh} [bar])');
ylabel('Linear model (p_{bh} [bar])');
xlim([minpbhLinAll-5 minpbhNewSimAll+5]);
ylim([minpbhLinAll-5 minpbhNewSimAll+5]);
title('bhp Approx')
hold off;


%% Save variables for Pyhub implementation
% convert to matrix first
[numRows, numCols] = size(epsilon{1});
epsilon_mat = zeros(nt, numRows, numCols);

[numRows, numCols] = size(invJred{1});
invJred_mat = zeros(nt, numRows, numCols);

[numRows, numCols] = size(Ared{1});
Ared_mat = zeros(nt, numRows, numCols);

[numRows, numCols] = size(Bred{1});
Bred_mat = zeros(nt, numRows, numCols);

for t =1:nt
    epsilon_mat(t, :) = epsilon{t};
    invJred_mat(t, :, :) = invJred{t};
    Ared_mat(t, :, :) = Ared{t};
    Bred_mat(t, :, :) = Bred{t};
end

abs_epsilon = abs(epsilon_mat);

filePath = "C:\Users\0954659\OneDrive - Universiteit Utrecht\Documents\AdOpT-NET0_luca\adopt_net0\data\technology_data\Sink\SalineAquifer_data\matrices_for_ROM";
save(filePath, 'epsilon_mat', 'invJred_mat', "Ared_mat", "Bred_mat", "u","uNew", 'weight', "phi", "ltot", "lp", "ls", "cellTopWell","pTraining_topWell", "abs_epsilon", "WI", "muco2", "mobTotal", "c_alpha_co2")


% %% Choose colors
% % Load the Crameri batlow colormap with 5 colors
% batlow_colors = crameri('batlow', 4);
% 
% % Create a new figure
% figure;
% hold on;
% 
% % Number of colors
% num_colors = size(batlow_colors, 1);
% 
% % Loop through colors and plot each one as a separate patch
% for i = 1:num_colors
%     % Plot each color as a rectangle
%     rectangle('Position', [i, 0, 1, 1], 'FaceColor', batlow_colors(i, :), 'EdgeColor', 'none');
% end
% 
% % Set axis properties for visualization
% xlim([1 num_colors+1]);
% ylim([0 1]);
% set(gca, 'XTick', [], 'YTick', [], 'Color', 'w');
% title('Crameri Batlow Color Palette with 5 Colors');
% 
% hold off;


%% Functions

function saveErrors(error, boundary_condition)
    % Define the path to the Excel file
    file_path = 'Results/Errors.xlsx';
    
    % Extract the first three errors
    error_values = struct2cell(error);  % Convert struct to cell array
    
    % Determine cell locations based on boundary condition
    if strcmp(boundary_condition, 'open')
        cells = {'C2', 'C3', 'C4'};
    else
        cells = {'C5', 'C6', 'C7'};
    end
    
    % Loop through the first three errors and write them to the corresponding cells
    for i = 1:3
        writematrix(error_values{i}, file_path, 'Sheet', 'Sheet1', 'Range', cells{i});
    end
end






