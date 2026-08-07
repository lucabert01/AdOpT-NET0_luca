%Use latex font as a default
set(0, 'defaulttextinterpreter', 'latex');
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter', 'latex');
set(groot, 'defaultTextInterpreter', 'latex');

clear all;

%% Access to subfolders and contained functions

folder = fileparts(which('main.m'));
addpath(genpath(folder));

%% Constants

NameExcelTable = 'Data.xlsx';
[plant, working, conditioning, electricity, temporaryStorage, ...
    filling_station, isotainer, vesselLoadingStation, truckBatch, truckBulk, ...
    truck, trainBatch, trainBulk, train, bargeBatch, bargeBulk, barge, ...
    shipBatch, shipBulk7barg, shipBulk15barg, pipelineGas, pipelineDense, ...
    storage, finance, buildNetwork, n_work] = OpenData(NameExcelTable);
pipeline = pipelineGas;

% DACE_2010_2015 = 100/93;
% ChemieTechnik_Montageleistung_2015_2021 = 114.2/100;
% ChemieTechnik_PCD_2015_2021 = 111.3/100;
% r = table2array(readtable('Data.xlsx','Sheet','Finance','Range','D2:D2'));
%
% SteelFactor = 350.9/191.7;
% Clab_EUR_per_m2 = 825.*DACE_2010_2015.*ChemieTechnik_Montageleistung_2015_2021;
% CROW_EUR_per_m = [0 83].*DACE_2010_2015.*ChemieTechnik_PCD_2015_2021; %offshore and onshore ROW cost
% C_el_EUR_per_MWh = table2array(readtable('Data.xlsx','Sheet','Electricity','Range','G2:G2')).*1e3;


%% Create data

z_m = 0; %[m] height difference
timeFrame = 3; %1 = short-term, 2 = mid-term, 3 = long-term

m_t_per_y_lin = [1e3 5e3 1e4 3e4 5e4 1e5 2e5 3e5 4e5 5e5 6e5 7e5 8e5 9e5 1e6 1.5e6 2e6 3e6 4e6 5e6 6e6 7e6 8e6 9e6 1e7 1.5e7 2e7];
L_km_lin = [10:10:100 150:50:2000]; 

%% Alg

[Terrain] = PipelineCostsforGrid(m_t_per_y_lin, L_km_lin, z_m, timeFrame, ...
    pipeline, finance, electricity, 1);

%% Save

save('PipelineModeling_withImpossibleCases.mat','Terrain')

for terr = 1:2
    for phase = 1:2
        Terrain(terr).Phase(phase).Data(Terrain(terr).Phase(phase).Data.LC_EUR_per_t == 0,:) = [];     
    end
end

save('PipelineModelingStruct.mat','Terrain')

%% Figures

m_kg_per_s_lin = m_t_per_y_lin.*1000./(365*24*3600);
[M, L] = meshgrid(m_kg_per_s_lin, L_km_lin);
M_Mt_per_y = M.*(365*24*3600)./1e9;
sizeML = size(M);
titleText = {'Offshore gas', 'Offshore dense'; 'Onshore gas', 'Onshore dense'};
load PipelineModeling_withImpossibleCases.mat

%LC cond & transp.

f = MakeFigPaper(1, 2, 2);
ax = gca;
box on; hold on;

tl = tiledlayout(2,2,"TileSpacing","compact",'Padding','compact');

for terr = 1:2
    for phase = 1:2

        LC = reshape(Terrain(terr).Phase(phase).Data.LC_EUR_per_t,sizeML);
        LC(LC == 0) = NaN;

        nexttile()

        contourf(M_Mt_per_y, L, LC, [1 2 3 4 5 10 15 20 25 30 40 50 100])
        title(titleText{terr,phase})
    end
end

xlabel(tl,'Mass flow [Mt/y]')
ylabel(tl,'Length [km]')
title(tl,'Levelized costs of CO_2 conditioned and transported [EUR/t]')
colormap jet
cbh = colorbar;
cbh.Layout.Tile = 'east';

% Save figure
nameFile = strcat(folder,'\Results\Pipeline\LCtot');
exportgraphics(f,append(nameFile,'.png'),'Resolution',300)
%saveas(f,'LCtot','epsc')

%Transport

f = MakeFigPaper(1, 2, 2);
ax = gca;
box on; hold on;

tl = tiledlayout(2,2,"TileSpacing","compact",'Padding','compact');

for terr = 1:2
    for phase = 1:2

        LC = reshape(Terrain(terr).Phase(phase).Data.LCtrans,sizeML);
        LC(LC == 0) = NaN;

        nexttile()

        contourf(M_Mt_per_y, L, LC, [0 1 2 3 4 5 10 15 20 25 30 40 50 100])
        title(titleText{terr,phase})
    end
end

xlabel(tl,'Mass flow [Mt/y]')
ylabel(tl,'Length [km]')
title(tl,'Levelized costs of CO_2 transported [EUR/t]')
colormap jet
cbh = colorbar;
cbh.Layout.Tile = 'east';

% Save figure
nameFile = strcat(folder,'\Results\Pipeline\LCtrans');
exportgraphics(f,append(nameFile,'.png'),'Resolution',300)

%Conditioned

f = MakeFigPaper(1, 2, 2);
ax = gca;
box on; hold on;

tl = tiledlayout(2,2,"TileSpacing","compact",'Padding','compact');

for terr = 1:2
    for phase = 1:2

        LC = reshape(Terrain(terr).Phase(phase).Data.LC_EUR_per_t,sizeML) - reshape(Terrain(terr).Phase(phase).Data.LCtrans,sizeML);
        LC(LC == 0) = NaN;

        nexttile()

        contourf(M_Mt_per_y, L, LC, 8:1:22)
        title(titleText{terr,phase})
    end
end

xlabel(tl,'Mass flow [Mt/y]')
ylabel(tl,'Length [km]')
title(tl,'Levelized costs of CO_2 conditioned [EUR/t]')
colormap jet
cbh = colorbar;
cbh.Layout.Tile = 'east';

% Save figure
nameFile = strcat(folder,'\Results\Pipeline\LCcond');
exportgraphics(f,append(nameFile,'.png'),'Resolution',300)

%Unitary costs transported

f = MakeFigPaper(1, 2, 2);
ax = gca;
box on; hold on;

tl = tiledlayout(2,2,"TileSpacing","compact",'Padding','compact');

for terr = 1:2
    for phase = 1:2

        LC = reshape(Terrain(terr).Phase(phase).Data.LCtrans,sizeML);
        LC(LC == 0) = NaN;

        nexttile()

        contourf(M_Mt_per_y, L, LC./L, 0:0.005:0.1)
        title(titleText{terr,phase})
    end
end

xlabel(tl,'Mass flow [Mt/y]')
ylabel(tl,'Length [km]')
title(tl,'Unitary costs of CO_2 transported [EUR/t/km]')
colormap jet
cbh = colorbar;
cbh.Layout.Tile = 'east';

% Save figure
nameFile = strcat(folder,'\Results\Pipeline\UCtrans');
exportgraphics(f,append(nameFile,'.png'),'Resolution',300)

f = MakeFigPaper(1, 2, 2);
ax = gca;
box on; hold on;

tl = tiledlayout(2,2,"TileSpacing","compact",'Padding','compact');

for terr = 1:2
    for phase = 1:2

        nexttile()

        LC = reshape(Terrain(terr).Phase(phase).Data.LCtrans,sizeML);
        LC(LC == 0) = NaN;

        colors = jet(length(L_km_lin));

        hold on
        for i = 1:length(L_km_lin)
            plot(M_Mt_per_y(i,:), LC(i,:)./L_km_lin(i), 'Color',colors(i,:))
        end
        hold off
        ylim([0 0.1])

        title(titleText{terr,phase})
    end
end

xlabel(tl,'Mass flow [Mt/y]')
ylabel(tl,'Unitary costs of CO_2 transported [EUR/t/km]')
legend(strcat(cellstr(num2str(L_km_lin')), ' km'))
colormap jet

% Save figure
nameFile = strcat(folder,'\Results\Pipeline\UCtrans_MF');
exportgraphics(f,append(nameFile,'.png'),'Resolution',300)

% title(tl,'Unitary costs of CO2 transported [EUR/t/km]')
% colormap jet

figure()
tl = tiledlayout(2,2);

for terr = 1:2
    for phase = 1:2
        nexttile()

        LC = reshape(Terrain(terr).Phase(phase).Data.LCtrans,sizeML);
        LC(LC == 0) = NaN;

        colors = jet(length(L_km_lin));

        hold on
        for i = 1:length(L_km_lin)
            semilogy(M(i,:), LC(i,:)./L_km_lin(i), 'Color',colors(i,:))
        end
        hold off
        %ylim([0 0.1])
        set(gca,'YScale','log')
        set(gca,'XScale','log')

    end
end
legend({'10 km'})
xlabel(tl,'Mass flow [kg/s]')
ylabel(tl,'Unitary costs of CO2 transported [EUR/t/km]')
% title(tl,'Unitary costs of CO2 transported [EUR/t/km]')
% colormap jet

%conditioning and transport
figure()
for terr = 1:2
    for phase = 1:2
        nexttile()

        LC_cond = reshape(Terrain(terr).Phase(phase).Data.LC_EUR_per_t,sizeML) - reshape(Terrain(terr).Phase(phase).Data.LCtrans,sizeML);
        
        LC_cond(LC_cond == 0) = NaN;

        colors = jet(length(L_km_lin));

        hold on
        for i = 1:length(L_km_lin)
            plot(M(i,:), LC_cond(i,:), 'Color',colors(i,:))
        end
        hold off
        %ylim([0 0.1])
        set(gca,'YScale','log')
        set(gca,'XScale','log')

    end
end
legend({'50 km'})
xlabel(tl,'Mass flow [kg/s]')
ylabel(tl,'Unitary costs of CO2 transported and conditioned [EUR/t/km]')
% title(tl,'Unitary costs of CO2 transported [EUR/t/km]')
% colormap jet






