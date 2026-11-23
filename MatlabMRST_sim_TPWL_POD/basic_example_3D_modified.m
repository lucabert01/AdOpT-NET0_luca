%% Basic 3D simulation of a two-phase water and gas system
% This example shows the creation of a simple 3D grid (not VE) in mrst from
% scratch and basic usage of the the TwoPhaseWaterGasModel for modelling a
% CO2-H2O system. CO2 is injected into an intially brine filled reservoir
% from an injection well at the bottom of the reservoir. We model two years
% of injection and two years post injection. CO2 properties are taken from 
% co2lab's tabulated co2props() function.


function [simData, muco2, WI,rhoc] = basic_example_3D_modified(sim4rom)
    %% Load modules
    mrstModule add co2lab ad-core ad-props ad-blackoil mrst-gui;
    
    simData = loadSimData();
    if sim4rom 
        nSim = simData.nNewSim;
    else 
        nSim = simData.nTraining; % number of training run simulations
    end
    nt = simData.nt;
    states = cell(1, nSim);
    wellSol = cell(1, nSim);
    
    
    %% Grid and rock
    
    % Make 3D prism grid with cells refined towards the top
    z_res = 13;%13;   % number of cells in depth direction (z) with 11*19*19 and lp=60 it works
    l_res = 23;%23;   % number of cells in lateral direction (x and y)
    
    
    depth = 1500; % depth of aquifer top surface
    H = 100;      % full thickness of aquifer
    L = 30000;     % horizontal extent 
    
    zcoord = linspace(0, 1, z_res+1).^1.5 * H; % thinner cells towards the top
    
    dL = linspace(0, 1, ceil(l_res/2)+1).^1.5; % narrower cells towards well
    lcoord = [-fliplr(dL(2:end)), dL(2:end)] * L;
    
    G = tensorGrid(lcoord, lcoord, zcoord, ...
                   'depthz', repmat(depth, 1, (l_res+1) * (l_res+1)));
    G = computeGeometry(G);
    
    rock.poro = repmat(0.25, G.cells.num, 1);
    rock.perm = repmat(250*milli*darcy, G.cells.num, 1);

    
    for s = 1:nSim    
        
        %% Initial state
        gravity on; % tell MRST to turn on gravity
        g = gravity; % get the gravity vector
        rhow = 1000; % density of brine 
        initState.pressure = rhow * g(3) * G.cells.centroids(:,3); % initial pressure
        initState.s = repmat([1, 0], G.cells.num, 1); % initial saturations
        initState.sGmax = initState.s(:,2); % initial max. gas saturation (hysteresis)
        
        %% Fluid model
        co2     = CO2props(); % load sampled tables of co2 fluid properties
        p_ref   = 15 * mega * Pascal; % choose reference pressure
        t_ref   = 25 + 273.15; % choose reference temperature, in Kelvin
        rhoc    = co2.rho(p_ref, t_ref); % co2 density at ref. press/temp
        cf_co2  = co2.rhoDP(p_ref, t_ref) / rhoc; % co2 compressibility
        cf_wat  = 0; % brine compressibility (zero)
        cf_rock = 4.35e-5 / barsa; % rock compressibility
        muw     = 8e-4 * Pascal * second; % brine viscosity
        muco2   = co2.mu(p_ref, t_ref) * Pascal * second; % co2 viscosity
        
        mrstModule add ad-props; % The module where initSimpleADIFluid is found
        
        % Use function 'initSimpleADIFluid' to make a simple fluid object
        fluid = initSimpleADIFluid('phases', 'WG'           , ...
                                   'mu'  , [muw, muco2]     , ...
                                   'rho' , [rhow, rhoc]     , ...
                                   'pRef', p_ref            , ...
                                   'c'   , [cf_wat, cf_co2] , ...
                                   'cR'  , cf_rock          , ...
                                   'n'   , [2 2]);
        
        % Change relperm curves
        srw = 0.27;
        src = 0.20;
        fluid.krW = @(s) fluid.krW(max((s-srw)./(1-srw), 0));
        fluid.krG = @(s) fluid.krG(max((s-src)./(1-src), 0));
        
        % Add capillary pressure curve
        pe = 5 * kilo * Pascal;
        pcWG = @(sw) pe * sw.^(-1/2);
        fluid.pcWG = @(sg) pcWG(max((1-sg-srw)./(1-srw), 1e-5)); %@@
        
        
        %% Wells
        
        % Well cell indices in 'global' grid: 48, 48, 6:10
        wc_global = false(G.cartDims);
        center_ix = ceil(l_res/2);
        
        wc_global(center_ix, center_ix, z_res) = true; % bottom center cell
        wc = find(wc_global(G.cells.indexMap));
        
        % plot well cells
        plotGrid(G, 'facecolor', 'none', 'edgealpha', 0.1);
        plotGrid(G, wc, 'facecolor', 'red');
        
        
        % Calculate the injection rate
        injRate = 1 * mega * 1e3 / year / fluid.rhoGS; % target injection
        
        % % Start with empty set of wells
        W = [];
    
        % Add a well to the set
        W = addWell(W, G, rock, wc, ...
                    'refDepth', G.cells.centroids(wc, 3), ... % BHP reference depth
                    'type', 'rate', ...  % inject at constant rate
                    'val', injRate, ... % volumetric injection rate
                    'comp_i', [0 1]);    % inject CO2, not water
    
        % W = [];
        W.WI = W.WI*1000;
        WI = W.WI;
        %% Boundary conditions
        
        % Start with an empty set of boundary faces
        bc = [];
        
        % identify all vertical faces
        vface_ind = (G.faces.normals(:,3) == 0);
        
        % identify all boundary faces (having only one cell neighbor
        bface_ind = (prod(G.faces.neighbors, 2) == 0);
        
        % identify all lateral boundary faces
        bc_face_ix = find(vface_ind & bface_ind);
        
        % identify cells neighbouring lateral boundary baces
        bc_cell_ix = sum(G.faces.neighbors(bc_face_ix,:), 2);
        
        % lateral boundary face pressure equals pressure of corresponding cell
        p_face_pressure = initState.pressure(bc_cell_ix); 
        
        % Add hydrostatic pressure conditions to open boundary faces
        bc = addBC(bc, bc_face_ix, 'pressure', p_face_pressure, 'sat', [1, 0]);
        
        %% Schedule
        schedule = createSchedule(simData, bc, W);
    
        %% Model
        model = GenericBlackOilModel(G, rock, fluid, 'oil', false);
        model = model.validateModel();
        
        %% Simulate
        
        output_handler = ResultHandler();
        if sim4rom
            [wellSol{s}, states{s}, report] = simulateScheduleAD(initState, model, schedule.(simData.newSimNames(s))); % Luca
        else 
            [wellSol{s}, states{s}, report] = simulateScheduleAD(initState, model, schedule.(simData.trainingNames(s))); % Luca
        end
    
        if ~sim4rom % retrieve the Jacobians only for training simulations
            jacForward = cell(1, nt);
            jacReverse = cell(1, nt);
            
            if s == 1 % we save the jacobians only of the fisrt simulation for now
                % If reverse = true: compute dg_{i+1}/dx_i, otherwise compute
                % dg_{i+1}/dx_{i+1}
                reverse = false;
                % Index of timestep to compute.
                for ix = 1:nt
                    if ix == 1
                        state0 = initState;
                    else
                        state0 = states{s}{ix-1};
                    end
                    state0 = model.validateState(state0);
                    state = states{s}{ix};
                    if reverse
                        [state0, primaryVars] = model.getReverseStateAD(state0);
                    end
                    % Make sure that well changes are accounted for
                    ctrl = schedule.(simData.trainingNames(s)).control(schedule.(simData.trainingNames(s)).step.control(ix));
                    dt = schedule.(simData.trainingNames(s)).step.val(ix);
                    [forces, fstruct] = model.getDrivingForces(ctrl);
                    fstruct.src = [];
                    model = model.validateModel(fstruct);
                    
                    problem = model.getAdjointEquations(state0, state, dt, fstruct, 'reverseMode', reverse);
                    [jacForward{ix}, b] = problem.getLinearSystem();
                end
                %%
                reverse = true;
                % Index of timestep to compute.
                for ix = 1:nt
                    if ix == 1
                        state0 = initState;
                    else
                        state0 = states{s}{ix-1};
                    end
                    state0 = model.validateState(state0);
                    state = states{s}{ix};
                    if reverse
                        [state0, primaryVars] = model.getReverseStateAD(state0);
                    end
                    % Make sure that well changes are accounted for
                    ctrl = schedule.(simData.trainingNames(s)).control(schedule.(simData.trainingNames(s)).step.control(ix));
                    dt = schedule.(simData.trainingNames(s)).step.val(ix);
                    [forces, fstruct] = model.getDrivingForces(ctrl);
                    fstruct.src = [];
                    model = model.validateModel(fstruct);
                    
                    problem = model.getAdjointEquations(state0, state, dt, fstruct, 'reverseMode', reverse);
                    [jacReverse{ix}, b] = problem.getLinearSystem();
                end
        
                % Create the subfolder if it doesn't exist
                if ~exist('TrainingData', 'dir')
                    mkdir('TrainingData');
                end
                
                save(fullfile('TrainingData', 'schedule.mat'), 'schedule');
                save(fullfile('TrainingData', 'jacForward.mat'), 'jacForward');
                save(fullfile('TrainingData', 'jacReverse.mat'), 'jacReverse');
                save(fullfile('TrainingData', 'muco2.mat'), 'muco2');
                save(fullfile('TrainingData', 'WI.mat'), 'WI');
                save(fullfile('TrainingData', 'rhoc.mat'), 'rhoc');
                save(fullfile('TrainingData', 'fluid.mat'), 'fluid');
            end
        end



    end 
    
    
    cellTopWell = W.cells(1);
    time = cumsum(schedule.(simData.trainingNames(1)).step.val)./year;
    
    if ~sim4rom
    
        % Create the subfolder if it doesn't exist
        if ~exist('TrainingData', 'dir')
            mkdir('TrainingData');
        end
        
        % Specify the full file paths for saving
        cellTopWell_file = fullfile('TrainingData', 'cellTopWell.mat');
        statesTraining_file = fullfile('TrainingData', 'statesTraining.mat');
        %statesNew_file = fullfile('TrainingData', 'statesNew.mat');
        wellSol_file = fullfile('TrainingData', 'wellSol.mat');
        time_file = fullfile('TrainingData', 'time.mat');
        
        % Save the variables in their respective files
        save(cellTopWell_file, 'cellTopWell');
        save(statesTraining_file, 'states');
        save(wellSol_file, 'wellSol');
        save(time_file, 'time');
        
        
        
        %% Plotting some results
        
        bhp = cell(1, nSim);
        pressureMax = cell(1, nSim);
        pTopWell = cell(1, nSim);
        sTopWell = cell(1, nSim);
        krgTopWell = cell(1, nSim);
        krwTopWell = cell(1, nSim);
        
        
        for s = 1:nSim
            for t=1:nt
                pressureMax{s}(t) = max(states{s}{t}.pressure);
                bhp{s}(t) = wellSol{s}{t}.bhp;
                pTopWell{s}(t) = states{s}{t}.pressure(cellTopWell);
                sTopWell{s}(t) = states{s}{t}.s(cellTopWell);
                krgTopWell{s}(t) = fluid.krG(sTopWell{s}(t));
                krwTopWell{s}(t) = fluid.krG(1-sTopWell{s}(t));
        
            end
            figure()
            plot(time, bhp{s}(:))
            hold on
            plot(time, pTopWell{s}(:))
            title("Training run n." + s)
            legend('bhp', 'pressure top well')
            xlabel('time')
            ylabel('pressure')
        end
    else
        nt = height(simData.inputSchedulesNewSim);
        pbh_fullScaleNewSim = {}; 
        for sim = 1:nSim
            pbh_fullScaleNewSim.(simData.newSimNames{sim}).pTopWell = ones(nt,1);
            pbh_fullScaleNewSim.(simData.newSimNames{sim}).pReal = ones(nt,1);
            for t =1:nt
                pbh_fullScaleNewSim.(simData.newSimNames{sim}).pTopWell(t) = states{sim}{t}.pressure(cellTopWell);
                pbh_fullScaleNewSim.(simData.newSimNames{sim}).pReal(t) = wellSol{sim}{t}.bhp;
            end
        end
        save('pbh_fullScaleNewSim.mat', 'pbh_fullScaleNewSim');
    end

%%
% q = zeros(dt, 1);
% mob = (krgTopWell{1}./muco2 + krwTopWell{1}./muw); % multiphase mobility
% mobApprox = 1/muco2;
% WI = W.WI;
% for i = 1:dt
%     q(i) = wellSol{1,1}{i,1}.val;
%     p_bhp_est(i) = pTopWell{1}(i) + q(i)/(WI*mob);
%     p_bhp(i) = wellSol{1,1}{i,1}.bhp;
% end
% 
% 
% figure()
% plot(p_bhp)
% hold on
% plot(p_bhp_est)
% legend('wellSol.bhp', 'estimated from inflow relationship')
% ylabel('bhp')
% xlabel('time')
% 
% %%
% figure()
% plot(pTopWell{s}(:), bhp{s}(:), 'Marker','.', 'LineStyle','none')
% xlabel('pTopWell')
% ylabel('bhp')

%% check units schdule.control

% totInjection = sum(schedule.control.W.val.*time)/inj_rate

%%
% <html>
% <p><font size="-1">
% Copyright 2009-2022 SINTEF Digital, Mathematics & Cybernetics.
% </font></p>
% <p><font size="-1">
% This file is part of The MATLAB Reservoir Simulation Toolbox (MRST).
% </font></p>
% <p><font size="-1">
% MRST is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
% </font></p>
% <p><font size="-1">
% MRST is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% </font></p>
% <p><font size="-1">
% You should have received a copy of the GNU General Public License
% along with MRST.  If not, see
% <a href="http://www.gnu.org/licenses/">http://www.gnu.org/licenses</a>.
% </font></p>
% </html>