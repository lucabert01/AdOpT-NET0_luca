%% Proper Orthogonal Decomposition of CO2 storage


function [basisPODpress, basisPODsat, lp, ls] = performPOD(states, cutoffPress, cutoffSat, lp, ls)

    % if lp and ls are not provided in the function, the cutoffs are used to
    % determine them
    %% Pressure and saturation
    
    % Prepare snapshot matrix
    nSim = length(states);
    nt_sim = numel(states{1});
    nt = length(states{1})*nSim; % number of timesteps
    nb = length(states{1}{1,1}.pressure); % number of blocks in the grid
    
    % Convert States (cell array) to a 2D matrix
    snapshotsPress = zeros(nb, nt); % SnapShots matrix
    snapshotsSat = zeros(nb, nt);

    for sim = 1:nSim
        for i = 1:nt_sim
            pressure = states{sim}{i}.pressure;
            sat = states{sim}{i}.s(:,2); 
            snapshotsPress(:,i+(sim-1)*nt_sim) = pressure;
            snapshotsSat(:,i+(sim-1)*nt_sim) = sat;
    
        end
    end
    % Subtract the temporal mean from each row
    Apress = snapshotsPress - repmat(mean(snapshotsPress), nb, 1); 
    Asat = snapshotsSat - repmat(mean(snapshotsSat), nb, 1);

   
    [Upress,Spress ,Vpress] = svd(Apress,"econ"); %/sqrt(nt-1)
    [Usat,Ssat ,Vsat] = svd(Asat, "econ");
    
    checkCorrectnessP = mean(mean(Apress - Upress * Spress *Vpress'));
    checkCorrectnessS = mean(mean(Asat - Usat * Ssat *Vsat'));

    assert(checkCorrectnessP < 10e-8, 'Error in the Singular Value Decomposition of pressure snapshots')
    assert(checkCorrectnessS < 10e-8, 'Error in the Singular Value Decomposition of saturation snapshots')

    
    %% Matrix reduction
    
    % Rescale the singular values as a fraction in range [0;1]
    Spress = diag(Spress)./sum(diag(Spress)); 
    Ssat = diag(Ssat)./sum(diag(Ssat)); 

    if nargin ~= 3 && nargin ~= 5
        error("The inputs for the function performPOD must be either 3 or 5")
    end
    if nargin == 4
        % Find intex of cut-off for pressure and saturation POD
        for lp = 1:nt % lp is the cut off index for the pressure POD
            sumSingularValues = sum(Spress(1:lp));
           if sumSingularValues >= cutoffPress
                break
            end
        end
        
        for ls = 1:nt 
            sumSingularValues = sum(Ssat(1:ls));
            if sumSingularValues >= cutoffSat
                break
            end
        end
    end

    basisPODpress = Upress(:,1:lp); % the columns of this matrix are the left singular vectors (basis of POD)
    basisPODsat = Usat(:,1:ls); % the columns of this matrix are the left singular vectors (basis of POD)

% %% Visulaization
% 
% imagesc(Apress);
% colorbar
% xlabel('Timestep')
% ylabel('Gridblock')
% title('Zero-centred snapshots (A)')
% 
% 
% 
% figure()
% imagesc(Upress)
% cmap = crameri('oslo'); % Get the current colormap
% cmap = flipud(cmap); % flip colormap colors
% colormap(cmap);% Update the colormap
% colorbar;% Show colorbar
% title('Spatial modes matrix (U)')
% 
% 
% figure()
% imagesc(Spress)
% cmap = crameri('oslo'); % Get the current colormap
% cmap = flipud(cmap); % flip colormap colors
% colormap(cmap);% Update the colormap
% colorbar;% Show colorbar
% title('Singular values matrix (S)')
% 
% figure()
% imagesc(Vpress)
% cmap = crameri('oslo'); % Get the current colormap
% cmap = flipud(cmap); % flip colormap colors
% colormap(cmap);% Update the colormap
% colorbar;% Show colorbar
% title('Temporal modes matrix (V)')
% 
% figure()
% bar(diag(Spress))
% title('Singular values (S)')

% 
% 
% figure()
% imagesc(basisPOD_p)
% cmap = crameri('bilbao'); % Get the current colormap
% colormap(cmap);% Update the colormap
% colorbar;% Show colorbar
% title('Reduced spatial modes matrix (Ureduced)')
% 
% 
% figure()
% imagesc(Sreduced)
% cmap = crameri('bilbao'); % Get the current colormap
% colormap(cmap);% Update the colormap
% colorbar;% Show colorbar
% title('Reduced singular values matrix (Sreduced)')
% 
% figure()
% imagesc(Vreduced)
% cmap = crameri('bilbao'); % Get the current colormap
% colormap(cmap);% Update the colormap
% colorbar;% Show colorbar
% title('Reduced temporal modes matrix (Vreduced)')







