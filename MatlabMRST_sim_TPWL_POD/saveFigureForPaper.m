function saveFigureForPaper(figHandle, filename, varargin)
    % Function to save a figure with specified settings for publication
    %
    % Inputs:
    %   - figHandle: Handle to the figure
    %   - filename: Name of the file (without extension)
    %   - varargin: Optional input for font size (default is 12)

    % Directory for saving
    filePathResults = "C:\Users\0954659\OneDrive - Universiteit Utrecht\Documents\PhD Luca\Papers\Geological CO2 storage\Paper\Figures\";

    % Set figure size
    width = 432;  
    height = 288; 
    set(figHandle, 'Units', 'Points', 'Position', [0, 0, width, height]);
    set(figHandle, 'PaperUnits', 'Points', 'PaperSize', [width, height]);

    % Ensure high-quality vector rendering
    set(figHandle, 'Renderer', 'painters');

    % Determine font size: Use default (12) if not provided
    if ~isempty(varargin)
        fontSize = varargin{1};
    else
        fontSize = 12; % Default font size
    end

    % Set font size and font name for all text objects
    set(findall(figHandle, '-property', 'FontSize'), 'FontSize', fontSize);
    set(findall(figHandle, '-property', 'FontName'), 'FontName', 'Arial');

    % Save the figure in vector formats
    saveas(figHandle, fullfile(filePathResults, filename + ".pdf"));
    saveas(figHandle, fullfile(filePathResults, filename + ".jpg"));
    figure(figHandle);
end
