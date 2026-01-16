%% ESTIMATE PORTION - Portion Size Estimation from Segmented Mask
% Estimates the portion size relative to a standard serving
%
% Syntax:
%   portionRatio = estimatePortion(mask)
%   portionRatio = estimatePortion(mask, foodClass)
%   [portionRatio, portionLabel, areaPixels] = estimatePortion(mask, foodClass)
%
% Inputs:
%   mask      - Binary mask of food region
%   foodClass - Food class name for class-specific reference (optional)
%
% Outputs:
%   portionRatio - Ratio relative to standard serving (1.0 = standard)
%   portionLabel - 'Small', 'Medium', 'Large', or 'Extra Large'
%   areaPixels   - Actual food area in pixels

function [portionRatio, portionLabel, areaPixels] = estimatePortion(mask, foodClass)
    %% Input validation
    if isempty(mask)
        portionRatio = 0;
        portionLabel = 'None';
        areaPixels = 0;
        return;
    end
    
    if nargin < 2
        foodClass = 'general';
    end
    
    %% Calculate food area
    areaPixels = sum(mask(:));
    
    %% Get reference area for this food class
    referenceArea = getReferenceArea(foodClass, size(mask));
    
    %% Calculate portion ratio
    portionRatio = areaPixels / referenceArea;
    
    % Clamp to reasonable range
    portionRatio = max(0.1, min(2.5, portionRatio));
    
    %% Determine portion label
    if portionRatio < 0.6
        portionLabel = 'Small';
    elseif portionRatio < 0.9
        portionLabel = 'Medium-Small';
    elseif portionRatio < 1.1
        portionLabel = 'Medium';
    elseif portionRatio < 1.4
        portionLabel = 'Medium-Large';
    elseif portionRatio < 1.8
        portionLabel = 'Large';
    else
        portionLabel = 'Extra Large';
    end
end

%% Get reference area for food class
function referenceArea = getReferenceArea(foodClass, imageSize)
    % Reference areas are calibrated for 512x512 images
    % These represent typical food coverage in a standard serving photo
    
    totalPixels = imageSize(1) * imageSize(2);
    
    switch lower(foodClass)
        case 'nasi_lemak'
            % Nasi lemak typically covers 40-50% of plate photo
            referenceRatio = 0.45;
            
        case 'roti_canai'
            % Roti canai (1-2 pieces) covers 35-45%
            referenceRatio = 0.40;
            
        case 'satay'
            % Satay sticks cover 30-40%
            referenceRatio = 0.35;
            
        case 'laksa'
            % Bowl of laksa covers 50-60%
            referenceRatio = 0.55;
            
        case 'popiah'
            % Popiah rolls cover 25-35%
            referenceRatio = 0.30;
            
        case 'kaya_toast'
            % Kaya toast (2 slices) covers 20-30%
            referenceRatio = 0.25;
            
        case 'mixed_rice'
            % Mixed rice plate covers 45-55%
            referenceRatio = 0.50;
            
        otherwise  % 'general'
            % Default assumption
            referenceRatio = 0.40;
    end
    
    referenceArea = totalPixels * referenceRatio;
end
