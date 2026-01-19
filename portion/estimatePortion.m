%% ESTIMATE PORTION - Portion Size Estimation from Segmented Mask
% Estimates the portion size relative to a standard serving
%
% Syntax:
%   portionRatio = estimatePortion(mask)
%   portionRatio = estimatePortion(mask, foodClass)
%   portionRatio = estimatePortion(mask, foodClass, img)
%   [portionRatio, portionLabel, areaPixels] = estimatePortion(mask, foodClass, img)
%
% Inputs:
%   mask      - Binary mask of food region
%   foodClass - Food class name for class-specific reference (optional)
%   img       - RGB image for color density analysis (optional, A++ feature)
%
% Outputs:
%   portionRatio - Ratio relative to standard serving (1.0 = standard)
%   portionLabel - 'Small', 'Medium', 'Large', or 'Extra Large'
%   areaPixels   - Actual food area in pixels

function [portionRatio, portionLabel, areaPixels] = estimatePortion(mask, foodClass, img)
    %% Input validation
    if isempty(mask)
        portionRatio = 0;
        portionLabel = 'None';
        areaPixels = 0;
        return;
    end
    
    if nargin < 2 || isempty(foodClass)
        foodClass = 'general';
    end
    if nargin < 3
        img = [];
    end
    
    %% Calculate food area
    areaPixels = sum(mask(:));
    
    %% Get reference area for this food class
    referenceArea = getReferenceArea(foodClass, size(mask));
    
    %% Calculate portion ratio
    portionRatio = areaPixels / referenceArea;
    
    %% Apply shape-based density adjustment (A++ Enhancement)
    % Compact, solid masks indicate denser food portions
    % Scattered, fragmented masks indicate lighter portions
    try
        stats = regionprops(mask, 'Solidity', 'Extent', 'Perimeter', 'Area');
        if ~isempty(stats)
            % Average solidity (how filled vs hollow the shape is)
            avgSolidity = mean([stats.Solidity]);
            % Extent (how much of bounding box is filled)
            avgExtent = mean([stats.Extent]);
            
            % Compactness factor: higher = denser food, lower = scattered/light
            % Solidity: 1.0 = solid filled shape, <0.5 = lots of holes
            % Extent: 1.0 = fills bounding box, <0.5 = sparse
            compactnessFactor = (avgSolidity * 0.6 + avgExtent * 0.4);
            
            % Map compactness to density multiplier (0.85 to 1.15 range)
            densityMultiplier = 0.85 + (compactnessFactor * 0.30);
            
            % Apply density adjustment
            portionRatio = portionRatio * densityMultiplier;
        end
    catch
        % If regionprops fails, continue without adjustment
    end
    %% A++ ENHANCEMENT: Color-based density analysis for complex foods
    % Analyzes color composition to estimate calorie density
    % High curry/sambal (red/orange) = higher calories
    % High vegetables (green) = lower calories
    complexFoods = {'nasi_lemak', 'mixed_rice', 'laksa'};
    if ~isempty(img) && any(strcmpi(foodClass, complexFoods))
        try
            % Convert to HSV for color analysis
            if isa(img, 'uint8')
                img = im2double(img);
            end
            hsvImg = rgb2hsv(img);
            H = hsvImg(:,:,1);
            S = hsvImg(:,:,2);
            
            % Get hue values within food mask only
            foodH = H(mask);
            foodS = S(mask);
            totalPixels = numel(foodH);
            
            if totalPixels > 100
                % Red/Orange pixels (Sambal, Curry) - High Calorie
                redOrangeMask = (foodH < 0.1 | foodH > 0.9) & foodS > 0.25;
                redRatio = sum(redOrangeMask) / totalPixels;
                
                % Green pixels (Vegetables) - Low Calorie  
                greenMask = (foodH >= 0.2 & foodH <= 0.45) & foodS > 0.2;
                greenRatio = sum(greenMask) / totalPixels;
                
                % Calculate color density factor (0.9 to 1.2 range)
                colorDensity = 1.0;
                colorDensity = colorDensity + (redRatio * 0.2);   % +20% for curry/sambal
                colorDensity = colorDensity - (greenRatio * 0.1); % -10% for vegetables
                colorDensity = max(0.9, min(1.2, colorDensity));
                
                % Apply color density adjustment
                portionRatio = portionRatio * colorDensity;
            end
        catch
            % Color analysis failed, continue without
        end
    end
    
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
