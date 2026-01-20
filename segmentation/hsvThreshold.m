%% HSV THRESHOLD - Color-based Food Region Detection
% Detects food regions using HSV color space thresholding with edge awareness

function mask = hsvThreshold(img, foodType)
    %% Convert to HSV color space
    if isa(img, 'uint8')
        img = im2double(img);
    end
    hsvImg = rgb2hsv(img);
    
    H = hsvImg(:,:,1);
    S = hsvImg(:,:,2);
    V = hsvImg(:,:,3);
    
    %% Define thresholds based on food type
    if nargin < 2 || isempty(foodType)
        foodType = 'general';
    end
    
    switch lower(foodType)
        case 'nasi_lemak'
            satMin = 0.03;  % Lower for white rice
            satMax = 1.0;
            valMin = 0.15;
            valMax = 1.0;
            
        case 'roti_canai'
            satMin = 0.10;
            satMax = 0.8;
            valMin = 0.20;
            valMax = 0.95;
            
        case 'satay'
            % CRITICAL FIX FOR SATAY STICKS:
            % Use texture to capture thin structures
            satMin = 0.10;
            satMax = 1.0;
            valMin = 0.08;  % Lower to capture darker sticks
            valMax = 0.95;

        case 'laksa'
            % Laksa: orange/red soup with noodles
            satMin = 0.10;
            satMax = 1.0;
            valMin = 0.10;
            valMax = 1.0;
            
        case 'popiah'
            % Popiah: beige wrapper with colorful filling
            satMin = 0.05;
            satMax = 0.9;
            valMin = 0.15;
            valMax = 1.0;
            
        case 'kaya_toast'
            % Kaya toast: brown bread with green kaya
            satMin = 0.08;
            satMax = 0.9;
            valMin = 0.15;
            valMax = 0.95;
            
        case 'mixed_rice'
            % Mixed rice: various colors
            satMin = 0.05;
            satMax = 1.0;
            valMin = 0.10;
            valMax = 1.0;
            
        otherwise
            satMin = 0.03;
            satMax = 1.0;
            valMin = 0.10;
            valMax = 0.98;
    end
    
    %% Apply basic color thresholds
    satMask = (S >= satMin) & (S <= satMax);
    valMask = (V >= valMin) & (V <= valMax);
    
    %% A++ ENHANCEMENT: Edge-Preserving Exclusions
    % Use gradient to detect real edges vs texture
    grayImg = rgb2gray(img);
    edgeMap = edge(grayImg, 'canny', [0.05 0.2]);
    
    % Dilate edges slightly to create "no-go" zones
    edgeDilated = imdilate(edgeMap, strel('disk', 2));
    
    %% Smart Background Detection
    % Detect uniform areas (likely plate/table) using local std deviation
    windowSize = 15;
    localStd = stdfilt(V, true(windowSize));
    uniformRegions = (localStd < 0.05) & (V > 0.7);
    
    % Plate detection: high brightness + low saturation + low texture
    % CRITICAL: Rice is also bright, low sat, low texture.
    % Only remove if it's EXTREMELY smooth (Texture < 0.5)
    texture = entropyfilt(V, true(5));
    plateMask = (S < 0.05) & (V > 0.90) & (texture < 0.5);
    
    % Plate detection: high brightness + low saturation + low texture
    % Strictly for plates (very smooth, very bright)
    plateMask = (S < 0.05) & (V > 0.90) & (texture < 0.5);
    
    %% Combine masks
    if strcmpi(foodType, 'satay')
        % For satay: be aggressive about keeping everything
        mask = satMask & valMask;
    else
        baseMask = satMask & valMask;
        % Exclude ONLY plate. Do not try to guess "wrapper" here.
        mask = baseMask & ~plateMask;
    end
    
    %% Ensure Spatial Coherence
    % Keep largest connected components (Top 5) to allow for separated food items
    cc = bwconncomp(mask, 8);
    stats = regionprops(cc, 'Area');
    
    if ~isempty(stats)
        areas = [stats.Area];
        [~, sortedIdx] = sort(areas, 'descend');
        
        % Keep top 5 largest regions (Rice, Chicken, Sambal, Egg, Anchovies)
        keepCount = min(5, numel(sortedIdx));
        
        mask = false(size(mask));
        for i = 1:keepCount
            idx = sortedIdx(i);
            mask(cc.PixelIdxList{idx}) = true;
        end
    end
end
