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
    
    %% A++ FIX 1: ULTRA BACKGROUND KILLER (Texture + Intensity + Spatial)
    % 1. Smooth White/Beige (Plate/Paper): Low S, Low Texture
    isSmooth = (localStd < 0.10);
    isPale = (S < 0.20) & (V > 0.40);
    platePaperMask = isSmooth & isPale;
    
    % 2. Deep Shadow (External): Very low V
    shadowMask = (V < 0.15);
    
    % 3. Highlights (Blown out): Very high V, Zero Sat
    highlightMask = (V > 0.95) & (S < 0.05);

    % Combine and dilate
    bgMask = platePaperMask | shadowMask | highlightMask;
    bgMask = imdilate(bgMask, strel('disk', 3));

    %% A++ FIX 2: EXTREME RICE RESCUE (Connectivity-Based)
    % We define RICE as: [Smooth/Pale] pixels that are [CONNECTED] to [Colorful Food].
    
    % Initial Food Core (High Saturation, Colorful)
    coreFood = (S > 0.30) & (V > 0.20) & ~bgMask;
    
    % Candidate Rice (Any smooth/pale area that might be rice)
    riceCandidates = (S < 0.45) & (V > 0.25) & bgMask;
    
    if any(coreFood(:)) && any(riceCandidates(:))
        % Merge all rice that is near the core food
        % Use a large proximity (40 pixels) to bridge gaps in nasi lemak boxes
        foodZone = imdilate(coreFood, strel('disk', 40));
        rescuedRice = riceCandidates & foodZone;
        
        % Remove rescued pixels from background mask
        bgMask(rescuedRice) = false;
    end

    %% Combine Final Masks
    % A++ CRITICAL FIX: Explicitly INJECT the rescued rice
    % Previously, satMask would kill the rice even if we rescued it from bgMask.
    % Now we force-add it.
    baseMask = satMask & valMask & ~bgMask;
    
    if exist('rescuedRice', 'var')
        baseMask = baseMask | rescuedRice;
        fprintf('  Debug: Injected %d rescued rice pixels.\n', sum(rescuedRice(:)));
    end
    
    %% A++ FIX 3: SPATIAL LOCKDOWN (Middle Focus)
    % Force kill anything in the outer 30% of the image
    [rows, cols] = size(S);
    [X, Y] = meshgrid(1:cols, 1:rows);
    distMap = sqrt(((X - (cols/2))/(cols/2)).^2 + ((Y - (rows/2))/(rows/2)).^2);
    
    % Hard kill in corners (dist > 0.7)
    % Unless it's extremely high saturation (Entrees touching edge)
    mask = baseMask;
    cornerKill = (distMap > 0.7) & (S < 0.5);
    mask(cornerKill) = false;
    
    %% Ensure Spatial Coherence
    % Keep largest connected components (Top 3) to allow for separated food items
    cc = bwconncomp(mask, 8);
    stats = regionprops(cc, 'Area');
    
    if ~isempty(stats)
        areas = [stats.Area];
        [~, sortedIdx] = sort(areas, 'descend');
        
        % Keep top 3 largest regions (e.g. Rice, Chicken, Sambal)
        keepCount = min(3, numel(sortedIdx));
        keepIdx = sortedIdx(1:keepCount);
        
        mask = false(size(mask));
        for i = keepIdx
            mask(cc.PixelIdxList{i}) = true;
        end
    end
end
