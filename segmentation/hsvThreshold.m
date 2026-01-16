%% HSV THRESHOLD - Color-based Food Region Detection
% Detects food regions using HSV color space thresholding
%
% Syntax:
%   mask = hsvThreshold(img)
%   mask = hsvThreshold(img, foodType)
%
% Inputs:
%   img      - RGB image
%   foodType - Optional food class for optimized thresholds
%
% Outputs:
%   mask - Binary mask where true = likely food region

function mask = hsvThreshold(img, foodType)
    %% Convert to HSV color space
    if isa(img, 'uint8')
        img = im2double(img);
    end
    hsvImg = rgb2hsv(img);
    
    H = hsvImg(:,:,1);  % Hue [0-1]
    S = hsvImg(:,:,2);  % Saturation [0-1]
    V = hsvImg(:,:,3);  % Value [0-1]
    
    %% Define thresholds based on food type
    if nargin < 2 || isempty(foodType)
        foodType = 'general';
    end
    
    switch lower(foodType)
        case 'nasi_lemak'
            % Nasi lemak: white rice, green leaves, red sambal
            % Exclude very low saturation (table) and very dark
            satMin = 0.08;
            satMax = 1.0;
            valMin = 0.15;
            valMax = 1.0;
            
        case 'roti_canai'
            % Roti canai: golden brown flatbread
            satMin = 0.10;
            satMax = 0.8;
            valMin = 0.20;
            valMax = 0.95;
            
        case 'satay'
            % Satay: brown meat on sticks
            satMin = 0.15;
            satMax = 1.0;
            valMin = 0.10;
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
            
        otherwise  % 'general'
            % General thresholds suitable for most foods
            satMin = 0.05;   % Minimum saturation (exclude gray backgrounds)
            satMax = 1.0;    % Maximum saturation
            valMin = 0.10;   % Minimum brightness (exclude very dark)
            valMax = 0.98;   % Maximum brightness (exclude overexposed white)
    end
    
    %% Apply thresholds
    satMask = (S >= satMin) & (S <= satMax);
    valMask = (V >= valMin) & (V <= valMax);
    
    %% Additional: Exclude white/gray backgrounds
    % White has low saturation AND high value
    whiteBackgroundMask = (S < 0.1) & (V > 0.85);
    
    % Very dark regions (shadows, edges)
    darkMask = V < 0.05;
    
    %% Combine masks
    mask = satMask & valMask & ~whiteBackgroundMask & ~darkMask;
    
    %% Additional color-based food detection
    % Foods typically have warmer hues (yellow, orange, red, brown)
    % H values: Red ~0 or ~1, Orange ~0.1, Yellow ~0.15, Green ~0.33, Blue ~0.6
    
    % Create masks for typical food colors
    redMask = (H < 0.05 | H > 0.95) & S > 0.2;
    orangeMask = (H >= 0.02 & H <= 0.12) & S > 0.2;
    yellowMask = (H >= 0.10 & H <= 0.20) & S > 0.15;
    brownMask = (H >= 0.02 & H <= 0.15) & (S >= 0.2 & S <= 0.7) & (V >= 0.1 & V <= 0.6);
    greenMask = (H >= 0.20 & H <= 0.45) & S > 0.15;
    
    foodColorMask = redMask | orangeMask | yellowMask | brownMask | greenMask;
    
    %% Final mask combines saturation/value with food colors
    % Be more inclusive - use OR to catch more food regions
    mask = mask | (foodColorMask & valMask);
end
