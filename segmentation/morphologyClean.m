%% MORPHOLOGY CLEAN - Morphological Operations for Mask Cleaning
% Applies morphological operations to clean and refine binary masks
%
% Syntax:
%   cleanMask = morphologyClean(mask)
%   cleanMask = morphologyClean(mask, options)
%
% Inputs:
%   mask    - Binary mask to clean
%   options - Optional struct with fields:
%             .openRadius   - Disk radius for opening (default: 5)
%             .closeRadius  - Disk radius for closing (default: 10)
%             .minArea      - Minimum region area to keep (default: 500)
%
% Outputs:
%   cleanMask - Cleaned binary mask

function cleanMask = morphologyClean(mask, options)
    %% Default parameters
    if nargin < 2
        options = struct();
    end
    
    openRadius = getfield_default(options, 'openRadius', 3);
    closeRadius = getfield_default(options, 'closeRadius', 10);
    minArea = getfield_default(options, 'minArea', 50); % Level 3: Smallest grains saved
    
    %% Ensure mask is binary
    mask = mask > 0;
    
    %% Step 1: Opening - Remove small noise objects
    seOpen = strel('disk', openRadius);
    mask = imopen(mask, seOpen);
    
    %% Step 2: Closing - Fill small holes and gaps
    seClose = strel('disk', closeRadius);
    mask = imclose(mask, seClose);
    
    %% Step 3: Fill holes completely
    mask = imfill(mask, 'holes');
    
    %% Step 4: Remove small objects (RE-INTRODUCED)
    mask = bwareaopen(mask, minArea);
    
    %% Step 5: Smooth boundaries
    seSmooth = strel('disk', 3);
    mask = imclose(mask, seSmooth);
    
    %% Output
    cleanMask = mask;
end

%% Helper function to get field with default value
function value = getfield_default(s, field, default)
    if isfield(s, field)
        value = s.(field);
    else
        value = default;
    end
end
