%% EXTRACT REGION PROPS - Region Property Extraction
% Extracts comprehensive region properties from a binary mask
%
% Syntax:
%   props = extractRegionProps(mask)
%
% Inputs:
%   mask - Binary mask of food region
%
% Outputs:
%   props - Struct with region properties:
%           .Area         - Total area in pixels
%           .Perimeter    - Perimeter length
%           .Compactness  - 4*pi*Area / Perimeter^2 (circle = 1)
%           .Eccentricity - Ellipse eccentricity (circle = 0)
%           .BoundingBox  - [x, y, width, height]
%           .Centroid     - [x, y] center of mass
%           .AspectRatio  - Width / Height of bounding box
%           .Extent       - Area / BoundingBox area
%           .Solidity     - Area / Convex hull area

function props = extractRegionProps(mask)
    %% Initialize output
    props = struct();
    
    %% Handle empty mask
    if isempty(mask) || ~any(mask(:))
        props.Area = 0;
        props.Perimeter = 0;
        props.Compactness = 0;
        props.Eccentricity = 0;
        props.BoundingBox = [0, 0, 0, 0];
        props.Centroid = [0, 0];
        props.AspectRatio = 1;
        props.Extent = 0;
        props.Solidity = 0;
        return;
    end
    
    %% Get region properties using regionprops
    stats = regionprops(mask, 'Area', 'Perimeter', 'BoundingBox', ...
                        'Centroid', 'Eccentricity', 'Extent', 'Solidity');
    
    %% Handle multiple regions (combine them)
    if length(stats) > 1
        % Sum areas
        props.Area = sum([stats.Area]);
        
        % Sum perimeters
        props.Perimeter = sum([stats.Perimeter]);
        
        % Use overall bounding box
        allBoxes = vertcat(stats.BoundingBox);
        minX = min(allBoxes(:, 1));
        minY = min(allBoxes(:, 2));
        maxX = max(allBoxes(:, 1) + allBoxes(:, 3));
        maxY = max(allBoxes(:, 2) + allBoxes(:, 4));
        props.BoundingBox = [minX, minY, maxX - minX, maxY - minY];
        
        % Area-weighted centroid
        totalArea = props.Area;
        centroids = vertcat(stats.Centroid);
        areas = [stats.Area];
        props.Centroid = sum(centroids .* areas', 1) / totalArea;
        
        % Area-weighted eccentricity
        props.Eccentricity = sum([stats.Eccentricity] .* areas) / totalArea;
        
        % Recalculate extent
        bbArea = props.BoundingBox(3) * props.BoundingBox(4);
        props.Extent = props.Area / bbArea;
        
        % Area-weighted solidity
        props.Solidity = sum([stats.Solidity] .* areas) / totalArea;
        
    else
        props.Area = stats.Area;
        props.Perimeter = stats.Perimeter;
        props.BoundingBox = stats.BoundingBox;
        props.Centroid = stats.Centroid;
        props.Eccentricity = stats.Eccentricity;
        props.Extent = stats.Extent;
        props.Solidity = stats.Solidity;
    end
    
    %% Calculate derived properties
    % Compactness (isoperimetric quotient)
    if props.Perimeter > 0
        props.Compactness = 4 * pi * props.Area / (props.Perimeter^2);
    else
        props.Compactness = 0;
    end
    
    % Aspect ratio
    if props.BoundingBox(4) > 0
        props.AspectRatio = props.BoundingBox(3) / props.BoundingBox(4);
    else
        props.AspectRatio = 1;
    end
end
