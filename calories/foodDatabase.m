%% FOOD DATABASE - Malaysian Food Calorie Database
% Returns nutritional information for Malaysian hawker foods
%
% Syntax:
%   db = foodDatabase()
%   foodInfo = foodDatabase(foodClass)
%
% Inputs:
%   foodClass - Optional specific food class to look up
%
% Outputs:
%   db/foodInfo - Struct with nutritional information per food class
%                 Fields: name, baseCalories, protein, carbs, fat,
%                         referenceServing, description
%
% Source: Malaysian Food Composition Database (MyFCD) and nutritional studies

function db = foodDatabase(foodClass)
    %% Define the complete database
    foodData = {
        % name, baseCalories, protein(g), carbs(g), fat(g), serving, description
        'nasi_lemak',   650, 15, 85, 28, '1 plate (250g)', 'Coconut rice with sambal, anchovies, peanuts, egg, and cucumber';
        'roti_canai',   300, 6, 36, 15, '1 piece (100g)', 'Flaky flatbread, typically served with dhal or curry';
        'satay',        200, 18, 8, 12, '5 sticks (150g)', 'Grilled meat skewers with peanut sauce';
        'laksa',        500, 15, 55, 25, '1 bowl (400g)', 'Spicy noodle soup with coconut milk or tamarind base';
        'popiah',       185, 5, 25, 7, '1 roll (100g)', 'Fresh spring roll with vegetables and tofu';
        'kaya_toast',   300, 6, 42, 12, '2 slices (80g)', 'Toasted bread with coconut jam and butter';
        'mixed_rice',   620, 20, 75, 28, '1 plate + 3 dishes', 'White rice with assorted dishes'
    };
    
    %% Create struct array
    numFoods = size(foodData, 1);
    db = struct();
    
    for i = 1:numFoods
        fieldName = foodData{i, 1};
        db.(fieldName).name = strrep(foodData{i, 1}, '_', ' ');  % Human-readable name
        db.(fieldName).displayName = capitalizeWords(strrep(foodData{i, 1}, '_', ' '));
        db.(fieldName).baseCalories = foodData{i, 2};
        db.(fieldName).protein = foodData{i, 3};
        db.(fieldName).carbs = foodData{i, 4};
        db.(fieldName).fat = foodData{i, 5};
        db.(fieldName).referenceServing = foodData{i, 6};
        db.(fieldName).description = foodData{i, 7};
    end
    
    %% Return specific food if requested
    if nargin > 0 && ~isempty(foodClass)
        foodClass = lower(strrep(foodClass, ' ', '_'));
        if isfield(db, foodClass)
            db = db.(foodClass);
        else
            warning('foodDatabase:UnknownFood', 'Food class not found: %s', foodClass);
            db = struct('name', foodClass, 'baseCalories', 400, ...
                        'protein', 10, 'carbs', 50, 'fat', 15, ...
                        'referenceServing', 'Unknown', ...
                        'description', 'Unknown food item');
        end
    end
end

%% Helper function to capitalize words
function str = capitalizeWords(str)
    words = strsplit(str, ' ');
    for i = 1:length(words)
        if ~isempty(words{i})
            words{i}(1) = upper(words{i}(1));
        end
    end
    str = strjoin(words, ' ');
end
