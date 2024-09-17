% Shifted images from the Deep Learning and Unsupervised Feature Learning
% course image dataset input class

classdef ShiftedImgInput < DataSet
    
    % Public constants
    properties (Constant)
        Path = strcat(fileparts(mfilename('fullpath')),"/../../datasets/IMAGES.mat")
        Classifs = createEnum({}) % classifications enumeration
        AllLbls = {} % cell with all labels enumerations
        Range = [0.1,0.9]
    end
    
    % Public constants
    properties (SetAccess = protected)
        AllNLbls = InputData.setAllNLbls(ShiftedImgInput.AllLbls)
    end
    
    % Main public methods
    methods
        % Hard restart (new object types)
        function hardRestart(obj,params)
            if any(params.resplit)
                if isfield(params,'splitprops')
                    obj.Data.splitprops = params.splitprops;
                end
                resplitsets = [obj.Data.splitsets{params.resplit}];
                cumprops = cumsum(obj.Data.splitprops(params.resplit));
                cumprops = cumprops/cumprops(end);
                dslimits = randi(length(resplitsets)) + [0,round(cumprops*length(resplitsets))] - 1;
                resplitposs = find(params.resplit);
                for i = 1:nnz(params.resplit)
                    obj.Data.splitsets{resplitposs(i)} = resplitsets(mod(dslimits(i)+1:dslimits(i+1),length(resplitsets))+1);
                end
            end
            obj.restart(params);
        end
    end
    
    % Data initialization methods
    methods (Access = protected)
        % Input data initialization
        function initData(obj,params)
            obj.readDataset(params);
            obj.Size = prod(obj.Dims);
            obj.normData();
            obj.splitInSets();
        end
        % Read dataset
        function readDataset(obj,params)
            obj.Data.images = open(ShiftedImgInput.Path).IMAGES;
            obj.Dims = [8,8];
            if isfield(params,'dims')
                obj.Dims = params.dims;
            end
            obj.Data.maxshift = 3;%min(obj.Dims)/2;
            obj.Data.nframes = 80;
            obj.Data.range = ceil(1 + obj.Data.maxshift*obj.Data.nframes)*ones(2,2);
            obj.Data.range(2,:) = floor(size(obj.Data.images,1,2) - obj.Data.maxshift*obj.Data.nframes - obj.Dims);
        end
        % Normalize input between 0.1 and 0.9
        function normData(obj)
            pstd = 3 * std(obj.Data.images(:));
            obj.Data.images = obj.Data.images - mean(obj.Data.images,[1,2]);
            obj.Data.images = max(min(obj.Data.images, pstd), -pstd) / pstd;
            obj.Data.images = (obj.Data.images + 1) * 0.4 + 0.1;
        end
        % Split data into sets
        function splitInSets(obj)
            obj.Data.splitsets = cell(1,length(obj.Data.splitprops));
            dslimits = [0,round(cumsum(obj.Data.splitprops)*size(obj.Data.images,3))];
            for i = 1:length(obj.Data.splitprops)
                obj.Data.splitsets{i} = dslimits(i)+1:dslimits(i+1);
            end
        end
    end
    
    % Data obtention methods
    methods (Access = protected)
        % Get sample sequence
        function [smpls,lbls] = getSequence(obj)
            img = obj.Data.images(:,:,obj.Stt.splitset(randi(length(obj.Stt.splitset))));
            pos0 = [randi([obj.Data.range(1,1),obj.Data.range(2,1)]),randi([obj.Data.range(1,2),obj.Data.range(2,2)])];
            angle = rand*2*pi;
            shift = rand*obj.Data.maxshift*[sin(angle),cos(angle)];
            smpls = zeros(obj.Size,obj.Data.nframes+1);
            lbls = zeros(length(ShiftedImgInput.AllLbls),obj.Data.nframes+1);
            for i = 1:obj.Data.nframes
                pos = pos0 + round((i-1)*shift);
                sample = img(pos(1):pos(1)+obj.Dims(1)-1,pos(2):pos(2)+obj.Dims(2)-1);
                smpls(:,i) = sample(:);
            end
            smpls(:,end) = 0.5*ones(obj.Size,1);
        end
    end
    
end