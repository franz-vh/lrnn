% Abstract class from which all inputs involving FSDD (Free Spoken Digit
% Dataset) inherit

classdef Fsdd < MultTrialDataSet
    
    % Public constants
    properties (Constant)
        Path = strcat(fileparts(mfilename('fullpath')),"/../../datasets/free-spoken-digit-dataset-master/recordings")
        Classifs = createEnum({'digits','subjects'}) % classifications enumeration
        DigitLbls = createEnum({'one','two','three','four','five','six','seven','eight','nine','zero'}) % digit labels enumeration
        SubjectLbls = createEnum({'george','jackson','lucas','nicolas','theo','yweweler'}) % subject labels enumeration
        AllLbls = {Fsdd.DigitLbls,Fsdd.SubjectLbls} % cell with all labels enumerations
    end
    
    % Public constants
    properties (SetAccess = protected)
        AllNLbls = InputData.setAllNLbls(Fsdd.AllLbls)
    end
    
    % Data initialization methods
    methods (Access = protected)
        % Read dataset
        function readDataset(obj)
            filenames = dir(fullfile(Fsdd.Path,"*.wav"));
            wdur = 0.02;
            fmax = 4000; % fs = 8000 , this is set here to avoid issues if different audios had different fs
            fstep = 100;
            trials = cell(1,length(filenames));
            for i = 1:length(filenames)
                [x,fs] = audioread(fullfile(Fsdd.Path,filenames(i).name));
                %soundsc(x,fs);
                if obj.Features
                    [~,~,~,trials{i}.x] = spectrogram(x,round(wdur*fs),round(wdur*fs/2),1/wdur:fstep:fmax,fs);
                else
                    trials{i}.x = x';
                end
                filenameparts = split(filenames(i).name,'_');
                trials{i}.lbl = zeros(length(Fsdd.AllLbls),1);
                trials{i}.lbl(Fsdd.Classifs.digits) = str2double(filenameparts{1});
                if trials{i}.lbl(Fsdd.Classifs.digits) == 0
                    trials{i}.lbl(Fsdd.Classifs.digits) = Fsdd.DigitLbls.zero;
                end
                trials{i}.lbl(Fsdd.Classifs.subjects) = find(strcmp(fieldnames(Fsdd.SubjectLbls),filenameparts{2}));
            end
            obj.Data.trials = trials;
        end
    end
    
end