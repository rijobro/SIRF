function prep_data = preprocess_acquisition_data(input_data)
% Apply NoiseAdjust, AsymmetricEchoAdjust and RemoveROOversampling gadgets
prep_data = input_data.process([ ...
    {'NoiseAdjustGadget'} ...
    {'AsymmetricEchoAdjustROGadget'} ...
    {'RemoveROOversamplingGadget'} ...
    ]);
end