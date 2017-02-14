function ml_kilosort(timeseries_path,firings_path,opts)

fprintf('Reading timeseries...\n');
timeseries=readmda(timeseries_path);

fprintf('Running kilosort...\n');
opts.base_tmp_fname=timeseries_path;
[times0,labels0]=run_kilosort(timeseries,opts);

fprintf('Writing firings output...\n');
firings=zeros(3,length(times0));
firings(2,:)=times0;
firings(3,:)=labels0;
writemda64(firings,firings_path);

