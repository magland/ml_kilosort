function [times0,labels0]=run_kilosort(timeseries,opts)

if nargin<1, run_kilosort_test; return; end;

if (~isfield(opts,'num_clusters')) opts.num_clusters=20; end;
if (~isfield(opts,'samplerate')) opts.samplerate=30000; end;

mfilepath=fileparts(mfilename('fullpath'));
addpath([mfilepath,'/../KiloSort']);
addpath([mfilepath,'/../KiloSort/utils']);
addpath([mfilepath,'/../KiloSort/preProcess']);
addpath([mfilepath,'/../KiloSort/initialize']);
addpath([mfilepath,'/../KiloSort/mainLoop']);
addpath([mfilepath,'/../KiloSort/mergesplits']);
addpath([mfilepath,'/../KiloSort/finalPass']);

M=size(timeseries,1); % Number of channels

fprintf('Writing the raw data as a temporary .dat file...\n');
raw_fname=[timeseries,'.run_kilosort.tmp.dat'];
temp_wh=[raw_fname,'-temp_wh.dat'];
write_raw_timeseries(timeseries,raw_fname);

fprintf('Getting the ks options...\n');
ks_ops=get_ks_ops(M,raw_fname,opts);

fprintf('Run the normal Kilosort processing on the simulated data...\n');
[rez, DATA, uproj] = preprocessData(ks_ops); % preprocess data and extract spikes for initializationinst
rez                = fitTemplates(rez, DATA, uproj);  % fit templates iteratively
rez                = fullMPMU(rez, DATA);% extract final spike times (overlapping extraction)

fprintf('Retrieve the times and labels');
times0=rez.st3(:,1);
labels0=rez.st3(:,2);

fprintf('Removing temporary files...\n');
delete(raw_fname);
delete(temp_wh);

function ops=get_ks_ops(M,raw_fname,opts)

% CM.Nchannels = M;
% CM.connected = true(M, 1);
% CM.chanMap   = 1:M;
% CM.chanMap0ind = CM.chanMap - 1;
% CM.xcoords   = ones(M,1);
% CM.ycoords   = [1:M]'; %'
% CM.kcoords   = ones(M,1); % grouping of channels (i.e. tetrode groups)
% save('chanMap.tmp.mat');
% ops.chanMap             = 'chanMap.tmp.mat'; % make this file using createChannelMapFile.m		
ops.chanMap=[];
ops.Nchan=M;
ops.NchanTOT=M;
ops.fs=opts.samplerate;

ops.GPU                 = 0; % whether to run this code on an Nvidia GPU (much faster, mexGPUall first)		

ops.parfor              = 1; % whether to use parfor to accelerate some parts of the algorithm		
ops.verbose             = 1; % whether to print command line progress		
ops.showfigures         = 0; % whether to plot figures during optimization		
		
ops.datatype            = 'dat';  % binary ('dat', 'bin') or 'openEphys'		
ops.fbinary             = raw_fname; % will be created for 'openEphys'		
ops.fproc               = temp_wh; % residual from RAM of preprocessed data		
%ops.root                = fpath; % 'openEphys' only: where raw files are		
% define the channel map as a filename (string) or simply an array		

% ops.chanMap = 1:ops.Nchan; % treated as linear probe if unavailable chanMap file		

ops.Nfilt               = opts.num_clusters;  % number of clusters to use (2-4 times more than Nchan, should be a multiple of 32)     		
ops.nNeighPC            = []; %12; % visualization only (Phy): number of channnels to mask the PCs, leave empty to skip (12)		
ops.nNeigh              = 16; % visualization only (Phy): number of neighboring templates to retain projections of (16)		
		
% options for channel whitening		
ops.whitening           = 'full'; % type of whitening (default 'full', for 'noSpikes' set options for spike detection below)		
ops.nSkipCov            = 1; % compute whitening matrix from every N-th batch (1)		
ops.whiteningRange      = 32; % how many channels to whiten together (Inf for whole probe whitening, should be fine if Nchan<=32)		
		
ops.criterionNoiseChannels = 0.2; % fraction of "noise" templates allowed to span all channel groups (see createChannelMapFile for more info). 		

% other options for controlling the model and optimization		
ops.Nrank               = 3;    % matrix rank of spike template model (3)		
ops.nfullpasses         = 6;    % number of complete passes through data during optimization (6)		
ops.maxFR               = 20000;  % maximum number of spikes to extract per batch (20000)		
ops.fshigh              = 200;   % frequency for high pass filtering		
% ops.fslow             = 2000;   % frequency for low pass filtering (optional)
ops.ntbuff              = 64;    % samples of symmetrical buffer for whitening and spike detection		
ops.scaleproc           = 200;   % int16 scaling of whitened data		
ops.NT                  = 128*1024+ ops.ntbuff;% this is the batch size (try decreasing if out of memory) 		
% for GPU should be multiple of 32 + ntbuff		
		
% the following options can improve/deteriorate results. 		
% when multiple values are provided for an option, the first two are beginning and ending anneal values, 		
% the third is the value used in the final pass. 		
ops.Th               = [4 10 10];    % threshold for detecting spikes on template-filtered data ([6 12 12])		
ops.lam              = [5 5 5];   % large means amplitudes are forced around the mean ([10 30 30])		
ops.nannealpasses    = 4;            % should be less than nfullpasses (4)		
ops.momentum         = 1./[20 400];  % start with high momentum and anneal (1./[20 1000])		
ops.shuffle_clusters = 1;            % allow merges and splits during optimization (1)		
ops.mergeT           = .1;           % upper threshold for merging (.1)		
ops.splitT           = .1;           % lower threshold for splitting (.1)		
		
% options for initializing spikes from data		
ops.initialize      = 'no';    %'fromData' or 'no'		
ops.spkTh           = -6;      % spike threshold in standard deviations (4)		
ops.loc_range       = [3  1];  % ranges to detect peaks; plus/minus in time and channel ([3 1])		
ops.long_range      = [30  6]; % ranges to detect isolated peaks ([30 6])		
ops.maskMaxChannels = 5;       % how many channels to mask up/down ([5])		
ops.crit            = .65;     % upper criterion for discarding spike repeates (0.65)		
ops.nFiltMax        = 10000;   % maximum "unique" spikes to consider (10000)		
		
% load predefined principal components (visualization only (Phy): used for features)		
%dd                  = load('PCspikes2.mat'); % you might want to recompute this from your own data		
%ops.wPCA            = dd.Wi(:,1:7);   % PCs 		
		
% options for posthoc merges (under construction)		
ops.fracse  = 0.1; % binning step along discriminant axis for posthoc merges (in units of sd)		
ops.epu     = Inf;		
		
ops.ForceMaxRAMforDat   = 20e9; % maximum RAM the algorithm will try to use; on Windows it will autodetect.


function write_raw_timeseries(X,fname)
F=fopen(fname,'wb');
fwrite(F,X(:),'int16');
fclose(F);

function [X,firings_true]=create_simulated_data(M,N,K,noise_level)
M=5;
T=800;
synth_opts.upsamplefac=25;
refr=10;

samplerate=30000;
firing_rates=3*ones(1,K);
amp_variations=ones(2,K); amp_variations(1,:)=0.9; amp_variations(2,:)=1.1; %amplitude variation

opts.geom_spread_coef1=0.4;
opts.upsamplefac=synth_opts.upsamplefac;
waveforms=synthesize_random_waveforms(M,T,K,opts);

% force equal on all channels
% for k=K:K
%     waveforms(1,:,k)=waveforms(end,:,k);
%     waveforms(2,:,k)=waveforms(end,:,k);
%     waveforms(3,:,k)=waveforms(end,:,k);
%     waveforms(4,:,k)=waveforms(end,:,k);
%     waveforms(5,:,k)=waveforms(end,:,k);
% end;

%figure; ms_view_templates(waveforms);

% events/sec * sec/timepoint * N
populations=ceil(firing_rates/samplerate*N);
times=zeros(1,0);
labels=zeros(1,0);
ampls=zeros(1,0);
for k=1:K
    times0=rand(1,populations(k))*(N-1)+1;
    times0=enforce_refractory_period(times0,refr/1000*samplerate);
    times=[times,times0];
    labels=[labels,k*ones(size(times0))];
    amp1=amp_variations(1,k);
    amp2=amp_variations(2,k);
    ampls=[ampls,rand_uniform(amp1,amp2,size(times0)).*ones(size(times0))];
end;

firings_true=zeros(3,length(times));
firings_true(2,:)=times;
firings_true(3,:)=labels;

X=synthesize_timeseries(waveforms,N,times,labels,ampls,synth_opts);
X=X+randn(size(X))*noise_level;
%writemda32(X,'raw.mda');
%writemda64(firings_true,'firings_true.mda');
firings_true=zeros(3,length(times));
firings_true(2,:)=times;
firings_true(3,:)=labels;

function times0=enforce_refractory_period(times0,refr)
if (length(times0)==0) return; end;
times0=sort(times0);
done=0;
while ~done
    diffs=times0(2:end)-times0(1:end-1);
    diffs=[diffs,inf]; %hack to make sure we handle the last one
    inds0=find((diffs(1:end-1)<=refr)&(diffs(2:end)>refr)); %only first violator in every group
    if (length(inds0)>0)
        times0(inds0)=-1; %kind of a hack, what's the better way?
        times0=times0(times0>=0);
    else
        done=1;
    end;
end

function X=rand_uniform(a,b,sz)
X=rand(sz);
X=a+(b-a)*X;

function run_kilosort_test

mfilepath=fileparts(mfilename('fullpath'));
addpath([mfilepath,'/run_kilosort_test']);

fprintf('Creating simulated data...\n');
M=4;
N=1e7;
K=3;
noise_level=1;
[X,firings_true]=create_simulated_data(M,N,K,noise_level);

fprintf('Running kilosort...\n');
opts.num_clusters=16;
[times0,labels0]=run_kilosort(X,opts);

try
    fprintf('Writing .mda output files...\n');
    firings=zeros(3,length(times0));
    firings(2,:)=times0;
    firings(3,:)=labels0;
    writemda(X,'simulated_timeseries.mda');
    writemda(firings,'firings_ks.mda');
    writemda(firings_true,'firings_true.mda');
catch err
    error('Unable to write output because writemda was not found. Run mouuntainlab_setup.m first.');
end;
